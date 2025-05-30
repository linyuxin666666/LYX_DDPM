import jittor as jt
from jittor import nn, Module
import os
import math
from abc import ABC, abstractmethod
from jittor import transform
from jittor.dataset.mnist import MNIST
from tqdm import tqdm 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt

if jt.has_cuda:
    jt.flags.use_cuda = 1
else:
    jt.flags.use_cuda = 0

def timestep_embedding(timesteps, dim, max_period=10000):
    half=dim//2
    freqs = jt.exp(
        -math.log(max_period) * jt.arange(start=0, end=half, dtype=jt.float32) / half
    )  #生成一个归一化张量
    args = timesteps[:, None].float() * freqs[None]#广播机制拓宽timestep维度，变成batch行half列的矩阵
    embedding = jt.concat([jt.cos(args), jt.sin(args)], dim=-1)#将正弦值和余弦值沿着half维度拼接起来得到embedding
    if dim % 2:#增强代码健壮性，如果dim是奇数的话，直接在最后一维添加一列0
        embedding = jt.concat([embedding, jt.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

# test_timesteps = jt.array([10, 50, 100]).float32() # 示例时间步
# test_dim = 64  # 示例嵌入维度

#     # 2. 调用函数
# result = timestep_embedding(test_timesteps, test_dim)
# print(result)

def norm_layer(channels):
      return nn.GroupNorm(32, channels) #将通道数分成32组，每组进行归一化



class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
     
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0

        self.norm = norm_layer(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def execute(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))#张量形状变为[B,3C,H,W]
        q, k, v = qkv.reshape(B*self.num_heads, -1, H*W).chunk(3, dim=1)#将qkv张量沿着第二维度拆分成三个张量，每个张量形状为[B*self.num_heads,C,H*W]
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        # attn = jt.einsum("bct,bcs->bts", q * scale, k * scale)#计算注意力权重，形状为[B*self.num_heads,H*W,H*W]
        #没有einsum，自己实现一下注意力机制的矩阵乘法
        q1 = q * scale
        k1 = k * scale
        q1_permuted = q1.permute(0, 2, 1)# (B*num_heads, head_dim, H*W) -> (B*num_heads, H*W, head_dim)
        attn = jt.matmul(q1_permuted, k1)
        attn = attn.softmax(dim=-1)
        h = jt.matmul(v, attn.permute(0, 2, 1)) # h = jt.einsum("bts,bcs->bct", attn, v)
        h = h.reshape(B, -1, H, W) #形状为[B,C,H,W]
        h = self.proj(h)#形状为[B,C,H,W]
        return h + x
    

class Upsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        else:
            self.conv = nn.Identity()
    def execute(self, x):
        x = jt.nn.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x
            

class Downsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)#因为stride=2，所以x的尺寸会缩小一半
        else:
            self.op = nn.AvgPool2d(stride=2)
    def execute(self, x):
        return self.op(x)
    
class TimestepBlock(nn.Module):
    @abstractmethod
    def execute(self, x, t):
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):#是否嵌入时间步的标志，即是否继承了TimestepBlock
        def execute(self, x, t):
          for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, t)
            else:
                x = layer(x)
          return x
class SiLU(nn.Module):#jittor没有SiLU，所以自己定义一个
    def __init__(self):
        super().__init__()

    def execute(self, x):
        return x * jt.sigmoid(x)

class ResidualBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            norm_layer(in_channels),
            SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        # pojection for time step embedding
        self.time_emb = nn.Sequential(
            SiLU(),
            nn.Linear(time_channels, out_channels)#time_channels是model_channels的四倍,用线性层转换一下通道数使其和x能正常相加
        )

        self.conv2 = nn.Sequential(
            norm_layer(out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)#如果输入通道和输出通道不符，需要作调整使得残差相加能起作用
        else:
            self.shortcut = nn.Identity()
    def execute(self, x, t):
        h = self.conv1(x)
        h += self.time_emb(t)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)

class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        dropout=0,
        channel_mult=(1, 2, 2, 2),
        conv_resample=True,
        num_heads=4
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads

        # time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # down blocks
        self.down_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1))
        ])
        down_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(ch, mult * model_channels, time_embed_dim, dropout)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_chans.append(ch)
            if level != len(channel_mult) - 1:  # don't use downsample for the last stage
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
                down_block_chans.append(ch)
                ds *= 2

        # middle block
        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(ch, ch, time_embed_dim, dropout),
            AttentionBlock(ch, num_heads=num_heads),
            ResidualBlock(ch, ch, time_embed_dim, dropout)
        )

        # up blocks
        self.up_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResidualBlock(
                        ch + down_block_chans.pop(),
                        model_channels * mult,
                        time_embed_dim,
                        dropout
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            norm_layer(ch),
            SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
        )
    def execute(self, x: jt.Var, timesteps: jt.Var):
       
        hs = []
        # down stage
        h: jt.Var = x
        t: jt.Var = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        for module in self.down_blocks:
            h = module(h, t)
            hs.append(h)
        # middle stage
        h = self.middle_block(h, t)
        # up stage
        for module in self.up_blocks:
            cat_in = jt.concat([h, hs.pop()], dim=1)
            h = module(cat_in, t)
        return self.out(h)
