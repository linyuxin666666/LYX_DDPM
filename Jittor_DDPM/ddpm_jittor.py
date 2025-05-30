import jittor as jt
from jittor import nn, Module
import os
import math
from abc import ABC, abstractmethod
from jittor import transform
from jittor.dataset.mnist import MNIST
from tqdm import tqdm # 添加 tqdm 导入
from PIL import Image 
import numpy as np 
import time # <--- Added for timing epochs
import matplotlib.pyplot as plt # <--- Added for plotting
# 优先使用GPU（如果可用），否则使用CPU
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

def linear_beta_schedule(timesteps):#生成α序列
   
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return jt.linspace(beta_start, beta_end, timesteps).float64()

class GaussianDiffusion:
    def __init__(
        self,
        timesteps=1000,
        beta_schedule='linear'
    ):
        self.timesteps = timesteps

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas

        self.alphas = 1. - self.betas
        self.alphas_cumprod = jt.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = jt.nn.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        # 计算q(x_t | x_{t-1}) 
        self.sqrt_alphas_cumprod = jt.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jt.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = jt.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = jt.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = jt.sqrt(1.0 / self.alphas_cumprod - 1)

        # 计算q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # 下面：log计算被剪裁，因为后验方差在扩散链的开始时为0
        self.posterior_log_variance_clipped = jt.log(self.posterior_variance.clamp(1e-20, float('inf')))
        self.posterior_mean_coef1 = self.betas *jt.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * jt.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

    def _extract(self, a: jt.Var, t: jt.Var, x_shape):
        # get the param of given timestep t
        batch_size = t.shape[0]
        out = a.gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    def q_sample(self, x_start: jt.Var, t: jt.Var, noise=None):
        # forward diffusion (using the nice property): q(x_t | x_0)
        if noise is None:
            noise = jt.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def q_mean_variance(self, x_start: jt.Var, t: jt.Var):
        # Get the mean and variance of q(x_t | x_0).
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_start: jt.Var, x_t: jt.Var, t: jt.Var):
        # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(self, x_t: jt.Var, t: jt.Var, noise: jt.Var):
        # compute x_0 from x_t and pred noise: the reverse of `q_sample`
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_mean_variance(self, model, x_t: jt.Var, t: jt.Var, clip_denoised=True):
        # compute predicted mean and variance of p(x_{t-1} | x_t)
        # predict noise using model
        pred_noise = model(x_t, t)
        # get the predicted x_0: different from the algorithm2 in the paper
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = jt.clamp(x_recon, -1., 1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance

    @jt.no_grad()
    def p_sample(self, model, x_t: jt.Var, t: jt.Var, clip_denoised=True):
        # denoise_step: sample x_{t-1} from x_t and pred_noise
        # predict mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t, clip_denoised=clip_denoised)
        noise = jt.randn_like(x_t)
        # no noise when t == 0
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        # compute x_{t-1}
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img

    @jt.no_grad()
    def sample(self, model: nn.Module, image_size, batch_size=8, channels=3):
        # denoise: reverse diffusion
        shape = (batch_size, channels, image_size, image_size)
        # start from pure noise (for each example in the batch)
        img = jt.randn(shape)  # x_T ~ N(0, 1)
        imgs = []
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            t = jt.full((batch_size,), i, dtype=jt.int64)
            img = self.p_sample(model, img, t)
            imgs.append(img.numpy())
        return imgs

    def train_losses(self, model, x_start: jt.Var, t: jt.Var):
        # compute train losses
        noise = jt.randn_like(x_start)  # random noise ~ N(0, 1)
        x_noisy = self.q_sample(x_start, t, noise=noise)  # x_t ~ q(x_t | x_0)
        predicted_noise = model(x_noisy, t)  # predict noise from noisy image
        loss = nn.mse_loss(noise, predicted_noise)
        return loss

batch_size = 64
timesteps = 500


transform = transform.Compose([
    transform.Gray(num_output_channels=1),
    transform.ToTensor(),
    transform.ImageNormalize(mean=[0.5], std=[0.5])
])

# use MNIST dataset
dataset = MNIST(train=True, download=True, transform=transform)
train_loader = dataset.set_attrs(batch_size=batch_size, shuffle=True)
# define model and diffusion

model = UNetModel(
    in_channels=1,
    model_channels=96,
    out_channels=1,
    channel_mult=(1, 2, 2),
    attention_resolutions=[]
)


optimizer = jt.optim.Adam(model.parameters(), lr=5e-4)
gaussian_diffusion = GaussianDiffusion(timesteps=timesteps)



# --- 日志记录设置 ---
LOG_DIR = "training_logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

STEP_LOSS_LOG_FILE = os.path.join(LOG_DIR, "step_losses.txt")
EPOCH_STATS_LOG_FILE = os.path.join(LOG_DIR, "epoch_stats.txt") # 这个文件的内容会改变
STEP_LOSS_CURVE_FILE = os.path.join(LOG_DIR, "step_loss_curve.png")

all_step_losses_for_plot = []
current_global_step = 0

print(f"每个step的Loss将记录到: {STEP_LOSS_LOG_FILE}")
print(f"每个epoch的统计数据将记录到: {EPOCH_STATS_LOG_FILE}") # 提示信息不变
print(f"Loss曲线图将保存到: {STEP_LOSS_CURVE_FILE}")
# --- 日志记录设置结束 ---

# --- 新增：计算模型总参数量 ---
total_params = 0
for param in model.parameters():
    total_params += param.numel() # numel() 获取参数的元素数量
print(f"模型总参数量: {total_params:,}") # 使用逗号分隔符，更易读
# --- 模型参数量计算结束 ---


# train
epochs = 10

with open(STEP_LOSS_LOG_FILE, 'w') as step_loss_file, \
     open(EPOCH_STATS_LOG_FILE, 'w') as epoch_stats_file:
    
    # --- 新增：在 epoch_stats.txt 文件开头写入模型总参数量 ---
    epoch_stats_file.write(f"Model Total Parameters: {total_params:,}\n")
    epoch_stats_file.write("-" * 50 + "\n") # 分隔线
    epoch_stats_file.write("Epoch,Total Duration (s),Avg Step Time (s)\n") # CSV风格的表头
    # --- 模型总参数量写入结束 ---

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()

        print(f"\n开始 Epoch {epoch + 1}/{epochs}")
        
        epoch_total_steps = 0
        for step, (images, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Steps")):
            optimizer.zero_grad()
            t = jt.randint(0, timesteps, (images.shape[0],)).long()
            loss = gaussian_diffusion.train_losses(model, images, t)
            optimizer.step(loss)

            loss_val = loss.item()
            step_loss_file.write(f"step{current_global_step + 1}: {loss_val:.6f}\n")
            all_step_losses_for_plot.append(loss_val)
            current_global_step += 1
            epoch_total_steps +=1

            if step % 200 == 0:
                print(f"Epoch: {epoch+1}, Step: {step}, Loss: {loss_val:.4f}")
        
        # --- Epoch结束，记录统计信息 (修改部分) ---
        epoch_end_time = time.time()
        epoch_duration_total = epoch_end_time - epoch_start_time
        avg_step_time_epoch = epoch_duration_total / epoch_total_steps if epoch_total_steps > 0 else 0
        
        # 不再记录显存，而是记录参数量（参数量在文件开头已记录一次，这里主要是耗时）
        stats_log_line = (f"{epoch + 1}," # Epoch编号
                          f"{epoch_duration_total:.2f}," # 总耗时
                          f"{avg_step_time_epoch:.4f}\n") # 平均step耗时
        epoch_stats_file.write(stats_log_line)
        
        # 控制台打印的信息也相应调整
        print(f"Epoch {epoch + 1} 完成. "
              f"总耗时: {epoch_duration_total:.2f}s, "
              f"平均Step耗时: {avg_step_time_epoch:.4f}s")
        # --- Epoch统计信息记录结束 ---

    print("\n所有训练周期完成。")

# ... (后续绘制Loss曲线和保存模型的代码保持不变) ...

# --- 保存训练好的模型 (这部分逻辑您原来应该有) ---
model_save_path = os.path.join(LOG_DIR, "ddpm_jittor_mnist_final.ckpt")
jt.save(model.state_dict(), model_save_path)
print(f"模型已成功保存到: {model_save_path}")

print("\n脚本执行完毕。")

# --- 加载模型并进行推理 ---
# 指定模型参数 (确保与训练时一致)
model = UNetModel(
    in_channels=1,
    model_channels=96,
    out_channels=1,
    channel_mult=(1, 2, 2),
    attention_resolutions=[]
)

# 指定你的checkpoint文件路径
checkpoint_path = "ddpm_jittor_mnist.ckpt" # <--- 修改这里为你的 .ckpt 文件路径

# 加载模型权重
print(f"正在从 {checkpoint_path} 加载模型...")
model.load_state_dict(jt.load(checkpoint_path))
print("模型加载成功！")



gaussian_diffusion = GaussianDiffusion(timesteps=timesteps)


batch_size = 64
generated_images = gaussian_diffusion.sample(model, 28, batch_size=batch_size, channels=1)
# generated_images: [timesteps, batch_size=64, channels=1, height=28, width=28]

# generate new images
fig = plt.figure(figsize=(12, 12), constrained_layout=True)
gs = fig.add_gridspec(8, 8)

imgs = generated_images[-1].reshape(8, 8, 28, 28)
for n_row in range(8):
    for n_col in range(8):
        f_ax = fig.add_subplot(gs[n_row, n_col])
        f_ax.imshow((imgs[n_row, n_col]+1.0) * 255 / 2, cmap="gray")
        f_ax.axis("off")

output_folder_name = "8_8img"

if not os.path.exists(output_folder_name):
    os.makedirs(output_folder_name)

file_name = "generated_8x8_grid_from_visualization.png" # 您可以自定义文件名
full_save_path = os.path.join(output_folder_name, file_name)

# 确保 matplotlib.pyplot 已导入为 plt (通常在文件顶部 import matplotlib.pyplot as plt)
plt.savefig(full_save_path)
plt.close(fig) # 保存后关闭图像，释放内存

print(f"基于显示的8x8网格图像已保存到: {full_save_path}")


# show the denoise steps
fig = plt.figure(figsize=(12, 12), constrained_layout=True)
nrows = 16
gs = fig.add_gridspec(nrows, 16)
for n_row in range(nrows):
    for n_col in range(16):
        f_ax = fig.add_subplot(gs[n_row, n_col])
        t_idx = (timesteps // 16) * n_col if n_col < 15 else -1
        img = generated_images[t_idx][n_row].reshape(28, 28)
        f_ax.imshow((img+1.0) * 255 / 2, cmap="gray")
        f_ax.axis("off")
# 1. 定义新的输出文件夹名称和文件名
new_output_folder = "denoise_grid_output"
grid_image_filename = "denoising_steps_16x16.png"

# 2. 确保新的输出文件夹存在 (需要提前 import os)
if not os.path.exists(new_output_folder):
    os.makedirs(new_output_folder)
    print(f"已创建新目录: {new_output_folder}")

# 3. 构建完整的文件保存路径
path_to_save_grid = os.path.join(new_output_folder, grid_image_filename)

# 4. 保存当前的图像 (fig 对象) 到指定路径 (需要提前 import matplotlib.pyplot as plt)
plt.savefig(path_to_save_grid)

# 5. 保存后关闭图像以释放内存
plt.close(fig) # 'fig' 是前面代码中 plt.figure() 创建的对象

# 6. 打印保存成功的消息
print(f"16x16 去噪步骤网格图像已保存到: {path_to_save_grid}")

