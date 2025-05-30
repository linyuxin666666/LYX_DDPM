import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
from torchvision import datasets, transforms
from model import UNetModel
from difussion import GaussianDiffusion
import time
from dataset import train_loader, model, optimizer, gaussian_diffusion, timesteps

device = "cuda" if torch.cuda.is_available() else "cpu"

epochs = 10



for epoch in range(epochs): # epochs 变量来自您的ddpm.py
    epoch_start_time = time.time()
    model.train() # 设置模型为训练模式

    print(f"\n开始 Epoch {epoch + 1}/{epochs} (PyTorch)")
    
    epoch_total_steps_pytorch = 0
    # 使用 tqdm 包装 train_loader
    for step, (images, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Steps (PyTorch)")):
        optimizer.zero_grad()

        images = images.to(device) # <--- 确保数据移动到设备
        
        # 采样时间步 t，并移动到设备
        t = torch.randint(0, timesteps, (images.shape[0],), device=device).long() # timesteps变量来自您的ddpm.py

        loss = gaussian_diffusion.train_losses(model, images, t) # train_losses应返回PyTorch的loss

        loss.backward() # PyTorch的反向传播
        optimizer.step() # PyTorch的参数更新
        
# 保存训练好的模型
model_save_path = "ddpm_pytorch_mnist.ckpt"
torch.save(model.state_dict(), model_save_path)
print(f"模型已成功保存到: {model_save_path}")

       
    