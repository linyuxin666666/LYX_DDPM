import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
from torchvision import datasets, transforms
from model import UNetModel
from difussion import GaussianDiffusion
import time
import matplotlib.pyplot as plt
import os
from dataset import model, gaussian_diffusion, timesteps

checkpoint_path = "ddpm_pytorch_mnist.ckpt"

# 加载模型权重
print(f"正在从 {checkpoint_path} 加载模型...")
model.load_state_dict(torch.load(checkpoint_path))
print("模型加载成功！")

batch_size = 64
generated_images = gaussian_diffusion.sample(model, 28, batch_size=batch_size, channels=1)


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

file_name = "generated_8x8_grid_from_visualization.png" 
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