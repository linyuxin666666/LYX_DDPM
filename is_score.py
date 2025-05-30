import torch
from torch_fidelity import calculate_metrics
import os

# --- 配置 ---
# 包含生成图像的文件夹路径
# 你需要将其替换为你的实际路径
generated_images_folder = 'Jittor_DDPM/generated_10k_individual_images' 

# --- 主程序 ---
if __name__ == "__main__":
    cuda_available = torch.cuda.is_available()
    device_type = "GPU" if cuda_available else "CPU"


    metrics_dict = calculate_metrics(
        input1=generated_images_folder,
        isc=True,
        fid=False,
        cuda=cuda_available,
        batch_size=50,

    )

    print("\n--- Inception Score (IS) ---")
    is_mean = metrics_dict.get('inception_score_mean', float('nan')) 
    is_std = metrics_dict.get('inception_score_std', float('nan'))
    print(f"IS Mean: {is_mean:.4f}")
    print(f"IS Std:  {is_std:.4f}")
 