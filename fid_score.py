import torch
# import torchvision # Not strictly needed as pytorch-fid handles Inception model loading
# import torchvision.transforms as transforms # Not strictly needed as pytorch-fid handles transforms
from pytorch_fid import fid_score

# 准备真实数据分布和生成模型的图像数据
real_images_folder = 'mnist_pngs_all/test' # 请确保替换为真实路径
generated_images_folder = 'Pytorch-DDPM/generated_10k_individual_images' # 请确保替换为真实路径

# pytorch-fid 会自动加载 Inception 模型并处理图像转换
# 因此我们不需要手动定义 inception_model 和 transform

# 确定运行设备 (GPU或CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 计算FID距离值
fid_value = fid_score.calculate_fid_given_paths(
    paths=[real_images_folder, generated_images_folder],
    batch_size=50,  # 你可以根据需要调整批处理大小
    device=device,
    dims=2048       # InceptionV3 特征的标准维度
)
print('FID value:', fid_value)
