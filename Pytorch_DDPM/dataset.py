import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
from torchvision import datasets, transforms
from model import UNetModel
from difussion import GaussianDiffusion

batch_size = 64
timesteps = 500

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# use MNIST dataset
dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
# define model and diffusion
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNetModel(
    in_channels=1,
    model_channels=96,
    out_channels=1,
    channel_mult=(1, 2, 2),
    attention_resolutions=[]
)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
gaussian_diffusion = GaussianDiffusion(timesteps=timesteps)
