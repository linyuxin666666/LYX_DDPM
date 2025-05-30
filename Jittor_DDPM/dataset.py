import jittor as jt
from jittor.dataset import MNIST
from jittor import transform
from model import UNetModel
from diffusion import GaussianDiffusion

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
