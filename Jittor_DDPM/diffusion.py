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
