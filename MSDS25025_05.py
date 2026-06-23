"""
MSDS25025_05.py
Deep Learning Assignment 5 (Bonus) - Image Generation Using Diffusion Models (DDPM)
Fahad Khalid | MSDS25025 | ITU
"""
import os
import glob
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# ----------------------------- Config -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 32          # resize target (small dataset -> keep small)
T = 1000               # number of diffusion timesteps

# 5 visually-distinct classes chosen from the 15 available
DEFAULT_CLASSES = ["Zebra", "Tiger", "Elephant", "Panda", "Dolphin"]

# ----------------------------- Step 1: DataLoader -----------------------------
class AnimalDataset(Dataset):
    """Reads animal images from class subfolders and normalizes them to [-1, 1]."""
    def __init__(self, root, classes, per_class=None, img_size=IMG_SIZE):
        self.paths = []
        for c in classes:
            files = sorted(glob.glob(os.path.join(root, c, "*")))
            if per_class is not None:
                files = files[:per_class]
            self.paths += files
        if len(self.paths) == 0:
            raise ValueError(f"No images found under {root} for classes {classes}")

        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),                       # [0,1]
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5)),        # -> [-1,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img)


# ----------------------------- Step 2: Forward (noising) process -----------------------------
# Precompute the noise schedule ONCE (linear beta schedule, the DDPM standard).
betas = torch.linspace(1e-4, 0.02, T)                    # (T,)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)            # alpha-bar_t
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)         # sqrt(alpha-bar_t)
sqrt_one_minus_acp = torch.sqrt(1.0 - alphas_cumprod)    # sqrt(1 - alpha-bar_t)

# move schedule tensors to device
betas = betas.to(device)
alphas = alphas.to(device)
alphas_cumprod = alphas_cumprod.to(device)
sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device)
sqrt_one_minus_acp = sqrt_one_minus_acp.to(device)


def forward_diffusion(x0, t, noise=None):
    """
    Closed-form forward process (reparameterization trick), NOT iterative addition.
    x_t = sqrt(alpha-bar_t) * x0 + sqrt(1 - alpha-bar_t) * noise
    Args:
        x0:    (B, C, H, W) clean images in [-1, 1]
        t:     (B,) integer timesteps, one per image
        noise: optional pre-sampled noise; if None, sampled from N(0, I)
    Returns:
        x_t:   (B, C, H, W) noised images at timestep t
        noise: (B, C, H, W) the noise that was added (the training target)
    """
    if noise is None:
        noise = torch.randn_like(x0)
    # gather the per-timestep scalars and reshape to broadcast over (B,C,H,W)
    s1 = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    s2 = sqrt_one_minus_acp[t].view(-1, 1, 1, 1)
    x_t = s1 * x0 + s2 * noise
    return x_t, noise


# ----------------------------- Step 3: Denoising U-Net -----------------------------
class TimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding -> small MLP. Lets ONE network handle all T noise levels."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class ResidualBlock(nn.Module):
    """Conv -> GroupNorm -> SiLU, with timestep injection and a residual skip."""
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.time_proj = nn.Linear(t_dim, out_ch)
        self.res = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]      # inject timestep
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.res(x)                               # residual connection


class UNet(nn.Module):
    """
    Small U-Net for noise prediction. Input: (x_t, t) -> Output: predicted noise (same shape as x_t).
    2 resolution levels (32x32 -> 16x16 -> bottleneck -> 16x16 -> 32x32) with skip connections.
    """
    def __init__(self, in_ch=3, base_ch=64, t_dim=128):
        super().__init__()
        self.time = TimeEmbedding(t_dim)

        # input projection
        self.in_conv = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)

        # down path
        self.down1 = ResidualBlock(base_ch, base_ch, t_dim)
        self.down2 = ResidualBlock(base_ch, base_ch * 2, t_dim)
        self.pool = nn.AvgPool2d(2)                              # 32 -> 16

        # bottleneck
        self.mid = ResidualBlock(base_ch * 2, base_ch * 2, t_dim)

        # up path (channels include the concatenated skips)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")    # 16 -> 32
        self.up1 = ResidualBlock(base_ch * 2 + base_ch * 2, base_ch, t_dim)
        self.up2 = ResidualBlock(base_ch + base_ch, base_ch, t_dim)

        # output projection: NO activation (predicted noise is unbounded)
        self.out_conv = nn.Conv2d(base_ch, in_ch, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time(t)

        x0 = self.in_conv(x)                 # (B, base, 32, 32)
        d1 = self.down1(x0, t_emb)           # (B, base, 32, 32)   -> skip
        d2 = self.down2(self.pool(d1), t_emb)  # (B, 2*base, 16, 16) -> skip

        m = self.mid(d2, t_emb)              # (B, 2*base, 16, 16)

        u = self.up1(torch.cat([m, d2], dim=1), t_emb)        # concat skip d2
        u = self.up(u)                                        # 16 -> 32
        u = self.up2(torch.cat([u, d1], dim=1), t_emb)        # concat skip d1

        return self.out_conv(u)              # (B, 3, 32, 32) predicted noise


# ----------------------------- Step 4: Custom loss -----------------------------
def diffusion_loss(pred_noise, true_noise, loss_type="l2"):
    """
    Customized loss: ||true_noise - pred_noise||.
    Implements the Algorithm-1 objective || eps - eps_theta(...) ||^2 from scratch
    (no nn.MSELoss / nn.L1Loss).
    Args:
        pred_noise: (B, C, H, W) noise predicted by the U-Net
        true_noise: (B, C, H, W) the actual noise added in the forward process
        loss_type:  "l2" (mean squared error) or "l1" (mean absolute error)
    Returns:
        scalar loss
    """
    diff = pred_noise - true_noise
    if loss_type == "l2":
        return (diff ** 2).mean()
    elif loss_type == "l1":
        return diff.abs().mean()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


# ----------------------------- Step 5: Training loop (Algorithm 1) -----------------------------
def train(model, dataloader, epochs=300, lr=2e-4, loss_type="l2", log_every=20):
    """
    Trains the DDPM noise predictor following Algorithm 1:
      repeat:
        x0 ~ data
        t  ~ Uniform({1..T})
        eps ~ N(0, I)
        gradient step on || eps - eps_theta(sqrt(acp)*x0 + sqrt(1-acp)*eps, t) ||^2
    Returns a list of per-step losses (for the loss curve in the report).
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x0 in dataloader:
            x0 = x0.to(device)
            t = torch.randint(0, T, (x0.size(0),), device=device)   # one t per image
            x_t, noise = forward_diffusion(x0, t)                   # Step 2
            pred = model(x_t, t)                                    # Step 3
            loss = diffusion_loss(pred, noise, loss_type)           # Step 4

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())
            epoch_loss += loss.item()

        if (epoch + 1) % log_every == 0 or epoch == 0:
            avg = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1:4d}/{epochs}  avg_loss={avg:.4f}")
    return losses


# ----------------------------- Step 6: Sampling (reverse process) -----------------------------
@torch.no_grad()
def sample(model, n=4, img_size=IMG_SIZE, save_every=None):
    """
    Reverse diffusion: start from pure noise x_T ~ N(0, I) and denoise to x_0.
    DDPM ancestral sampling using the noise-prediction parameterization.
    Args:
        model:      trained U-Net
        n:          number of images to generate
        save_every: if set, store intermediate x at every `save_every` steps (for the
                    noise->image progression figure)
    Returns:
        x:      (n, 3, H, W) generated images in [-1, 1]
        frames: list of intermediate tensors (empty if save_every is None)
    """
    model.eval()
    x = torch.randn(n, 3, img_size, img_size, device=device)   # x_T
    frames = []
    for i in reversed(range(T)):                               # T-1 ... 0
        t = torch.full((n,), i, device=device, dtype=torch.long)
        eps = model(x, t)                                      # predicted noise

        alpha = alphas[i]
        acp = alphas_cumprod[i]
        beta = betas[i]

        # mean of p_theta(x_{t-1} | x_t)
        coef = beta / torch.sqrt(1.0 - acp)
        mean = (1.0 / torch.sqrt(alpha)) * (x - coef * eps)

        if i > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(beta) * noise                # add stochastic term
        else:
            x = mean                                           # last step: no noise

        if save_every is not None and i % save_every == 0:
            frames.append(x.clamp(-1, 1).cpu())

    return x.clamp(-1, 1), frames
