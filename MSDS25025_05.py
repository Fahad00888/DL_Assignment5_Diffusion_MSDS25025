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
