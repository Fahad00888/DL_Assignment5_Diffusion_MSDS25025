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
