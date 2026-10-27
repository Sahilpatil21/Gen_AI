import os
import random

import numpy as np
import torch
import torchvision.utils as vutils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_grid(images, path, nrow=4, title=None):
    """Save a batch of images in [-1, 1] as a grid PNG."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    grid = vutils.make_grid(images.detach().cpu(), nrow=nrow, normalize=True)
    plt.figure(figsize=(6, 6))
    plt.axis("off")
    if title:
        plt.title(title)
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_losses(g_losses, d_losses, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="Generator")
    plt.plot(d_losses, label="Discriminator")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
