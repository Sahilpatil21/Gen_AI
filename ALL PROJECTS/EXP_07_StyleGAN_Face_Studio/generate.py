"""
Inference demos using a trained checkpoint.

  python generate.py truncation              # truncation trick (psi sweep)
  python generate.py mixing --crossover 2    # style-mixing grid
  python generate.py noise                   # same w, different noise
"""
import argparse
import os

import numpy as np
import torch
import torchvision.utils as vutils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from models import MappingNetwork, Generator

device = config.DEVICE


def load_models(ckpt_path):
    mapping = MappingNetwork(config.Z_DIM, config.W_DIM, config.MAPPING_LAYERS).to(device)
    generator = Generator(w_dim=config.W_DIM).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    mapping.load_state_dict(ckpt["mapping"])
    generator.load_state_dict(ckpt["generator"])
    mapping.eval(); generator.eval()
    return mapping, generator


def _img(t, nrow):
    return np.transpose(vutils.make_grid(t.cpu(), nrow=nrow, normalize=True), (1, 2, 0))


@torch.no_grad()
def w_average(mapping, n_samples=2000):
    return mapping(torch.randn(n_samples, config.Z_DIM, device=device)).mean(0, keepdim=True)


@torch.no_grad()
def truncation_demo(mapping, generator, out):
    w_avg = w_average(mapping)
    z = torch.randn(4, config.Z_DIM, device=device)   # same latents across psi values
    w = mapping(z)
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, psi in zip(axes, [1.0, 0.7, 0.4, 0.1]):
        imgs = generator(w_avg + psi * (w - w_avg))
        ax.imshow(_img(imgs, 2)); ax.set_title(f"psi = {psi}"); ax.axis("off")
    plt.suptitle("Truncation Trick: Diversity (psi=1) vs Quality (psi -> 0)")
    plt.savefig(out, bbox_inches="tight"); plt.close()


@torch.no_grad()
def style_mix_demo(mapping, generator, out, crossover=2, n=4):
    z_src = torch.randn(n, config.Z_DIM, device=device)
    z_tgt = torch.randn(n, config.Z_DIM, device=device)
    w_src, w_tgt = mapping(z_src), mapping(z_tgt)
    img_src, img_tgt = generator(w_src), generator(w_tgt)

    fig, axes = plt.subplots(n + 1, n + 1, figsize=(12, 12))
    axes[0, 0].axis("off")
    for i in range(n):
        axes[0, i + 1].imshow(_img(img_tgt[i:i+1], 1)); axes[0, i + 1].set_title("fine style src"); axes[0, i + 1].axis("off")
        axes[i + 1, 0].imshow(_img(img_src[i:i+1], 1)); axes[i + 1, 0].set_ylabel("coarse style src"); axes[i + 1, 0].axis("off")
        for j in range(n):
            ws = [w_src[i:i+1] if k < crossover else w_tgt[j:j+1] for k in range(generator.num_style_inputs)]
            axes[i + 1, j + 1].imshow(_img(generator(ws), 1)); axes[i + 1, j + 1].axis("off")
    plt.suptitle(f"Style Mixing Grid (crossover at block {crossover}) — rows: coarse style, cols: fine style")
    plt.tight_layout(); plt.savefig(out, bbox_inches="tight"); plt.close()


@torch.no_grad()
def noise_demo(mapping, generator, out, n=6):
    w = mapping(torch.randn(1, config.Z_DIM, device=device)).repeat(n, 1)
    imgs = generator(w)   # same w, noise re-sampled per forward call
    plt.figure(figsize=(12, 3)); plt.imshow(_img(imgs, n))
    plt.title("Same w (identity) — different noise (fine stochastic detail)"); plt.axis("off")
    plt.savefig(out, bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["truncation", "mixing", "noise"])
    p.add_argument("--ckpt", default=os.path.join(config.CHECKPOINT_DIR, "stylegan_latest.pt"))
    p.add_argument("--crossover", type=int, default=2)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    m, g = load_models(a.ckpt)
    out = a.out or os.path.join(config.RESULTS_DIR, {"truncation": "truncation_trick.png",
                                                      "mixing": "style_mixing_grid.png",
                                                      "noise": "noise_variation.png"}[a.mode])
    {"truncation": lambda: truncation_demo(m, g, out),
     "mixing": lambda: style_mix_demo(m, g, out, a.crossover),
     "noise": lambda: noise_demo(m, g, out)}[a.mode]()
    print("Saved", out)
