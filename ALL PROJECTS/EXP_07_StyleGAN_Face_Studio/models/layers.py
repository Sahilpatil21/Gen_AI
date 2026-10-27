import torch
import torch.nn as nn


class PixelNorm(nn.Module):
    """Normalizes each latent vector to unit norm — stabilizes the mapping network."""
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class AdaIN(nn.Module):
    """Adaptive Instance Normalization: style (w) modulates normalized features."""
    def __init__(self, channels, w_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels, affine=False)
        self.style_scale = nn.Linear(w_dim, channels)
        self.style_bias = nn.Linear(w_dim, channels)
        nn.init.ones_(self.style_scale.bias)
        nn.init.zeros_(self.style_bias.bias)

    def forward(self, x, w):
        x = self.instance_norm(x)
        scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)
        return scale * x + bias


class NoiseInjection(nn.Module):
    """Adds per-pixel Gaussian noise scaled by a learned per-channel weight."""
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        return x + self.weight * noise
