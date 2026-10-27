import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import AdaIN, NoiseInjection


class SynthesisBlock(nn.Module):
    """One resolution stage: (upsample) -> conv -> noise -> AdaIN, twice."""
    def __init__(self, in_ch, out_ch, w_dim, upsample=True):
        super().__init__()
        self.upsample = upsample
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.noise1 = NoiseInjection(out_ch)
        self.adain1 = AdaIN(out_ch, w_dim)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.noise2 = NoiseInjection(out_ch)
        self.adain2 = AdaIN(out_ch, w_dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, w):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.act(self.conv1(x))
        x = self.noise1(x)
        x = self.adain1(x, w)
        x = self.act(self.conv2(x))
        x = self.noise2(x)
        x = self.adain2(x, w)
        return x


class Generator(nn.Module):
    """
    Style-based synthesis network. Starts from a LEARNED CONSTANT at 4x4 and
    grows to 64x64, receiving style (w) at every block via AdaIN and
    independent noise at every block.
    """
    def __init__(self, w_dim=256, base_channels=256, img_channels=3):
        super().__init__()
        self.channels = {4: 256, 8: 256, 16: 128, 32: 64, 64: 32}
        self.resolutions = [4, 8, 16, 32, 64]

        self.starting_constant = nn.Parameter(torch.randn(1, self.channels[4], 4, 4))
        self.initial_noise = NoiseInjection(self.channels[4])
        self.initial_adain1 = AdaIN(self.channels[4], w_dim)
        self.initial_conv = nn.Conv2d(self.channels[4], self.channels[4], 3, padding=1)
        self.initial_noise2 = NoiseInjection(self.channels[4])
        self.initial_adain2 = AdaIN(self.channels[4], w_dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        self.blocks = nn.ModuleList()
        for i in range(1, len(self.resolutions)):
            in_ch = self.channels[self.resolutions[i - 1]]
            out_ch = self.channels[self.resolutions[i]]
            self.blocks.append(SynthesisBlock(in_ch, out_ch, w_dim, upsample=True))

        self.to_rgb = nn.Conv2d(self.channels[64], img_channels, kernel_size=1)
        self.num_style_inputs = 1 + len(self.blocks)  # number of w's usable (style mixing)

    def forward(self, ws):
        """
        ws: a single [B, w_dim] tensor (same style at every layer) OR a list of
            `num_style_inputs` tensors, one per block, for STYLE MIXING.
        """
        if isinstance(ws, torch.Tensor):
            ws = [ws] * self.num_style_inputs
        assert len(ws) == self.num_style_inputs

        batch = ws[0].shape[0]
        x = self.starting_constant.expand(batch, -1, -1, -1)
        x = self.initial_noise(x)
        x = self.initial_adain1(x, ws[0])
        x = self.act(self.initial_conv(x))
        x = self.initial_noise2(x)
        x = self.initial_adain2(x, ws[0])

        for i, block in enumerate(self.blocks):
            x = block(x, ws[i + 1])

        return torch.tanh(self.to_rgb(x))
