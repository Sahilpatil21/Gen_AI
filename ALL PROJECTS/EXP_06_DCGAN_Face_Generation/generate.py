"""
generate.py
===========
CLI tool to generate synthetic face images using the pretrained DCGAN generator checkpoint.
Pure PyTorch and Pillow implementation (no torchvision required).
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from model import Generator


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a (C, H, W) PyTorch tensor in [-1, 1] to a PIL Image."""
    arr = ((tensor + 1.0) / 2.0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def main():
    parser = argparse.ArgumentParser(description="Generate 64x64 face images with the supplied DCGAN generator.")
    parser.add_argument("--count", type=int, default=1, help="Number of images to generate (default: 1)")
    parser.add_argument("--output", default="outputs", help="Output directory (default: outputs)")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for repeatable output")
    parser.add_argument("--scale", type=int, default=1, help="Upscale factor: 1 (64x64), 2 (128x128), 4 (256x256)")
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be at least 1")
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(__file__).with_name("generator.pth")
    model = Generator().to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        noise = torch.randn(args.count, 100, 1, 1, device=device)
        images = model(noise)
        for i, img_tensor in enumerate(images, start=1):
            pil_img = tensor_to_pil(img_tensor)
            if args.scale > 1:
                new_size = (64 * args.scale, 64 * args.scale)
                pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
            path = output_dir / f"generated_face_{i:03d}.png"
            pil_img.save(path)
            print(f"Saved: {path} ({pil_img.size[0]}x{pil_img.size[1]})")

    print(f"Device used: {device}")


if __name__ == "__main__":
    main()
