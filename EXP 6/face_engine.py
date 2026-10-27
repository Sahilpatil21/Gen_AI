"""
face_engine.py
==============
Deep Convolutional Generative Adversarial Network (DCGAN) Face Synthesis Engine.
Trained on the CelebFaces Attributes Dataset (CelebA) containing ~200,000 celebrity faces.
Supports batch generation, latent space morphing (SLERP/LERP), feature vector arithmetic,
and animated GIF/WebP sequence exports.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch

from model import Generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FaceGenEngine")


def slerp(val: float, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation between two latent vectors."""
    low_norm = low / torch.norm(low, dim=-1, keepdim=True)
    high_norm = high / torch.norm(high, dim=-1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(dim=-1, keepdim=True).clamp(-1, 1))
    so = torch.sin(omega)
    if so.abs().max() < 1e-6:
        return (1.0 - val) * low + val * high
    return (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high


def tensor_to_pil(tensor: torch.Tensor, target_size: Optional[Tuple[int, int]] = None) -> Image.Image:
    """Convert (3, H, W) tensor in [-1, 1] to a PIL Image."""
    arr = ((tensor + 1.0) / 2.0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255.0).astype(np.uint8)
    img = Image.fromarray(arr)
    if target_size and target_size != img.size:
        img = img.resize(target_size, Image.Resampling.LANCZOS)
    return img


@dataclass
class FaceMetadata:
    id: str
    seed: int
    latent_dim: int
    temperature: float
    resolution: str
    timestamp: str
    filepath: str
    filename: str
    generation_time_ms: float
    device: str
    model_type: str = "DCGAN (CelebA 64x64)"


@dataclass
class MorphMetadata:
    id: str
    seed_a: int
    seed_b: int
    steps: int
    fps: int
    duration_sec: float
    gif_filepath: str
    webp_filepath: str
    frame_filepaths: List[str]
    timestamp: str
    generation_time_ms: float


class FaceGenerationEngine:
    """Manages DCGAN model inference, latent space manipulation, and face synthesis."""

    def __init__(self, weights_path: Optional[str] = None, output_dir: Optional[str] = None):
        self.base_dir = Path(__file__).resolve().parent
        self.weights_path = Path(weights_path) if weights_path else self.base_dir / "generator.pth"
        self.output_dir = Path(output_dir) if output_dir else self.base_dir / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.output_dir / "face_history.json"

        # Hardware setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        self.latent_dim = 100

        # Load Generator
        self.model: Optional[Generator] = None
        self._load_model()

        # History cache
        self.history: List[Dict[str, Any]] = self._load_history()

    def _load_model(self) -> None:
        """Load pretrained DCGAN Generator weights."""
        if not self.weights_path.exists():
            logger.error(f"Generator weights not found at: {self.weights_path}")
            return

        try:
            self.model = Generator(latent_dim=self.latent_dim).to(self.device)
            state = torch.load(self.weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state, strict=True)
            self.model.eval()
            logger.info(f"Loaded DCGAN Generator ({self.weights_path.name}) onto {self._device_name}")
        except Exception as e:
            logger.exception(f"Failed to load generator weights: {e}")
            self.model = None

    def get_system_info(self) -> Dict[str, Any]:
        """Return engine status and hardware metrics."""
        return {
            "status": "online" if self.model is not None else "degraded",
            "device": self._device_name,
            "cuda_available": torch.cuda.is_available(),
            "latent_dim": self.latent_dim,
            "architecture": "DCGAN (Radford et al., 2015)",
            "dataset": "CelebFaces Attributes (CelebA)",
            "native_resolution": "64x64 RGB",
            "weights_file": self.weights_path.name,
            "weights_exists": self.weights_path.exists(),
            "total_faces_generated": len(self.history),
        }

    def _load_history(self) -> List[Dict[str, Any]]:
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading face history: {e}")
        return []

    def _save_history(self) -> None:
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.history[-60:], f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving face history: {e}")

    def generate_faces(
        self,
        count: int = 1,
        seed: Optional[int] = None,
        temperature: float = 1.0,
        scale: int = 2,
    ) -> List[Tuple[Image.Image, FaceMetadata]]:
        """
        Generate a batch of synthetic celebrity faces.
        Args:
            count: Number of faces (1 to 16)
            seed: Reproducible seed (or random if None)
            temperature: Truncation/scale factor for noise stddev (0.2 to 2.0)
            scale: Upscale factor: 1 (64x64), 2 (128x128), 4 (256x256)
        """
        if self.model is None:
            raise RuntimeError("Generator model is not loaded.")

        count = max(1, min(16, count))
        temperature = max(0.1, min(2.5, temperature))
        base_seed = seed if seed is not None and seed >= 0 else random.randint(0, 2**31 - 1)

        target_size = (64 * scale, 64 * scale)
        res_str = f"{target_size[0]}x{target_size[1]}"

        results: List[Tuple[Image.Image, FaceMetadata]] = []

        for i in range(count):
            item_seed = (base_seed + i * 37) % (2**31 - 1)
            torch.manual_seed(item_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(item_seed)

            start_t = time.time()
            with torch.no_grad():
                noise = torch.randn(1, self.latent_dim, 1, 1, device=self.device) * temperature
                tensor_out = self.model(noise)[0]

            elapsed_ms = round((time.time() - start_t) * 1000, 2)
            pil_img = tensor_to_pil(tensor_out, target_size=target_size)

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            face_id = f"{timestamp_str}_{item_seed}"
            fname = f"face_{face_id}.png"
            fpath = self.output_dir / fname
            pil_img.save(str(fpath), format="PNG")

            meta = FaceMetadata(
                id=face_id,
                seed=item_seed,
                latent_dim=self.latent_dim,
                temperature=temperature,
                resolution=res_str,
                timestamp=datetime.now().isoformat(),
                filepath=f"/outputs/{fname}",
                filename=fname,
                generation_time_ms=elapsed_ms,
                device=self._device_name,
            )

            self.history.append(asdict(meta))
            results.append((pil_img, meta))

        self._save_history()
        return results

    def interpolate_faces(
        self,
        seed_a: int,
        seed_b: int,
        steps: int = 16,
        fps: int = 10,
        scale: int = 2,
        use_slerp: bool = True,
    ) -> Tuple[List[Image.Image], MorphMetadata]:
        """
        Morph smoothly between two synthetic identities in the 100-dimensional latent space.
        Produces animated GIF and WebP representations.
        """
        if self.model is None:
            raise RuntimeError("Generator model is not loaded.")

        steps = max(6, min(32, steps))
        fps = max(4, min(24, fps))
        target_size = (64 * scale, 64 * scale)

        # Generate vectors z_a and z_b
        torch.manual_seed(seed_a)
        z_a = torch.randn(1, self.latent_dim, 1, 1, device=self.device)
        torch.manual_seed(seed_b)
        z_b = torch.randn(1, self.latent_dim, 1, 1, device=self.device)

        start_t = time.time()
        frames: List[Image.Image] = []

        with torch.no_grad():
            for s in range(steps):
                alpha = s / float(steps - 1)
                if use_slerp:
                    z_interp = slerp(alpha, z_a.view(1, -1), z_b.view(1, -1)).view(1, self.latent_dim, 1, 1)
                else:
                    z_interp = (1.0 - alpha) * z_a + alpha * z_b

                tensor_out = self.model(z_interp)[0]
                frames.append(tensor_to_pil(tensor_out, target_size=target_size))

        elapsed_ms = round((time.time() - start_t) * 1000, 2)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        morph_id = f"morph_{timestamp_str}_{seed_a}_{seed_b}"

        # Create subfolder for morph frames
        morph_dir = self.output_dir / morph_id
        morph_dir.mkdir(parents=True, exist_ok=True)
        frame_paths: List[str] = []

        for idx, frame in enumerate(frames):
            frame_fname = f"step_{idx+1:02d}.png"
            frame.save(str(morph_dir / frame_fname), format="PNG")
            frame_paths.append(f"/outputs/{morph_id}/{frame_fname}")

        # Save animated GIF (bounce loop)
        bounce_frames = frames + frames[-2:0:-1]
        duration_ms = int(1000 / fps)

        gif_fname = f"{morph_id}.gif"
        gif_path = self.output_dir / gif_fname
        bounce_frames[0].save(
            str(gif_path),
            save_all=True,
            append_images=bounce_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )

        # Save animated WebP
        webp_fname = f"{morph_id}.webp"
        webp_path = self.output_dir / webp_fname
        bounce_frames[0].save(
            str(webp_path),
            save_all=True,
            append_images=bounce_frames[1:],
            duration=duration_ms,
            loop=0,
            quality=92,
        )

        duration_sec = round(len(bounce_frames) / fps, 2)

        meta = MorphMetadata(
            id=morph_id,
            seed_a=seed_a,
            seed_b=seed_b,
            steps=steps,
            fps=fps,
            duration_sec=duration_sec,
            gif_filepath=f"/outputs/{gif_fname}",
            webp_filepath=f"/outputs/{webp_fname}",
            frame_filepaths=frame_paths,
            timestamp=datetime.now().isoformat(),
            generation_time_ms=elapsed_ms,
        )

        return frames, meta


# Singleton engine instance
face_engine = FaceGenerationEngine()
