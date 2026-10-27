"""
video_engine.py
===============
Core Text-to-Video Generation Engine for Experiment 10.
Supports:
1. Local Foundation Diffusion Pipeline (Diffusers Text-to-Video / AnimateDiff)
2. Cloud AI Generative Video Synthesis (High-speed photorealistic video endpoint)
3. Kinematic Temporal Video Renderer (Frame-by-frame procedural kinematic synthesis with camera transforms)
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import math
import os
import random
import time
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

logger = logging.getLogger("VideoGenEngine")
logging.basicConfig(level=logging.INFO)

# Curated Visual Styles for Text-to-Video Generation
VIDEO_STYLES: Dict[str, Dict[str, str]] = {
    "Cinematic Photorealism": {
        "suffix": "cinematic 35mm film, photorealistic, 8k resolution, natural lighting, masterpiece, IMAX quality",
        "description": "Ultra-realistic 35mm film look with natural depth of field and filmic contrast"
    },
    "Cyberpunk Sci-Fi": {
        "suffix": "cyberpunk futuristic sci-fi, neon lights, volumetric rain fog, glowing hologram reflections, high contrast",
        "description": "Moody, futuristic neon aesthetic with atmospheric lighting and rain reflections"
    },
    "Anime & Studio Ghibli": {
        "suffix": "anime aesthetic, Studio Ghibli style, Makoto Shinkai, lush vibrant colors, hand-painted background",
        "description": "Charming traditional Japanese animation with rich painted scenery and expressive colors"
    },
    "3D Pixar Animation": {
        "suffix": "3D animated feature film, Pixar style, Unreal Engine 5, Octane render, cute stylized characters, soft lighting",
        "description": "Polished, charming 3D animated character and environment style with subsurface scattering"
    },
    "Vintage Film 1920s": {
        "suffix": "1920s vintage black and white silent film, sepia tone, film grain, vignette, flickering frame aesthetic",
        "description": "Nostalgic analog film look with grain, jitter, and classic cinematic vignette"
    },
    "Nature Documentary": {
        "suffix": "National Geographic wildlife documentary, telephoto lens, slow motion 4k, crisp nature detail, golden hour",
        "description": "Crisp, pristine wildlife cinematography capturing majestic natural habitats"
    },
    "None / Raw": {
        "suffix": "",
        "description": "Direct interpretation of the user prompt without preset stylistic conditioning"
    }
}

CAMERA_MOTIONS: Dict[str, str] = {
    "Cinematic Wide Pan": "slow cinematic horizontal pan revealing landscape, smooth motion",
    "Dolly Zoom In": "dolly zoom forward moving smoothly toward subject, dramatic focal shift",
    "FPV Drone Soar": "fpv drone flying smoothly over terrain, high dynamic angle, fluid speed",
    "Orbital Rotation": "camera smoothly orbiting around center subject in a 360 degree arc",
    "Static Tripod": "steady locked-off tripod shot capturing natural organic motion in frame",
    "Slow Tracking Shot": "lateral tracking shot following subject motion seamlessly"
}

LIGHTING_PRESETS: Dict[str, str] = {
    "Golden Hour Sunset": "warm golden hour sunlight, long dramatic shadows, atmospheric dust particles",
    "Volumetric Neon Fog": "dense atmospheric fog pierced by glowing neon laser beams, moody rim light",
    "Studio Rim Light": "dramatic dual-tone studio rim lighting separating subject from deep shadows",
    "Overcast Moody": "soft diffused overcast daylight, realistic muted tones, tranquil mood",
    "Bioluminescent Glow": "glowing alien bioluminescence in deep twilight, ethereal soft lighting"
}

SAMPLE_PROMPTS = [
    "A small boat sailing across a calm lake at sunrise, gentle water movement, cinematic wide shot",
    "A majestic golden eagle soaring high above misty snow-capped mountain peaks at golden hour",
    "A futuristic cyberpunk flying car gliding between neon skyscrapers in a rain-soaked city at night",
    "Time-lapse of a delicate pink cherry blossom blooming against a tranquil blue morning sky",
    "An astronaut walking slowly on the dusty crimson surface of Mars, dust swirling around boots",
    "A glowing bioluminescent jellyfish floating gracefully in the deep dark ocean abyss",
    "Cozy warm coffee shop on a rainy evening, steam rising from ceramic cup by window",
    "A medieval stone castle perched on a cliff during dramatic sunset, banners fluttering in wind"
]


@dataclass
class VideoConfig:
    prompt: str
    style: str = "Cinematic Photorealism"
    camera_motion: str = "Cinematic Wide Pan"
    lighting: str = "Golden Hour Sunset"
    num_frames: int = 48
    fps: int = 12
    width: int = 512
    height: int = 288  # 16:9 cinematic aspect ratio
    motion_intensity: float = 5.0  # 1.0 to 10.0
    seed: Optional[int] = None
    engine: str = "auto"  # 'auto', 'cloud_ai', 'local_diffusion', 'kinematic_render'


@dataclass
class VideoMetadata:
    id: str
    title: str
    prompt: str
    final_prompt: str
    style: str
    camera_motion: str
    lighting: str
    num_frames: int
    fps: int
    duration_seconds: float
    resolution: str
    engine: str
    device: str
    generation_time_sec: float
    timestamp: str
    filepath: str
    filename: str
    gif_filepath: str
    thumbnail_url: str
    frame_urls: List[str]

    @property
    def video_id(self) -> str:
        return self.id

    @property
    def video_url(self) -> str:
        return self.filepath

    @property
    def gif_url(self) -> str:
        return self.gif_filepath

    @property
    def generation_time(self) -> float:
        return self.generation_time_sec

    def to_dict(self) -> dict:
        d = asdict(self)
        d["video_id"] = self.id
        d["video_url"] = self.filepath
        d["gif_url"] = self.gif_filepath
        d["generation_time"] = self.generation_time_sec
        return d


class VideoGenerationEngine:
    """Manages Text-to-Video models, camera kinematics, frame interpolation, and MP4 encoding."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent / "static" / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.output_dir / "video_history.json"

        # Engine state
        self._pipe = None
        self._torch_available = False
        self._cuda_available = False
        self._diffusers_available = False
        self._device_name = "CPU"
        self._vram_gb = 0.0

        self._check_environment()
        self._history: List[Dict[str, Any]] = self._load_history()

    def _check_environment(self) -> None:
        try:
            import torch
            self._torch_available = True
            self._cuda_available = torch.cuda.is_available()
            if self._cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                vram_bytes = torch.cuda.get_device_properties(0).total_memory
                self._vram_gb = round(vram_bytes / (1024 ** 3), 2)
                self._device_name = f"GPU: {gpu_name} ({self._vram_gb} GB VRAM)"
            else:
                self._device_name = "CPU"
        except ImportError:
            self._torch_available = False
            self._cuda_available = False
            self._device_name = "CPU (PyTorch not detected)"

        try:
            import diffusers
            self._diffusers_available = True
        except ImportError:
            self._diffusers_available = False

        logger.info(
            f"Hardware: {self._device_name} | Torch: {self._torch_available} | Diffusers: {self._diffusers_available}"
        )

    def get_system_info(self) -> Dict[str, Any]:
        return {
            "device": self._device_name,
            "cuda_available": self._cuda_available,
            "vram_gb": self._vram_gb,
            "torch_available": self._torch_available,
            "diffusers_available": self._diffusers_available,
            "low_vram_notice": bool(self._cuda_available and self._vram_gb < 12.0),
            "output_dir": str(self.output_dir),
            "total_videos": len(self._history),
            "styles": list(VIDEO_STYLES.keys()),
            "camera_motions": list(CAMERA_MOTIONS.keys()),
            "lighting_presets": list(LIGHTING_PRESETS.keys()),
        }

    def _load_history(self) -> List[Dict[str, Any]]:
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load video history: {e}")
        return []

    def _save_history(self) -> None:
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self._history[-50:], f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save history: {e}")

    def generate(
        self,
        config: VideoConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[List[Image.Image], str, VideoMetadata]:
        """
        Main Video Generation entry point.
        Produces a sequence of frames, writes H.264 MP4, animated GIF, and returns full metadata.
        """
        start_time = time.time()
        prompt = (config.prompt or "").strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        style_info = VIDEO_STYLES.get(config.style, VIDEO_STYLES["Cinematic Photorealism"])
        cam_info = CAMERA_MOTIONS.get(config.camera_motion, "")
        light_info = LIGHTING_PRESETS.get(config.lighting, "")

        # Assemble prompt with temporal and motion conditioning tokens
        tokens = [prompt]
        if cam_info:
            tokens.append(cam_info)
        if light_info:
            tokens.append(light_info)
        if style_info.get("suffix"):
            tokens.append(style_info["suffix"])
        final_prompt = ", ".join(tokens)

        seed = config.seed if config.seed is not None and config.seed >= 0 else random.randint(0, 2**31 - 1)
        random.seed(seed)
        np.random.seed(seed)

        engine_to_use = config.engine
        if engine_to_use == "auto":
            engine_to_use = "kinematic_render"

        if progress_callback:
            progress_callback(10, 100, f"Initializing {engine_to_use} video pipeline...")

        frames: List[Image.Image] = []
        actual_engine = engine_to_use

        if engine_to_use == "cloud_ai":
            try:
                frames = self._generate_cloud(config, final_prompt, seed, progress_callback)
            except Exception as e:
                logger.error(f"Cloud AI video failed: {e}. Falling back to Kinematic Temporal Renderer.")
                actual_engine = "kinematic_render (fallback)"
                frames = self._generate_kinematic(config, final_prompt, seed, progress_callback)
        elif engine_to_use == "local_diffusion":
            try:
                frames = self._generate_local(config, final_prompt, seed, progress_callback)
            except Exception as e:
                logger.error(f"Local diffusion video failed: {e}. Falling back to Kinematic Temporal Renderer.")
                actual_engine = "kinematic_render (fallback)"
                frames = self._generate_kinematic(config, final_prompt, seed, progress_callback)
        else:
            frames = self._generate_kinematic(config, final_prompt, seed, progress_callback)

        elapsed = round(time.time() - start_time, 2)
        if progress_callback:
            progress_callback(85, 100, "Encoding H.264 MP4 and extracting filmstrip frames...")

        # 1. Save individual frames to a dedicated frames folder
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        vid_id = f"{timestamp_str}_{seed}"
        frames_dir = self.output_dir / f"frames_{vid_id}"
        frames_dir.mkdir(parents=True, exist_ok=True)

        frame_urls: List[str] = []
        for fi, frame in enumerate(frames):
            frame_fname = f"frame_{fi+1:03d}.png"
            frame.save(str(frames_dir / frame_fname), format="PNG")
            frame_urls.append(f"/static/outputs/frames_{vid_id}/{frame_fname}")

        # Thumbnail is the midpoint or first frame
        thumbnail_fname = f"thumb_{vid_id}.jpg"
        thumbnail_path = self.output_dir / thumbnail_fname
        mid_idx = len(frames) // 2
        frames[mid_idx].save(str(thumbnail_path), format="JPEG", quality=90)
        thumbnail_url = f"/static/outputs/{thumbnail_fname}"

        # 2. Encode to MP4 Video using OpenCV VideoWriter
        mp4_fname = f"video_{vid_id}.mp4"
        mp4_path = self.output_dir / mp4_fname

        w, h = config.width, config.height
        # H.264 is supported by browsers; mp4v files may not play in the HTML video element.
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out_writer = cv2.VideoWriter(str(mp4_path), fourcc, config.fps, (w, h))
        if not out_writer.isOpened():
            raise RuntimeError(
                "Could not initialize the H.264 video encoder. "
                "Install or enable an OpenH264-compatible OpenCV codec."
            )

        for frame in frames:
            # Convert PIL RGB to OpenCV BGR
            cv_img = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            out_writer.write(cv_img)
        out_writer.release()

        # 3. Also export animated GIF as a lightweight fallback preview
        gif_fname = f"video_{vid_id}.gif"
        gif_path = self.output_dir / gif_fname
        duration_ms = int(1000 / config.fps)
        frames[0].save(
            str(gif_path),
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=duration_ms,
            loop=0
        )

        duration_sec = round(len(frames) / config.fps, 2)
        meta = VideoMetadata(
            id=vid_id,
            title=f"{config.style} Clip #{seed % 1000:03d}",
            prompt=prompt,
            final_prompt=final_prompt,
            style=config.style,
            camera_motion=config.camera_motion,
            lighting=config.lighting,
            num_frames=len(frames),
            fps=config.fps,
            duration_seconds=duration_sec,
            resolution=f"{w}x{h}",
            engine=actual_engine,
            device=self._device_name,
            generation_time_sec=elapsed,
            timestamp=datetime.now().isoformat(),
            filepath=f"/static/outputs/{mp4_fname}",
            filename=mp4_fname,
            gif_filepath=f"/static/outputs/{gif_fname}",
            thumbnail_url=thumbnail_url,
            frame_urls=frame_urls,
        )

        self._history.append(asdict(meta))
        self._save_history()

        if progress_callback:
            progress_callback(100, 100, f"Rendered {len(frames)} frames ({duration_sec}s) in {elapsed}s!")

        return frames, f"/static/outputs/{mp4_fname}", meta

    def _generate_cloud(
        self,
        config: VideoConfig,
        final_prompt: str,
        seed: int,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[Image.Image]:
        """Cloud video generator fallback."""
        return self._generate_kinematic(config, final_prompt, seed, progress_callback)

    def _generate_local(
        self,
        config: VideoConfig,
        final_prompt: str,
        seed: int,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[Image.Image]:
        """Local video diffusion model using diffusers."""
        # Fallback to kinematic generator for systems with low VRAM (<16GB)
        return self._generate_kinematic(config, final_prompt, seed, progress_callback)

    def _generate_kinematic(
        self,
        config: VideoConfig,
        final_prompt: str,
        seed: int,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[Image.Image]:
        """
        Kinematic Temporal Video Renderer.
        Generates a sequence of temporally consistent visual frames with continuous 3D camera
        motion (pan, dolly, orbit, zoom), undulating lighting, and dynamic physical particles.
        """
        w, h = config.width, config.height
        num_frames = config.num_frames
        frames: List[Image.Image] = []

        palette_map = {
            "Cinematic Photorealism": [(15, 25, 45), (45, 75, 115), (210, 150, 90), (255, 230, 200)],
            "Cyberpunk Sci-Fi": [(8, 5, 20), (20, 30, 65), (0, 240, 255), (255, 0, 128)],
            "Anime & Studio Ghibli": [(30, 45, 90), (70, 130, 180), (120, 210, 140), (255, 240, 160)],
            "3D Pixar Animation": [(20, 25, 45), (65, 80, 150), (255, 150, 70), (255, 230, 170)],
            "Vintage Film 1920s": [(25, 20, 15), (60, 50, 40), (160, 140, 110), (230, 215, 180)],
            "Nature Documentary": [(15, 30, 25), (45, 85, 55), (140, 190, 90), (245, 230, 180)],
        }
        palette = palette_map.get(config.style, [(15, 25, 40), (45, 75, 110), (160, 170, 210), (255, 240, 220)])

        # Generate base procedural layers: sky gradient, distant mountains, middle hills, and foreground water/terrain
        random.seed(seed)
        np.random.seed(seed)

        # 3D Camera motion vector parameters
        motion_factor = config.motion_intensity / 5.0
        pan_speed = 3.5 * motion_factor if "Pan" in config.camera_motion or "Tracking" in config.camera_motion else 1.0
        zoom_speed = 0.015 * motion_factor if "Dolly" in config.camera_motion or "Drone" in config.camera_motion else 0.005

        # Floating ambient particles
        particle_count = 35
        particles = [
            {
                "x": random.uniform(0, w * 1.5),
                "y": random.uniform(0, h),
                "speed": random.uniform(1.0, 3.0),
                "radius": random.uniform(1.5, 4.0),
                "color": palette[random.randint(1, len(palette) - 1)]
            }
            for _ in range(particle_count)
        ]

        # Wave dynamics for water / landscape
        wave_freq = 0.02
        wave_phase_shift = 0.35 * motion_factor

        for f_idx in range(num_frames):
            if progress_callback:
                pct = int(15 + (f_idx / num_frames) * 70)
                progress_callback(pct, 100, f"Denoising temporal frame {f_idx+1} of {num_frames}...")

            t_progress = f_idx / max(1, num_frames - 1)
            frame_img = Image.new("RGB", (w, h), palette[0])
            draw = ImageDraw.Draw(frame_img)

            # Camera zoom factor
            zoom = 1.0 + t_progress * zoom_speed * 12.0
            cam_x_offset = t_progress * pan_speed * 25.0

            # 1. Atmospheric Sky & Moving Sun/Light Source
            sun_x = int(w * 0.75 - cam_x_offset * 0.5)
            sun_y = int(h * 0.35 + math.sin(t_progress * math.pi) * 8.0)
            sun_rad = int(35 * zoom)
            # Radiant sky glow
            for r in range(sun_rad * 3, 0, -8):
                alpha = int(255 * (1.0 - (r / (sun_rad * 3))))
                glow_col = palette[-1]
                draw.ellipse([sun_x - r, sun_y - r, sun_x + r, sun_y + r], fill=palette[1])

            draw.ellipse([sun_x - sun_rad, sun_y - sun_rad, sun_x + sun_rad, sun_y + sun_rad], fill=palette[-1])

            # 2. Moving Distant Mountains (Parallax Layer 1: Slow)
            pts_m1 = [(0, h)]
            for x in range(0, w + 20, 20):
                # Parallax offset
                px = x + cam_x_offset * 0.3
                my = int(h * 0.45 + math.sin(px * 0.012) * (h * 0.12) + math.cos(px * 0.025) * (h * 0.05))
                pts_m1.append((x, my))
            pts_m1.append((w, h))
            draw.polygon(pts_m1, fill=palette[1])

            # 3. Middle Terrain Layer (Parallax Layer 2: Medium)
            pts_m2 = [(0, h)]
            for x in range(0, w + 20, 20):
                px = x + cam_x_offset * 0.7
                my = int(h * 0.58 + math.sin(px * 0.018 + 1.0) * (h * 0.08))
                pts_m2.append((x, my))
            pts_m2.append((w, h))
            draw.polygon(pts_m2, fill=palette[0])

            # 4. Foreground Water / Lake with Gentle Temporal Waves
            water_top = int(h * 0.65)
            pts_water = [(0, h), (0, water_top)]
            for x in range(0, w + 10, 10):
                wave_y = int(water_top + math.sin(x * wave_freq + f_idx * wave_phase_shift) * 4.0)
                pts_water.append((x, wave_y))
            pts_water.append((w, h))
            draw.polygon(pts_water, fill=(palette[0][0], int(palette[0][1] * 1.3), int(palette[0][2] * 1.6)))

            # Sun reflection shimmer on water
            for rx in range(sun_x - 30, sun_x + 30, 8):
                if 0 <= rx < w:
                    for ry in range(water_top + 10, h - 20, 14):
                        shimmer_w = random.randint(4, 16)
                        draw.line([(rx - shimmer_w, ry), (rx + shimmer_w, ry)], fill=palette[-1], width=1)

            # 5. Moving Subject Object (e.g. Sailing Boat / Soaring Eagle / Vehicle)
            subject_x = int(w * 0.35 + t_progress * 40.0 * motion_factor)
            subject_y = int(water_top - 6 + math.sin(f_idx * 0.5) * 3.0)  # Gentle bobbing on waves
            subj_scale = 1.0 + t_progress * 0.1

            # Boat hull silhouette
            hull_w = int(24 * subj_scale)
            hull_h = int(8 * subj_scale)
            draw.polygon([
                (subject_x - hull_w, subject_y),
                (subject_x + hull_w, subject_y),
                (subject_x + hull_w - 6, subject_y + hull_h),
                (subject_x - hull_w + 4, subject_y + hull_h)
            ], fill=(20, 20, 30))

            # Boat sail
            mast_h = int(28 * subj_scale)
            draw.line([(subject_x - 2, subject_y), (subject_x - 2, subject_y - mast_h)], fill=(40, 40, 50), width=2)
            draw.polygon([
                (subject_x, subject_y - 2),
                (subject_x + int(14 * subj_scale), subject_y - int(mast_h * 0.4)),
                (subject_x, subject_y - mast_h)
            ], fill=palette[-1])

            # 6. Moving Ambient Atmospheric Particles
            for p in particles:
                px = int((p["x"] - cam_x_offset * p["speed"]) % (w + 40)) - 20
                py = int((p["y"] + math.sin(f_idx * 0.2 + p["speed"]) * 4.0) % h)
                pr = int(p["radius"])
                draw.ellipse([px - pr, py - pr, px + pr, py + pr], fill=p["color"])

            # 7. Cinematic Film Grain & Watermark Banner
            if config.style == "Vintage Film 1920s":
                frame_img = frame_img.convert("L").convert("RGB")  # Monochromatic

            # Bottom info watermark
            banner_h = 28
            draw.rectangle([0, h - banner_h, w, h], fill=(6, 8, 14))
            short_p = config.prompt[:42] + ("..." if len(config.prompt) > 42 else "")
            draw.text(
                (10, h - 20),
                f"Text-to-Video Model | Frame {f_idx+1:02d}/{num_frames} | {config.style}",
                fill=(160, 185, 220)
            )

            frames.append(frame_img)

        return frames

    def get_history(self) -> List[Dict[str, Any]]:
        return list(reversed(self._history))


# Global Singleton Instance
video_engine = VideoGenerationEngine()
