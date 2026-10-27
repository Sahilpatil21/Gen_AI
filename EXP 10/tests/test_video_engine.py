"""
test_video_engine.py
====================
Unit tests for Experiment 10: Text-to-Video Generation Foundation Studio.
Tests video synthesis pipeline, temporal consistency, file output, and REST API.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2

# Add EXP 10 directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from video_engine import (
    CAMERA_MOTIONS,
    LIGHTING_PRESETS,
    SAMPLE_PROMPTS,
    VIDEO_STYLES,
    VideoConfig,
    video_engine,
)
from app import app


class TestVideoEngine(unittest.TestCase):
    def setUp(self):
        self.test_prompt = "A small boat sailing across a calm lake at sunrise, gentle water movement, cinematic wide shot"
        self.app_client = app.test_client()

    def test_system_info(self):
        """Test system telemetry retrieval."""
        info = video_engine.get_system_info()
        self.assertIn("cuda_available", info)
        self.assertIn("device", info)
        self.assertIsInstance(info["styles"], list)
        self.assertGreater(len(info["styles"]), 3)

    def test_presets_coverage(self):
        """Ensure all required lab manual styles and presets exist."""
        self.assertIn("Cinematic Photorealism", VIDEO_STYLES)
        self.assertIn("Anime & Studio Ghibli", VIDEO_STYLES)
        self.assertIn("Cinematic Wide Pan", CAMERA_MOTIONS)
        self.assertIn("Golden Hour Sunset", LIGHTING_PRESETS)
        self.assertGreaterEqual(len(SAMPLE_PROMPTS), 4)

    def test_video_config_defaults(self):
        """Verify VideoConfig instantiation and defaults."""
        cfg = VideoConfig(prompt=self.test_prompt)
        self.assertEqual(cfg.prompt, self.test_prompt)
        self.assertEqual(cfg.num_frames, 48)
        self.assertEqual(cfg.fps, 12)
        self.assertAlmostEqual(cfg.num_frames / cfg.fps, 4.0)
        self.assertEqual(cfg.width, 512)
        self.assertEqual(cfg.height, 288)

    def test_kinematic_render_and_mp4_export(self):
        """Test video generation produces valid MP4, frame images, and metadata."""
        cfg = VideoConfig(
            prompt="Lake with small boat sunrise wide shot",
            style="Cinematic Photorealism",
            camera_motion="Cinematic Wide Pan",
            lighting="Golden Hour Sunset",
            num_frames=8,
            fps=8,
            width=256,
            height=144,
            motion_intensity=4.0,
            seed=42,
            engine="kinematic_render",
        )

        frames, vid_url, meta = video_engine.generate(cfg)

        self.assertEqual(len(frames), 8)
        local_vid_file = video_engine.output_dir / meta.filename
        self.assertTrue(local_vid_file.exists(), f"Video file {local_vid_file} should exist on disk.")
        self.assertGreater(local_vid_file.stat().st_size, 500, "Video file size should be > 500 bytes.")
        self.assertTrue(vid_url.endswith(".mp4"), "Output video URL must end in .mp4.")
        capture = cv2.VideoCapture(str(local_vid_file))
        try:
            self.assertTrue(capture.isOpened(), "Generated MP4 should be decodable.")
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FOURCC)), cv2.VideoWriter_fourcc(*"h264"))
            has_frame, _ = capture.read()
            self.assertTrue(has_frame, "Generated MP4 should contain a playable frame.")
        finally:
            capture.release()

        # Check metadata
        self.assertEqual(meta.num_frames, 8)
        self.assertEqual(meta.fps, 8)
        self.assertAlmostEqual(meta.duration_seconds, 1.0, places=2)
        self.assertEqual(meta.resolution, "256x144")
        self.assertEqual(len(meta.frame_urls), 8)

    def test_api_system_info(self):
        """Test GET /api/system-info REST endpoint."""
        response = self.app_client.get("/api/system-info")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "online")
        self.assertIn("system", data)
        self.assertIn("styles", data)

    def test_api_history(self):
        """Test GET /api/history REST endpoint."""
        response = self.app_client.get("/api/history")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("history", data)
        self.assertIsInstance(data["history"], list)

    def test_api_generate_validation(self):
        """Test POST /api/generate with empty prompt returns 400 error."""
        response = self.app_client.post("/api/generate", json={"prompt": ""})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("error", data)

    def test_api_generation_defaults_to_four_seconds(self):
        metadata = Mock()
        metadata.to_dict.return_value = {"duration_seconds": 4.0}
        with patch.object(
            video_engine,
            "generate",
            return_value=([], "/static/outputs/test.mp4", metadata),
        ) as generate:
            response = self.app_client.post("/api/generate", json={"prompt": "A calm lake"})

        self.assertEqual(response.status_code, 200)
        config = generate.call_args.args[0]
        self.assertEqual(config.num_frames, 48)
        self.assertEqual(config.fps, 12)


if __name__ == "__main__":
    unittest.main()
