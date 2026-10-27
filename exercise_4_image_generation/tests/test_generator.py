"""
test_generator.py
=================
Automated test suite for Exercise 4 AI Image Generation Engine & Web Endpoints.

Run: python -m pytest tests/
or: python tests/test_generator.py
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app
from pipeline_engine import (
    ASPECT_RATIOS,
    STYLES,
    GenerationConfig,
    ImageGenerationEngine,
)


class TestImageGenEngine(unittest.TestCase):
    def setUp(self):
        self.test_engine = ImageGenerationEngine()

    def test_system_info(self):
        info = self.test_engine.get_system_info()
        self.assertIn("device", info)
        self.assertIn("cuda_available", info)
        self.assertIn("vram_gb", info)
        self.assertIn("recommended_engine", info)

    def test_canvas_generation(self):
        config = GenerationConfig(
            prompt="A majestic lion on a savanna sunset",
            style="Realistic",
            num_images=1,
            num_inference_steps=15,
            guidance_scale=7.5,
            width=256,
            height=256,
            seed=42,
            engine="offline_canvas",
        )
        images, metadata, zip_url = self.test_engine.generate(config)
        self.assertEqual(len(images), 1)
        self.assertEqual(len(metadata), 1)
        self.assertEqual(images[0].size, (256, 256))
        self.assertTrue(zip_url.endswith(".zip"))
        self.assertEqual(metadata[0].seed, 42)

    def test_styles_and_ratios(self):
        self.assertIn("Realistic", STYLES)
        self.assertIn("Anime", STYLES)
        self.assertIn("Cyberpunk", STYLES)
        self.assertIn("1:1 (Square)", ASPECT_RATIOS)


class TestFlaskEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_index_route(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Lumina", response.data)
        self.assertIn(b"Creation Studio", response.data)

    def test_system_info_endpoint(self):
        response = self.client.get("/api/system-info")
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertEqual(json_data["status"], "online")
        self.assertIn("system", json_data)

    def test_presets_endpoint(self):
        response = self.client.get("/api/presets")
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertIn("prompts", json_data)
        self.assertIn("styles", json_data)

    def test_generate_endpoint_validation(self):
        # Empty prompt should return 400
        response = self.client.post("/api/generate", json={"prompt": ""})
        self.assertEqual(response.status_code, 400)
        json_data = response.get_json()
        self.assertFalse(json_data["success"])

    def test_generate_endpoint_success(self):
        # Test generation with procedural canvas engine
        response = self.client.post(
            "/api/generate",
            json={
                "prompt": "Test cybernetic owl",
                "style": "Cyberpunk",
                "num_images": 1,
                "engine": "offline_canvas",
            },
        )
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertTrue(json_data["success"])
        self.assertEqual(len(json_data["images"]), 1)
        self.assertTrue(json_data["zip_url"].endswith(".zip"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
