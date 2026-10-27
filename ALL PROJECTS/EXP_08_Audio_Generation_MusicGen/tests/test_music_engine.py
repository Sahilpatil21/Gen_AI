"""
test_music_engine.py
====================
Unit test suite for Experiment 8 Music Generation Engine & API Endpoints.

Run: python -m unittest tests/test_music_engine.py
"""

import sys
import unittest
from pathlib import Path

import importlib.util

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

_spec = importlib.util.spec_from_file_location("exp08_app", BASE_DIR / "app.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["exp08_app"] = _mod
_spec.loader.exec_module(_mod)
app = _mod.app
from music_engine import (
    GENRE_PRESETS,
    MusicConfig,
    MusicGenerationEngine,
)


class TestMusicEngine(unittest.TestCase):
    def setUp(self):
        self.engine = MusicGenerationEngine()

    def test_system_info(self):
        info = self.engine.get_system_info()
        self.assertIn("device", info)
        self.assertIn("genres", info)
        self.assertIn("output_dir", info)

    def test_algorithmic_synthesis(self):
        config = MusicConfig(
            prompt="An upbeat lo-fi hip-hop track with relaxed piano",
            genre="Lo-Fi Hip-Hop",
            duration_seconds=5,
            engine="algorithmic_synth",
            seed=42,
        )
        audio_norm, sr, meta = self.engine.generate(config)
        self.assertEqual(sr, 32000)
        self.assertEqual(len(audio_norm), 32000 * 5)
        self.assertTrue(np_max := (audio_norm.max() <= 1.0))
        self.assertTrue(np_min := (audio_norm.min() >= -1.0))
        self.assertTrue(meta.filepath.endswith(".wav"))
        self.assertEqual(meta.bpm, 85)
        self.assertEqual(meta.musical_key, "C Minor")

    def test_genre_presets(self):
        self.assertIn("Lo-Fi Hip-Hop", GENRE_PRESETS)
        self.assertIn("Cyberpunk Synthwave", GENRE_PRESETS)
        self.assertIn("Cinematic Orchestral", GENRE_PRESETS)


class TestFlaskEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_index_route(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Sonora", res.data)
        self.assertIn(b"Composition Studio", res.data)

    def test_system_info_route(self):
        res = self.client.get("/api/system-info")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "online")
        self.assertIn("genres", data)

    def test_generate_validation_error(self):
        res = self.client.post("/api/generate", json={"prompt": ""})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertFalse(data["success"])

    def test_generate_endpoint_success(self):
        res = self.client.post(
            "/api/generate",
            json={
                "prompt": "Cyberpunk driving bassline",
                "genre": "Cyberpunk Synthwave",
                "duration_seconds": 4,
                "engine": "algorithmic_synth",
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("track", data)
        self.assertTrue(data["track"]["filepath"].endswith(".wav"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
