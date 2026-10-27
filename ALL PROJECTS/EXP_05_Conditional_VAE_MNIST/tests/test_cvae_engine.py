"""
test_cvae_engine.py
===================
Unit tests for Experiment 5 Conditional VAE model and endpoints.
"""

import sys
import unittest
from pathlib import Path

import importlib.util

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

_spec = importlib.util.spec_from_file_location("exp05_app", BASE_DIR / "app.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["exp05_app"] = _mod
_spec.loader.exec_module(_mod)
app = _mod.app
load_decoder = _mod.load_decoder


class TestCVAEEngine(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_load_decoder(self):
        decoder = load_decoder()
        self.assertIsNotNone(decoder)
        self.assertEqual(len(decoder.input_shape), 2)  # [z, label]

    def test_health_endpoint(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["model_loaded"])

    def test_index_route(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Conditional VAE", res.data)

    def test_generate_digit(self):
        res = self.client.post("/generate", json={"digit": 7})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["selected_digit"], 7)
        self.assertIn("image", data)
        self.assertGreater(len(data["image"]), 100)

    def test_generate_invalid_digit(self):
        res = self.client.post("/generate", json={"digit": 99})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertFalse(data["success"])


if __name__ == "__main__":
    unittest.main()
