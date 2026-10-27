"""
test_face_engine.py
===================
Unit tests for Experiment 6: Face Generation using DCGAN on CelebA Dataset.
"""

import os
import sys
import unittest
from pathlib import Path

# Add EXP 6 directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from face_engine import face_engine, FaceGenerationEngine
import numpy as np


class TestFaceEngine(unittest.TestCase):
    def setUp(self):
        self.engine = face_engine

    def test_system_info(self):
        """Test system telemetry and model weight status."""
        info = self.engine.get_system_info()
        self.assertIn("device", info)
        self.assertEqual(info["architecture"], "DCGAN (Radford et al., 2015)")
        self.assertEqual(info["dataset"], "CelebFaces Attributes (CelebA)")
        self.assertTrue(info["weights_exists"], "generator.pth weights file should exist")
        self.assertEqual(info["latent_dim"], 100)

    def test_single_face_generation(self):
        """Test generating a single 128x128 face with seed."""
        results = self.engine.generate_faces(count=1, seed=42, scale=2)
        self.assertEqual(len(results), 1)
        img, meta = results[0]
        self.assertEqual(img.size, (128, 128))
        self.assertEqual(meta.resolution, "128x128")
        self.assertEqual(meta.seed, 42)

        local_file = self.engine.output_dir / meta.filename
        self.assertTrue(local_file.exists(), f"Generated image file {local_file} should exist.")
        self.assertGreater(local_file.stat().st_size, 1000)

    def test_batch_generation(self):
        """Test generating 4 faces simultaneously."""
        results = self.engine.generate_faces(count=4, seed=100, scale=1)
        self.assertEqual(len(results), 4)
        for img, meta in results:
            self.assertEqual(img.size, (64, 64))
            self.assertTrue((self.engine.output_dir / meta.filename).exists())

    def test_latent_morphing(self):
        """Test smooth latent space interpolation between two seeds."""
        frames, morph_meta = self.engine.interpolate_faces(seed_a=11, seed_b=99, steps=8, fps=8, scale=1)
        self.assertEqual(len(frames), 8)
        gif_file = self.engine.output_dir / f"{morph_meta.id}.gif"
        self.assertTrue(gif_file.exists(), f"Morph GIF {gif_file} should exist.")
        self.assertGreater(gif_file.stat().st_size, 2000)

    def test_deterministic_reproducibility(self):
        """Test that identical seeds produce identical pixel matrices."""
        res1 = self.engine.generate_faces(count=1, seed=777, scale=1)
        res2 = self.engine.generate_faces(count=1, seed=777, scale=1)
        arr1 = np.array(res1[0][0])
        arr2 = np.array(res2[0][0])
        np.testing.assert_array_equal(arr1, arr2)


if __name__ == "__main__":
    unittest.main()
