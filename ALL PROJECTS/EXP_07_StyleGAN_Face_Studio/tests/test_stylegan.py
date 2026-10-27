"""
test_stylegan.py
================
Unit tests for Experiment 7 StyleGAN model architecture and output directories.
"""

import sys
import unittest
from pathlib import Path
import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import MappingNetwork, Generator, Discriminator
import config


class TestStyleGAN(unittest.TestCase):
    def test_mapping_network(self):
        mapping = MappingNetwork(z_dim=64, w_dim=64, n_layers=4)
        z = torch.randn(2, 64)
        w = mapping(z)
        self.assertEqual(w.shape, (2, 64))

    def test_generator_forward(self):
        generator = Generator(w_dim=64)
        w = torch.randn(2, 64)
        img = generator(w)
        self.assertEqual(img.shape, (2, 3, 64, 64))

    def test_discriminator_forward(self):
        discriminator = Discriminator()
        x = torch.randn(2, 3, 64, 64)
        out = discriminator(x)
        self.assertEqual(out.shape, (2,))

    def test_results_exist(self):
        results_dir = Path(__file__).resolve().parent.parent / "results" / "notebook_outputs"
        self.assertTrue(results_dir.is_dir())
        images = list(results_dir.glob("*.png"))
        self.assertGreaterEqual(len(images), 10)


if __name__ == "__main__":
    unittest.main()
