"""
run.py
======
Unified runner for Experiment 5: Conditional VAE MNIST Digit Studio.
Usage:
  python run.py                 # Launches Web Studio on http://127.0.0.1:5005
  python run.py --port 8080     # Custom port
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))


def main():
    parser = argparse.ArgumentParser(description="Experiment 5: Conditional VAE MNIST Digit Generator")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5005,
        help="Port number to bind (default: 5005)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 68)
    print("   CONDITIONAL VAE MNIST STUDIO — EXPERIMENT 5")
    print("=" * 68)
    print("   Mode: Modern Web Studio")
    print(f"   URL:  http://{args.host}:{args.port}")
    print("=" * 68 + "\n")

    from app import app
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
