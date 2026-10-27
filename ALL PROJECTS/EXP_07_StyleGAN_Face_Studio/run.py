"""
run.py
======
Unified runner for Experiment 7: StyleGAN CelebA Face Studio.
Usage:
  python run.py                 # Launches Modern Web Gallery on http://127.0.0.1:5007
  python run.py --ui streamlit  # Launches Streamlit Gallery on http://127.0.0.1:8507
  python run.py --port 8080     # Custom port
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))


def main():
    parser = argparse.ArgumentParser(description="Experiment 7: StyleGAN Face Studio Gallery")
    parser.add_argument(
        "--ui",
        choices=["web", "streamlit"],
        default="web",
        help="UI mode: 'web' (Flask modern gallery, default) or 'streamlit'",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port number to bind (default 5007 for web, 8507 for streamlit)",
    )

    args = parser.parse_args()

    if args.ui == "streamlit":
        port = args.port or 8507
        app_path = str(BASE_DIR / "streamlit_app.py")
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            app_path,
            "--server.port",
            str(port),
            "--server.headless",
            "true",
        ]
        print(f"\n* Launching StyleGAN Streamlit Gallery on http://127.0.0.1:{port} ...\n")
        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        port = args.port or 5007
        from app import app
        print("\n" + "=" * 68)
        print("   STYLEGAN CELEBA STUDIO GALLERY — EXPERIMENT 7")
        print("=" * 68)
        print("   Mode: Modern Web Gallery")
        print(f"   URL:  http://127.0.0.1:{port}")
        print("=" * 68 + "\n")
        app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
