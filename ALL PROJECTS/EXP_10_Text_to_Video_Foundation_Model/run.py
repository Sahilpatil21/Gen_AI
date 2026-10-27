"""
run.py
======
Unified runner for Experiment 10: AI Text-to-Video Generation Studio.
Options:
  python run.py                 # Launches Flask Web Studio on http://127.0.0.1:5002
  python run.py --ui gradio     # Launches Gradio Interface on http://127.0.0.1:7862
  python run.py --port 5002     # Custom port
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))


def main():
    parser = argparse.ArgumentParser(description="Experiment 10: Text-to-Video Generation Studio")
    parser.add_argument(
        "--ui",
        choices=["web", "gradio"],
        default="web",
        help="UI framework: 'web' (Flask modern glassmorphism) or 'gradio' (Gradio blocks)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Custom port (default 5002 for web, 7862 for gradio)",
    )

    args = parser.parse_args()

    if args.ui == "gradio":
        port = args.port or 7862
        from gradio_app import create_gradio_demo
        demo = create_gradio_demo()
        print(f"\nLaunching CineGen Gradio Studio on http://127.0.0.1:{port} ...\n")
        demo.launch(server_name="127.0.0.1", server_port=port, share=False)
    else:
        port = args.port or 5002
        from app import app
        print(f"\nLaunching CineGen Cinema Web Studio on http://127.0.0.1:{port} ...\n")
        app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
