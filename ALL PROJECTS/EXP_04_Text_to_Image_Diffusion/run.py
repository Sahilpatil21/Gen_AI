"""
run.py
======
Unified Launcher CLI for Exercise 4 (AI Image Generation).

Usage:
    python run.py               # Starts modern Web UI on http://127.0.0.1:5000
    python run.py --ui gradio   # Starts Gradio UI on http://127.0.0.1:7860
    python run.py --port 8080   # Custom port
"""

from __future__ import annotations

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Exercise 4 — AI Image Generation (Stable Diffusion v1.5 Studio)"
    )
    parser.add_argument(
        "--ui",
        choices=["web", "gradio"],
        default="web",
        help="UI framework to run: 'web' (Modern Glassmorphic Web App, default) or 'gradio' (Gradio Blocks)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port number to bind (default: 5000 for web, 7860 for gradio)",
    )

    args = parser.parse_args()

    if args.ui == "web":
        port = args.port or 5000
        print("\n" + "=" * 68)
        print("   LUMINA AI STUDIO - EXERCISE 4 (STABLE DIFFUSION v1.5)")
        print("=" * 68)
        print(f"   Mode: Modern Web Studio")
        print(f"   URL:  http://{args.host}:{port}")
        print("=" * 68 + "\n")
        from app import app
        app.run(host=args.host, port=port, debug=False)
    else:
        port = args.port or 7860
        print("\n" + "=" * 68)
        print("   GRADIO INTERFACE - EXERCISE 4 (STABLE DIFFUSION v1.5)")
        print("=" * 68)
        print(f"   Mode: Gradio Blocks UI")
        print(f"   URL:  http://{args.host}:{port}")
        print("=" * 68 + "\n")
        from gradio_app import build_gradio_ui
        demo = build_gradio_ui()
        demo.launch(server_name=args.host, server_port=port, share=False)


if __name__ == "__main__":
    main()
