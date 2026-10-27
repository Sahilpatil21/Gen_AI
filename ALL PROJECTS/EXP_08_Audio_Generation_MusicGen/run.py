"""
run.py
======
Unified Launcher CLI for Experiment 8 (AI Music Generation with MusicGen).

Usage:
    python run.py               # Starts modern Web UI on http://127.0.0.1:5001
    python run.py --ui gradio   # Starts Gradio UI on http://127.0.0.1:7861
    python run.py --port 8080   # Run on custom port
"""

from __future__ import annotations

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 8 — AI Music Generation (MusicGen Foundation Transformer Studio)"
    )
    parser.add_argument(
        "--ui",
        choices=["web", "gradio"],
        default="web",
        help="UI framework to run: 'web' (Modern Web Studio, default) or 'gradio' (Gradio UI)",
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
        help="Port number to bind (default: 5001 for web, 7861 for gradio)",
    )

    args = parser.parse_args()

    if args.ui == "web":
        port = args.port or 5001
        print("\n" + "=" * 70)
        print("   SONORA AI MUSIC STUDIO — EXPERIMENT 8 (MUSICGEN)")
        print("=" * 70)
        print(f"   Mode: Modern Web Studio with Live Audio Visualizer")
        print(f"   URL:  http://{args.host}:{port}")
        print("=" * 70 + "\n")
        from app import app
        app.run(host=args.host, port=port, debug=False)
    else:
        port = args.port or 7861
        print("\n" + "=" * 70)
        print("   GRADIO INTERFACE — EXPERIMENT 8 (MUSICGEN)")
        print("=" * 70)
        print(f"   Mode: Gradio Blocks UI")
        print(f"   URL:  http://{args.host}:{port}")
        print("=" * 70 + "\n")
        from gradio_app import build_gradio_ui
        demo = build_gradio_ui()
        demo.launch(server_name=args.host, server_port=port, share=False)


if __name__ == "__main__":
    main()
