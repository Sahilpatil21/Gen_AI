"""
run.py
======
Unified runner for Experiment 6: DCGAN Face Synthesis & Latent Morphing Studio.
Usage:
  python run.py                 # Launches Modern Web Studio on http://127.0.0.1:5006
  python run.py --ui gradio     # Launches Gradio Interface on http://127.0.0.1:7866
  python run.py --cli           # Generates faces via CLI
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))


def main():
    parser = argparse.ArgumentParser(description="Experiment 6: DCGAN Face Synthesis Studio")
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
        help="Custom port (default 5006 for web, 7866 for gradio)",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run direct CLI generation instead of launching server",
    )
    parser.add_argument("--count", type=int, default=4, help="CLI face count (default: 4)")
    parser.add_argument("--seed", type=int, default=None, help="CLI seed")

    args = parser.parse_args()

    if args.cli:
        from face_engine import face_engine
        print(f"\n[FaceGen CLI] Generating {args.count} faces with seed={args.seed} ...")
        results = face_engine.generate_faces(count=args.count, seed=args.seed, scale=2)
        for _, meta in results:
            print(f" ✓ Saved: {meta.filename} ({meta.resolution}, seed={meta.seed}) in {meta.generation_time_ms}ms")
        return

    if args.ui == "gradio":
        port = args.port or 7866
        from gradio_app import create_gradio_demo
        demo = create_gradio_demo()
        print(f"\n🧑 Launching FaceGen Gradio Studio on http://127.0.0.1:{port} ...\n")
        demo.launch(server_name="127.0.0.1", server_port=port, share=False)
    else:
        port = args.port or 5006
        from app import app
        print(f"\n🧑 Launching FaceGen Modern Cinema Studio on http://127.0.0.1:{port} ...\n")
        app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
