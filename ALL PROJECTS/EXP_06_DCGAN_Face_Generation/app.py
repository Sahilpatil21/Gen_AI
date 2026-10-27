"""
app.py
======
Flask REST API and Web Application for Experiment 6:
Face Generation using GANs (DCGAN) on CelebA Dataset (by Dr. Sachin Balawant Takmare).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

from face_engine import face_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FaceGenApp")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
OUTPUTS_DIR = BASE_DIR / "outputs"

app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    template_folder=str(TEMPLATES_DIR),
)

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


@app.route("/")
def index():
    """Render the AI Face Generator Studio."""
    return render_template("index.html")


@app.route("/api/system-info", methods=["GET"])
def system_info():
    """Return model status, PyTorch device, and architecture details."""
    info = face_engine.get_system_info()
    return jsonify({"status": "online", "system": info})


@app.route("/api/history", methods=["GET"])
def history():
    """Return past generated faces."""
    return jsonify({"history": face_engine.history, "count": len(face_engine.history)})


@app.route("/api/generate", methods=["POST"])
def generate():
    """
    Generate synthetic celebrity faces from random Gaussian noise in latent space.
    Payload:
      - count: int (1 to 16, default 4)
      - seed: int or None (random if -1 or None)
      - temperature: float (0.2 to 2.0, default 1.0)
      - scale: int (1 for 64x64, 2 for 128x128, 4 for 256x256)
    """
    try:
        data = request.get_json(force=True) if request.is_json else request.form.to_dict()
    except Exception as e:
        return jsonify({"success": False, "error": f"Invalid request: {str(e)}"}), 400

    count = max(1, min(16, int(data.get("count", 4))))
    temperature = max(0.1, min(2.5, float(data.get("temperature", 1.0))))
    scale = int(data.get("scale", 2))
    if scale not in (1, 2, 4):
        scale = 2

    seed_val = data.get("seed")
    seed = None
    if seed_val not in (None, "", "null", -1):
        try:
            seed = int(seed_val)
        except (ValueError, TypeError):
            seed = None

    try:
        start_t = time.time()
        results = face_engine.generate_faces(count=count, seed=seed, temperature=temperature, scale=scale)
        elapsed_sec = round(time.time() - start_t, 3)

        faces_data = [asdict(meta) for _, meta in results]

        return jsonify({
            "success": True,
            "faces": faces_data,
            "count": len(faces_data),
            "elapsed_seconds": elapsed_sec,
        })
    except Exception as e:
        logger.exception("Face generation error:")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/morph", methods=["POST"])
def morph():
    """
    Interpolate between two identities in 100-D latent space.
    Payload:
      - seed_a: int (default 42)
      - seed_b: int (default 100)
      - steps: int (6 to 32, default 16)
      - fps: int (4 to 24, default 10)
      - scale: int (1, 2, 4)
    """
    try:
        data = request.get_json(force=True) if request.is_json else request.form.to_dict()
    except Exception as e:
        return jsonify({"success": False, "error": f"Invalid request: {str(e)}"}), 400

    seed_a = int(data.get("seed_a", 42))
    seed_b = int(data.get("seed_b", 100))
    steps = max(6, min(32, int(data.get("steps", 16))))
    fps = max(4, min(24, int(data.get("fps", 10))))
    scale = int(data.get("scale", 2))

    try:
        start_t = time.time()
        frames, morph_meta = face_engine.interpolate_faces(
            seed_a=seed_a,
            seed_b=seed_b,
            steps=steps,
            fps=fps,
            scale=scale,
            use_slerp=True,
        )
        elapsed_sec = round(time.time() - start_t, 3)

        return jsonify({
            "success": True,
            "morph": asdict(morph_meta),
            "elapsed_seconds": elapsed_sec,
        })
    except Exception as e:
        logger.exception("Latent morphing error:")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    """Serve output face images and animated morphs."""
    return send_from_directory(OUTPUTS_DIR, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5006))
    print("\n" + "=" * 70)
    print("   EXPERIMENT 6: CELEBA DCGAN FACE SYNTHESIS STUDIO")
    print("=" * 70)
    print(f"   Studio URL:    http://127.0.0.1:{port}")
    print(f"   Outputs Dir:   {OUTPUTS_DIR}")
    print("=" * 70 + "\n")
    app.run(host="127.0.0.1", port=port, debug=False)
