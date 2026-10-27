"""
app.py
======
Flask Web Server for Experiment 8 (AI Music Generation with MusicGen).
Serves the dark-mode glassmorphic Music Studio UI and REST API.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

from music_engine import (
    GENRE_PRESETS,
    MusicConfig,
    music_engine,
)

logger = logging.getLogger("MusicGenApp")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
OUTPUTS_DIR = STATIC_DIR / "outputs"

app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    template_folder=str(TEMPLATES_DIR)
)

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


@app.route("/")
def index():
    """Render the AI Music Generation Studio."""
    return render_template(
        "index.html",
        genres=GENRE_PRESETS,
    )


@app.route("/api/system-info", methods=["GET"])
def system_info():
    """Return hardware status, GPU info, and genre presets."""
    info = music_engine.get_system_info()
    return jsonify({
        "status": "online",
        "system": info,
        "genres": GENRE_PRESETS
    })


@app.route("/api/history", methods=["GET"])
def history():
    """Return generated music tracks."""
    tracks = music_engine.get_history()
    return jsonify({"history": tracks, "count": len(tracks)})


@app.route("/api/generate", methods=["POST"])
def generate():
    """
    Generate musical audio from a text prompt.
    Payload:
      - prompt (str, required)
      - genre (str, default 'Lo-Fi Hip-Hop')
      - duration_seconds (int, 5-30)
      - temperature (float, 0.5-2.0)
      - top_k (int, 50-500)
      - guidance_scale (float, 1.0-10.0)
      - seed (int or None)
      - engine (str, 'auto', 'local_musicgen', 'cloud_ai', 'algorithmic_synth')
    """
    try:
        data = request.get_json(force=True) if request.is_json else request.form.to_dict()
    except Exception as e:
        return jsonify({"success": False, "error": f"Invalid request body: {str(e)}"}), 400

    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"success": False, "error": "Please enter a descriptive music prompt."}), 400

    genre = data.get("genre", "Lo-Fi Hip-Hop")
    duration = max(3, min(30, int(data.get("duration_seconds", 10))))
    temperature = max(0.2, min(2.5, float(data.get("temperature", 1.0))))
    top_k = max(20, min(500, int(data.get("top_k", 250))))
    guidance = max(1.0, min(10.0, float(data.get("guidance_scale", 3.0))))

    seed_val = data.get("seed")
    seed = None
    if seed_val not in (None, "", "null", -1):
        try:
            seed = int(seed_val)
        except (ValueError, TypeError):
            seed = None

    selected_engine = data.get("engine", "auto")

    config = MusicConfig(
        prompt=prompt,
        genre=genre,
        duration_seconds=duration,
        temperature=temperature,
        top_k=top_k,
        guidance_scale=guidance,
        seed=seed,
        engine=selected_engine,
    )

    try:
        start_t = time.time()
        audio_norm, sr, meta = music_engine.generate(config)
        elapsed = round(time.time() - start_t, 2)

        return jsonify({
            "success": True,
            "track": asdict(meta),
            "elapsed_seconds": elapsed,
        })
    except Exception as e:
        logger.exception("Music generation error:")
        return jsonify({
            "success": False,
            "error": str(e),
            "suggestion": "Try selecting 'Algorithmic Synth' or 'Cloud AI' if local model is downloading or out of memory."
        }), 500


@app.route("/static/outputs/<path:filename>")
def serve_output(filename):
    """Serve output audio tracks."""
    return send_from_directory(OUTPUTS_DIR, filename, mimetype="audio/wav")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print("\n" + "=" * 68)
    print("   EXPERIMENT 8: AI MUSIC GENERATION STUDIO (MusicGen)")
    print("=" * 68)
    print(f"   Studio URL:   http://127.0.0.1:{port}")
    print(f"   Audio Output: {OUTPUTS_DIR}")
    print("=" * 68 + "\n")
    app.run(host="127.0.0.1", port=port, debug=False)
