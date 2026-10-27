"""
app.py
======
Flask Web Server for Experiment 10 (Text-to-Video Generation Foundation Studio).
Serves the modern cinema glassmorphic web UI and REST API endpoints.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

from video_engine import (
    CAMERA_MOTIONS,
    LIGHTING_PRESETS,
    SAMPLE_PROMPTS,
    VIDEO_STYLES,
    VideoConfig,
    video_engine,
)

logger = logging.getLogger("VideoGenApp")

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
    """Render the AI Video Generation Studio."""
    return render_template(
        "index.html",
        styles=VIDEO_STYLES,
        camera_motions=CAMERA_MOTIONS,
        lighting_presets=LIGHTING_PRESETS,
        sample_prompts=SAMPLE_PROMPTS,
    )


@app.route("/api/system-info", methods=["GET"])
def system_info():
    """Return hardware status, GPU info, and video presets."""
    info = video_engine.get_system_info()
    return jsonify({
        "status": "online",
        "system": info,
        "styles": list(VIDEO_STYLES.keys()),
        "camera_motions": list(CAMERA_MOTIONS.keys()),
        "lighting_presets": list(LIGHTING_PRESETS.keys()),
    })


@app.route("/api/history", methods=["GET"])
def history():
    """Return generated video library."""
    items = video_engine.get_history()
    return jsonify({"history": items, "count": len(items)})


@app.route("/api/generate", methods=["POST"])
def generate():
    """
    Generate video from text prompt.
    Payload:
      - prompt (str, required)
      - style (str, default 'Cinematic Photorealism')
      - camera_motion (str, default 'Cinematic Wide Pan')
      - lighting (str, default 'Golden Hour Sunset')
      - num_frames (int, 8-96; defaults to 48 frames, about 4 seconds at 12 FPS)
      - fps (int, 8-24)
      - resolution (str, e.g. '512x288')
      - motion_intensity (float, 1.0-10.0)
      - seed (int or None)
      - engine (str, 'auto', 'cloud_ai', 'local_diffusion', 'kinematic_render')
    """
    try:
        data = request.get_json(force=True) if request.is_json else request.form.to_dict()
    except Exception as e:
        return jsonify({"success": False, "error": f"Invalid request: {str(e)}"}), 400

    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"success": False, "error": "Please enter a descriptive video prompt."}), 400

    style = data.get("style", "Cinematic Photorealism")
    camera_motion = data.get("camera_motion", "Cinematic Wide Pan")
    lighting = data.get("lighting", "Golden Hour Sunset")
    num_frames = max(8, min(96, int(data.get("num_frames", 48))))
    fps = max(8, min(24, int(data.get("fps", 12))))
    motion_intensity = max(1.0, min(10.0, float(data.get("motion_intensity", 5.0))))

    res_str = data.get("resolution", "512x288")
    if res_str == "384x256":
        w, h = 384, 256
    elif res_str == "256x256":
        w, h = 256, 256
    elif res_str == "288x512":
        w, h = 288, 512
    else:
        w, h = 512, 288

    seed_val = data.get("seed")
    seed = None
    if seed_val not in (None, "", "null", -1):
        try:
            seed = int(seed_val)
        except (ValueError, TypeError):
            seed = None

    selected_engine = data.get("engine", "auto")

    config = VideoConfig(
        prompt=prompt,
        style=style,
        camera_motion=camera_motion,
        lighting=lighting,
        num_frames=num_frames,
        fps=fps,
        width=w,
        height=h,
        motion_intensity=motion_intensity,
        seed=seed,
        engine=selected_engine,
    )

    try:
        start_t = time.time()
        frames, vid_url, meta = video_engine.generate(config)
        elapsed = round(time.time() - start_t, 2)

        return jsonify({
            "success": True,
            "video": meta.to_dict(),
            "elapsed_seconds": elapsed,
        })
    except Exception as e:
        logger.exception("Video generation error:")
        return jsonify({
            "success": False,
            "error": str(e),
            "suggestion": "Try setting engine to 'Kinematic Temporal Renderer' or reducing resolution."
        }), 500


@app.route("/static/outputs/<path:filename>")
def serve_output(filename):
    """Serve output video clips and frames."""
    return send_from_directory(OUTPUTS_DIR, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    print("\n" + "=" * 70)
    print("   EXPERIMENT 10: TEXT-TO-VIDEO GENERATION FOUNDATION STUDIO")
    print("=" * 70)
    print(f"   Studio URL:    http://127.0.0.1:{port}")
    print(f"   Outputs Dir:   {OUTPUTS_DIR}")
    print("=" * 70 + "\n")
    app.run(host="127.0.0.1", port=port, debug=False)
