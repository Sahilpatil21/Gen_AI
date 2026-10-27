"""
app.py
======
Flask Web Server for Exercise 4 - AI Image Generation Studio.
Provides REST API endpoints and serves the modern glassmorphic web UI.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

from pipeline_engine import (
    ASPECT_RATIOS,
    DEFAULT_NEGATIVE,
    PROMPT_SUGGESTIONS,
    STYLES,
    GenerationConfig,
    engine,
)

logger = logging.getLogger("ImageGenApp")

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
    """Render the main AI Image Generation Studio UI."""
    return render_template(
        "index.html",
        styles=STYLES,
        aspect_ratios=ASPECT_RATIOS,
        default_negative=DEFAULT_NEGATIVE,
        prompt_suggestions=PROMPT_SUGGESTIONS
    )


@app.route("/api/system-info", methods=["GET"])
def system_info():
    """Return hardware diagnostics and model readiness."""
    info = engine.get_system_info()
    return jsonify({
        "status": "online",
        "system": info,
        "styles": list(STYLES.keys()),
        "aspect_ratios": list(ASPECT_RATIOS.keys())
    })


@app.route("/api/presets", methods=["GET"])
def presets():
    """Return style presets and curated prompt suggestions."""
    return jsonify({
        "styles": STYLES,
        "aspect_ratios": list(ASPECT_RATIOS.keys()),
        "prompts": PROMPT_SUGGESTIONS,
        "default_negative": DEFAULT_NEGATIVE
    })


@app.route("/api/history", methods=["GET"])
def history():
    """Return generation history."""
    items = engine.get_history()
    return jsonify({"history": items, "count": len(items)})


@app.route("/api/generate", methods=["POST"])
def generate():
    """
    Generate images from text prompt with selected style and settings.
    Expects JSON payload with:
      - prompt (str, required)
      - style (str, default 'Realistic')
      - negative_prompt (str)
      - num_images (int, 1-4)
      - num_inference_steps (int, 10-50)
      - guidance_scale (float, 1.0-20.0)
      - aspect_ratio (str, e.g. '1:1 (Square)')
      - seed (int or None)
      - engine (str, 'auto', 'cloud_ai', 'local_diffusers', 'offline_canvas')
    """
    try:
        data = request.get_json(force=True) if request.is_json else request.form.to_dict()
    except Exception as e:
        return jsonify({"success": False, "error": f"Invalid request body: {str(e)}"}), 400

    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"success": False, "error": "Please enter a descriptive prompt."}), 400

    style = data.get("style", "Realistic")
    negative_prompt = data.get("negative_prompt", DEFAULT_NEGATIVE)
    num_images = max(1, min(4, int(data.get("num_images", 1))))
    steps = max(10, min(50, int(data.get("num_inference_steps", 25))))
    guidance = max(1.0, min(20.0, float(data.get("guidance_scale", 7.5))))

    aspect_ratio_str = data.get("aspect_ratio", "1:1 (Square)")
    width, height = ASPECT_RATIOS.get(aspect_ratio_str, (512, 512))

    seed_val = data.get("seed")
    seed = None
    if seed_val not in (None, "", "null", -1):
        try:
            seed = int(seed_val)
        except (ValueError, TypeError):
            seed = None

    selected_engine = data.get("engine", "auto")

    config = GenerationConfig(
        prompt=prompt,
        style=style,
        negative_prompt=negative_prompt,
        num_images=num_images,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=width,
        height=height,
        seed=seed,
        engine=selected_engine
    )

    try:
        start_t = time.time()
        images, metadata_list, zip_url = engine.generate(config)
        elapsed = round(time.time() - start_t, 2)

        return jsonify({
            "success": True,
            "images": [asdict(m) for m in metadata_list],
            "zip_url": zip_url,
            "elapsed_seconds": elapsed,
            "count": len(images)
        })
    except Exception as e:
        logger.exception("Generation error:")
        return jsonify({
            "success": False,
            "error": str(e),
            "suggestion": "Try switching the Engine to 'Cloud AI' in settings if your local GPU ran out of memory."
        }), 500


@app.route("/static/outputs/<path:filename>")
def serve_output(filename):
    """Serve output images and download archives."""
    return send_from_directory(OUTPUTS_DIR, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "=" * 65)
    print("  EXERCISE 4: AI IMAGE GENERATION STUDIO (Stable Diffusion)")
    print("=" * 65)
    print(f"  Web UI running at: http://127.0.0.1:{port}")
    print(f"  Outputs stored in: {OUTPUTS_DIR}")
    print("=" * 65 + "\n")
    app.run(host="127.0.0.1", port=port, debug=False)
