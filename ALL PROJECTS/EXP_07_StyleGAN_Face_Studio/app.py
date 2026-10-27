"""
app.py
======
Flask Web Studio for Experiment 7: StyleGAN CelebA Face Studio.
Serves the generated face samples, style mixing grids, truncation sweeps, and loss curves.
"""

from __future__ import annotations

import os
from pathlib import Path
from flask import Flask, jsonify, render_template, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results" / "notebook_outputs"

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

IMAGE_DESCRIPTIONS = {
    "00_real_samples.png": "CelebA dataset real ground-truth face samples (64x64)",
    "loss_curves.png": "Adversarial Generator and Discriminator training loss over 15 epochs",
    "noise_variation.png": "Stochastic variation: holding style latent w constant while injecting random per-pixel noise",
    "style_mixing_grid.png": "Style mixing: injecting source style at coarse layers and target style at fine layers",
    "truncation_trick.png": "Truncation trick: trading diversity (psi=1.0) for fidelity/quality (psi=0.4 to 0.1)",
}


def get_image_items():
    if not RESULTS_DIR.is_dir():
        return []

    items = []
    for f in sorted(RESULTS_DIR.glob("*.png")):
        name = f.name
        if name.startswith("epoch_"):
            num = name.replace("epoch_", "").replace(".png", "")
            title = f"Epoch {int(num):02d} Generated Faces"
            desc = f"Synthetic face samples generated after training epoch {int(num)}"
        else:
            title = name.replace(".png", "").replace("_", " ").title()
            desc = IMAGE_DESCRIPTIONS.get(name, "StyleGAN experiment visualization")

        items.append({
            "name": name,
            "title": title,
            "description": desc,
        })
    return items


@app.route("/")
def index():
    items = get_image_items()
    epoch_images = [item for item in items if item["name"].startswith("epoch_")]
    vis_images = [item for item in items if not item["name"].startswith("epoch_")]
    return render_template(
        "index.html",
        images=items,
        epoch_images=epoch_images,
        vis_images=vis_images,
    )


@app.route("/results/<path:filename>")
def serve_result(filename):
    return send_from_directory(str(RESULTS_DIR), filename)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "experiment": "EXP_07_StyleGAN_Face_Studio",
        "results_count": len(get_image_items()),
    })


@app.route("/api/system-info")
def system_info():
    import torch
    return jsonify({
        "status": "online",
        "device": "CUDA" if torch.cuda.is_available() else "CPU",
        "dataset": "CelebA (20,000 face images, 64x64)",
        "resolution": "64x64",
        "latent_z_dim": 512,
        "latent_w_dim": 512,
        "mapping_layers": 6,
        "results_available": len(get_image_items()),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5007))
    print(f"\n* Launching StyleGAN Face Studio on http://127.0.0.1:{port} ...\n")
    app.run(host="127.0.0.1", port=port, debug=False)
