"""
gradio_app.py
=============
Gradio Web Interface for Experiment 6:
Face Generation using GANs (DCGAN) on CelebA Dataset (by Dr. Sachin Balawant Takmare).
"""

from __future__ import annotations

import os
from pathlib import Path
import gradio as gr
from face_engine import face_engine

BASE_DIR = Path(__file__).resolve().parent


def generate_faces_gradio(count: int, seed: int, temperature: float, resolution: str):
    """Gradio handler for batch face synthesis."""
    scale = 1 if resolution == "64x64 (Native)" else (4 if resolution == "256x256 (Ultra)" else 2)
    actual_seed = None if seed in (None, -1, "") else int(seed)

    results = face_engine.generate_faces(
        count=int(count),
        seed=actual_seed,
        temperature=float(temperature),
        scale=scale,
    )

    images = [img for img, _ in results]
    first_meta = results[0][1]
    meta_info = (
        f"**Batch Size:** {len(results)} faces\n\n"
        f"**Resolution:** {first_meta.resolution}\n\n"
        f"**Latent Dimension:** {first_meta.latent_dim}-D Gaussian\n\n"
        f"**Truncation (ψ):** {first_meta.temperature}\n\n"
        f"**Base Seed:** {first_meta.seed}\n\n"
        f"**Device:** {first_meta.device}"
    )
    return images, meta_info


def morph_faces_gradio(seed_a: int, seed_b: int, steps: int, fps: int):
    """Gradio handler for latent space morphing."""
    frames, morph_meta = face_engine.interpolate_faces(
        seed_a=int(seed_a),
        seed_b=int(seed_b),
        steps=int(steps),
        fps=int(fps),
        scale=2,
    )

    gif_path = str(face_engine.output_dir / f"{morph_meta.id}.gif")
    meta_info = (
        f"**Trajectory ID:** `{morph_meta.id}`\n\n"
        f"**Identity A Seed:** {morph_meta.seed_a} | **Identity B Seed:** {morph_meta.seed_b}\n\n"
        f"**Geodesic Steps:** {morph_meta.steps} intermediate latent points\n\n"
        f"**Duration:** {morph_meta.duration_sec}s @ {morph_meta.fps} FPS\n\n"
        f"**Interpolation:** Spherical Linear Interpolation (SLERP)"
    )
    return gif_path, frames, meta_info


def create_gradio_demo():
    custom_css = """
    .gradio-container {
        font-family: 'Inter', -apple-system, sans-serif !important;
        max-width: 1150px !important;
        margin: auto;
    }
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #f59e0b 0%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem !important;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    """

    with gr.Blocks(title="FaceGen DCGAN Studio", css=custom_css, theme=gr.themes.Soft()) as demo:
        gr.HTML(
            """
            <div style="text-align:center; padding: 1.2rem 0;">
                <h1 class="main-title">🧑 FaceGen: DCGAN Face Synthesis Studio</h1>
                <p class="sub-title">Experiment 6 · Deep Convolutional GAN trained on CelebA Dataset (~200,000 Celebrity Faces)</p>
            </div>
            """
        )

        with gr.Tabs():
            # Tab 1: Synthesis
            with gr.TabItem("🧑 Face Synthesis Studio"):
                with gr.Row():
                    with gr.Column(scale=4):
                        count = gr.Slider(minimum=1, maximum=16, step=1, value=4, label="Batch Size (Faces)")
                        resolution = gr.Dropdown(
                            choices=["64x64 (Native)", "128x128 (Enhanced HD)", "256x256 (Ultra)"],
                            value="128x128 (Enhanced HD)",
                            label="Output Resolution",
                        )
                        temperature = gr.Slider(
                            minimum=0.3,
                            maximum=1.8,
                            step=0.05,
                            value=1.0,
                            label="Truncation Factor ψ (Diversity vs Quality)",
                        )
                        seed = gr.Number(value=-1, label="Seed (-1 for Dynamic Noise)", precision=0)

                        generate_btn = gr.Button("🧑 Generate Synthetic Faces", variant="primary", size="lg")

                    with gr.Column(scale=7):
                        gallery = gr.Gallery(label="Generated Celebrity Faces", columns=4, height="auto")
                        info_box = gr.Markdown(label="Generation Telemetry")

                generate_btn.click(
                    fn=generate_faces_gradio,
                    inputs=[count, seed, temperature, resolution],
                    outputs=[gallery, info_box],
                )

            # Tab 2: Morphing
            with gr.TabItem("🌀 Latent Space Morphing Lab"):
                with gr.Row():
                    with gr.Column(scale=4):
                        with gr.Row():
                            seed_a = gr.Number(value=42, label="Identity A Seed", precision=0)
                            seed_b = gr.Number(value=100, label="Identity B Seed", precision=0)

                        steps = gr.Slider(minimum=8, maximum=32, step=4, value=16, label="Intermediate Steps")
                        fps = gr.Slider(minimum=6, maximum=24, step=2, value=12, label="Frame Rate (FPS)")

                        morph_btn = gr.Button("🌀 Compute Latent Morphing", variant="primary", size="lg")

                    with gr.Column(scale=7):
                        morph_output = gr.Image(label="Latent Morphing Animation (GIF / Loop)")
                        morph_info = gr.Markdown(label="Trajectory Metrics")
                        with gr.Accordion("Intermediate Geodesic Frames", open=False):
                            morph_gallery = gr.Gallery(label="Morph Frames", columns=4, height="auto")

                morph_btn.click(
                    fn=morph_faces_gradio,
                    inputs=[seed_a, seed_b, steps, fps],
                    outputs=[morph_output, morph_info, morph_gallery],
                )

    return demo


if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_PORT", 7866))
    demo = create_gradio_demo()
    print(f"\n[FaceGen Gradio] Starting on http://127.0.0.1:{port}\n")
    demo.launch(server_name="127.0.0.1", server_port=port, share=False)
