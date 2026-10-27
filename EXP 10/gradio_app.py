"""
gradio_app.py
=============
Gradio Web Interface for Experiment 10:
Generation of Videos from Text Prompts using a Pretrained Open-Source Video
Generation Foundation Model.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict
from pathlib import Path

import gradio as gr

from video_engine import (
    CAMERA_MOTIONS,
    LIGHTING_PRESETS,
    SAMPLE_PROMPTS,
    VIDEO_STYLES,
    VideoConfig,
    video_engine,
)

BASE_DIR = Path(__file__).resolve().parent


def generate_video_gradio(
    prompt: str,
    style: str,
    camera_motion: str,
    lighting: str,
    resolution: str,
    num_frames: int,
    fps: int,
    motion_intensity: float,
    seed: int | None,
    engine: str,
):
    """Gradio handler for video generation."""
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a descriptive video prompt.")

    if resolution == "384x256":
        w, h = 384, 256
    elif resolution == "256x256":
        w, h = 256, 256
    elif resolution == "288x512":
        w, h = 288, 512
    else:
        w, h = 512, 288

    actual_seed = None if seed in (None, -1, "") else int(seed)

    config = VideoConfig(
        prompt=prompt.strip(),
        style=style,
        camera_motion=camera_motion,
        lighting=lighting,
        num_frames=int(num_frames),
        fps=int(fps),
        width=w,
        height=h,
        motion_intensity=float(motion_intensity),
        seed=actual_seed,
        engine=engine,
    )

    start_t = time.time()
    frames, vid_path, meta = video_engine.generate(config)
    elapsed = round(time.time() - start_t, 2)

    meta_text = (
        f"**Video ID:** `{meta.video_id}`\n\n"
        f"**Engine:** {meta.engine}\n\n"
        f"**Duration:** {meta.duration_seconds:.2f}s ({meta.num_frames} frames @ {meta.fps} FPS)\n\n"
        f"**Resolution:** {meta.resolution}\n\n"
        f"**Generation Latency:** {elapsed}s\n\n"
        f"**Camera:** {meta.camera_motion} | **Lighting:** {meta.lighting}"
    )

    return vid_path, frames, meta_text


def create_gradio_demo():
    """Build Gradio Blocks Interface."""
    custom_css = """
    .gradio-container {
        font-family: 'Inter', -apple-system, sans-serif !important;
        max-width: 1200px !important;
        margin: auto;
    }
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%);
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

    with gr.Blocks(title="CineGen AI Video Studio", css=custom_css, theme=gr.themes.Soft()) as demo:
        gr.HTML(
            """
            <div style="text-align:center; padding: 1.5rem 0;">
                <h1 class="main-title">🎬 CineGen: Text-to-Video Foundation Studio</h1>
                <p class="sub-title">Experiment 10 · Open-Source Video Generation Foundation Model & Temporal Synthesis</p>
            </div>
            """
        )

        with gr.Row():
            # Left Controls
            with gr.Column(scale=5):
                prompt = gr.Textbox(
                    label="Scene Prompt",
                    placeholder="Describe scene, motion, camera, and lighting...",
                    value="A small boat sailing across a calm lake at sunrise, gentle water movement, cinematic wide shot",
                    lines=3,
                )

                with gr.Row():
                    style = gr.Dropdown(
                        choices=list(VIDEO_STYLES.keys()),
                        value="Cinematic Photorealism",
                        label="Visual Style",
                    )
                    resolution = gr.Dropdown(
                        choices=["512x288", "384x256", "256x256", "288x512"],
                        value="512x288",
                        label="Resolution (16:9 / 3:2 / 1:1 / 9:16)",
                    )

                with gr.Row():
                    camera_motion = gr.Dropdown(
                        choices=list(CAMERA_MOTIONS.keys()),
                        value="Cinematic Wide Pan",
                        label="Camera Movement",
                    )
                    lighting = gr.Dropdown(
                        choices=list(LIGHTING_PRESETS.keys()),
                        value="Golden Hour Sunset",
                        label="Lighting & Atmosphere",
                    )

                with gr.Accordion("Temporal Tuning & Foundation Engine", open=False):
                    with gr.Row():
                        num_frames = gr.Slider(
                            minimum=24,
                            maximum=96,
                            step=8,
                            value=48,
                            label="Frames Count (4s at 12 FPS by default)",
                        )
                        fps = gr.Slider(
                            minimum=8,
                            maximum=24,
                            step=2,
                            value=12,
                            label="Frame Rate (FPS)",
                        )

                    with gr.Row():
                        motion_intensity = gr.Slider(
                            minimum=1.0,
                            maximum=10.0,
                            step=0.5,
                            value=5.0,
                            label="Motion Dynamics",
                        )
                        seed = gr.Number(
                            value=-1,
                            label="Seed (-1 for random)",
                            precision=0,
                        )

                    engine = gr.Radio(
                        choices=["auto", "kinematic_render", "local_diffusion", "cloud_ai"],
                        value="auto",
                        label="Synthesis Pipeline Engine",
                    )

                generate_btn = gr.Button("🎬 Generate Video Sequence", variant="primary", size="lg")

                gr.Examples(
                    examples=[[p] for p in SAMPLE_PROMPTS],
                    inputs=[prompt],
                    label="Curated Lab Manual Prompts",
                )

            # Right Viewport
            with gr.Column(scale=6):
                video_output = gr.Video(label="Screening Preview (H.264 MP4)", autoplay=True, loop=True)
                meta_output = gr.Markdown(label="Sequence Diagnostics")

                with gr.Accordion("Individual Video Frame Sequence (Filmstrip)", open=False):
                    gallery_output = gr.Gallery(label="Generated Latent Frames", columns=4, height="auto")

        generate_btn.click(
            fn=generate_video_gradio,
            inputs=[
                prompt,
                style,
                camera_motion,
                lighting,
                resolution,
                num_frames,
                fps,
                motion_intensity,
                seed,
                engine,
            ],
            outputs=[video_output, gallery_output, meta_output],
        )

    return demo


if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_PORT", 7862))
    demo = create_gradio_demo()
    print(f"\n[CineGen Gradio] Starting on http://127.0.0.1:{port}\n")
    demo.launch(server_name="127.0.0.1", server_port=port, share=False)
