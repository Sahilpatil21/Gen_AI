"""
gradio_app.py
=============
Gradio UI interface for Exercise 4 (AI Image Generation).
Updated and hardened version of exe_4_image_generation.py with multi-engine support,
memory safety guards, and aspect ratio options.

Run: python gradio_app.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import gradio as gr
from pipeline_engine import (
    ASPECT_RATIOS,
    DEFAULT_NEGATIVE,
    STYLES,
    GenerationConfig,
    engine,
)


def generate_images_gradio(
    prompt: str,
    style: str,
    negative_prompt: str,
    aspect_ratio: str,
    num_images: float,
    num_inference_steps: float,
    guidance_scale: float,
    seed_input: float,
    engine_choice: str,
) -> Tuple[List[str], str, str]:
    """Execute generation and format outputs for Gradio gallery and file download."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Please enter a prompt describing the image you want.")

    width, height = ASPECT_RATIOS.get(aspect_ratio, (512, 512))
    seed = int(seed_input) if seed_input >= 0 else None

    # Map user-friendly engine choice
    engine_map = {
        "Auto (Smart Hardware Selection)": "auto",
        "Cloud AI (Fast 8K Synthesis)": "cloud_ai",
        "Local Diffusers (Stable Diffusion v1.5)": "local_diffusers",
        "Artistic Canvas (Offline Fallback)": "offline_canvas",
    }
    selected_engine = engine_map.get(engine_choice, "auto")

    config = GenerationConfig(
        prompt=prompt,
        style=style,
        negative_prompt=negative_prompt,
        num_images=int(num_images),
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        width=width,
        height=height,
        seed=seed,
        engine=selected_engine,
    )

    try:
        images, metadata_list, zip_rel_url = engine.generate(config)
        # Convert relative url to absolute local path for Gradio File component
        zip_path = str(engine.output_dir / Path(zip_rel_url).name)

        # Get filepaths for gallery
        image_paths = [str(engine.output_dir / m.filename) for m in metadata_list]

        meta0 = metadata_list[0]
        status = (
            f"Successfully generated {len(images)} image(s) in {meta0.generation_time_sec}s! | "
            f"Style: {style} | Engine: {meta0.engine} | Seed: {meta0.seed} | Device: {meta0.device}"
        )
        return image_paths, zip_path, status
    except Exception as e:
        raise gr.Error(f"Generation error: {str(e)}")


def build_gradio_ui() -> gr.Blocks:
    sys_info = engine.get_system_info()
    device_label = sys_info["device"]

    custom_theme = gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="blue",
        neutral_hue="slate",
    )

    with gr.Blocks(title="AI Image Generation Studio (SD v1.5)", theme=custom_theme) as demo:
        gr.Markdown(
            f"""
            # AI Image Generation Studio
            Create high-quality images from natural text descriptions using **Stable Diffusion v1.5**.
            
            **System Telemetry:** Running on `{device_label}` | Diffusers: `{'Available' if sys_info['diffusers_available'] else 'Not Installed'}` | PyTorch: `{'Available' if sys_info['torch_available'] else 'Not Installed'}`
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="E.g., A majestic snow leopard perched atop a misty Himalayan summit at golden hour",
                    lines=3,
                )
                with gr.Row():
                    style = gr.Dropdown(
                        label="Art Style",
                        choices=list(STYLES.keys()),
                        value="Realistic",
                    )
                    aspect_ratio = gr.Dropdown(
                        label="Aspect Ratio",
                        choices=list(ASPECT_RATIOS.keys()),
                        value="1:1 (Square)",
                    )

                with gr.Accordion("Advanced Parameters", open=False):
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="Elements to avoid (e.g., blurry, watermark, extra limbs)",
                        value=DEFAULT_NEGATIVE,
                        lines=2,
                    )
                    with gr.Row():
                        num_images = gr.Slider(
                            minimum=1,
                            maximum=4,
                            value=1,
                            step=1,
                            label="Batch Count (Images)",
                        )
                        num_inference_steps = gr.Slider(
                            minimum=10,
                            maximum=50,
                            value=25,
                            step=1,
                            label="Inference Steps",
                        )
                    with gr.Row():
                        guidance_scale = gr.Slider(
                            minimum=1.0,
                            maximum=20.0,
                            value=7.5,
                            step=0.5,
                            label="Guidance Scale (CFG)",
                        )
                        seed_input = gr.Number(
                            label="Seed (-1 for Random)",
                            value=-1,
                            precision=0,
                        )

                    engine_choice = gr.Radio(
                        label="Generation Engine",
                        choices=[
                            "Auto (Smart Hardware Selection)",
                            "Cloud AI (Fast 8K Synthesis)",
                            "Local Diffusers (Stable Diffusion v1.5)",
                            "Artistic Canvas (Offline Fallback)",
                        ],
                        value="Auto (Smart Hardware Selection)",
                    )

                with gr.Row():
                    generate_btn = gr.Button("Generate Images", variant="primary", size="lg")
                    clear_btn = gr.ClearButton(
                        components=[prompt, negative_prompt],
                        value="Clear Prompt",
                    )

            with gr.Column(scale=1):
                gallery = gr.Gallery(
                    label="Generated Images",
                    columns=2,
                    height=450,
                    object_fit="contain",
                    show_label=True,
                )
                download = gr.File(label="Download All Images (ZIP Archive)")
                status = gr.Textbox(label="Status & Diagnostics", interactive=False)

        generate_btn.click(
            fn=generate_images_gradio,
            inputs=[
                prompt,
                style,
                negative_prompt,
                aspect_ratio,
                num_images,
                num_inference_steps,
                guidance_scale,
                seed_input,
                engine_choice,
            ],
            outputs=[gallery, download, status],
        )

        gr.Markdown(
            """
            ---
            ### Pro Tips for Optimal Results:
            - **Be Descriptive**: Mention lighting (*golden hour, cinematic rim light*), environment (*misty mountains, cyberpunk alley*), and perspective (*macro close-up, wide-angle 35mm*).
            - **Inference Steps**: 20-25 steps offer the best balance of speed and photorealistic detail.
            - **Seed Locking**: Enter a fixed number for the seed if you want to reproduce or iterate on the exact same composition.
            """
        )

    return demo


if __name__ == "__main__":
    app = build_gradio_ui()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
