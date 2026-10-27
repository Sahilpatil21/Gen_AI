"""
gradio_app.py
=============
Gradio UI interface for Experiment 8 (AI Music Generation with MusicGen).

Run: python gradio_app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import gradio as gr
from music_engine import (
    GENRE_PRESETS,
    MusicConfig,
    music_engine,
)


def generate_music_gradio(
    prompt: str,
    genre: str,
    duration: float,
    temperature: float,
    top_k: float,
    guidance: float,
    seed_val: float,
    engine_choice: str,
) -> Tuple[str, str]:
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Please enter a music prompt description.")

    seed = int(seed_val) if seed_val >= 0 else None

    engine_map = {
        "Auto (Smart Selection)": "auto",
        "Local MusicGen Transformer (Meta AI)": "local_musicgen",
        "Algorithmic Harmonic Synth (Instant Multi-Track)": "algorithmic_synth",
        "Cloud AI Music Engine": "cloud_ai",
    }
    selected_engine = engine_map.get(engine_choice, "auto")

    config = MusicConfig(
        prompt=prompt,
        genre=genre,
        duration_seconds=int(duration),
        temperature=float(temperature),
        top_k=int(top_k),
        guidance_scale=float(guidance),
        seed=seed,
        engine=selected_engine,
    )

    try:
        audio_norm, sr, meta = music_engine.generate(config)
        audio_path = str(music_engine.output_dir / meta.filename)

        status = (
            f"Successfully generated '{meta.title}' ({meta.duration_seconds}s) in {meta.generation_time_sec}s! | "
            f"Genre: {meta.genre} | Key: {meta.musical_key} | BPM: {meta.bpm} | Engine: {meta.engine}"
        )
        return audio_path, status
    except Exception as e:
        raise gr.Error(f"Music generation error: {str(e)}")


def build_gradio_ui() -> gr.Blocks:
    sys_info = music_engine.get_system_info()
    device_label = sys_info["device"]

    custom_theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="cyan",
        neutral_hue="slate",
    )

    with gr.Blocks(title="AI Music Studio — MusicGen (Experiment 8)", theme=custom_theme) as demo:
        gr.Markdown(
            f"""
            # AI Music Generation Studio (MusicGen)
            Generate novel, continuous musical compositions from natural text prompts using Meta AI's **MusicGen** foundation transformer.
            
            **System Telemetry:** Running on `{device_label}` | Audio Codec: `EnCodec (32 kHz)` | Transformers: `Ready`
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Music Description Prompt",
                    placeholder="E.g., An upbeat lo-fi hip-hop track with a relaxed piano melody, vinyl crackle, and chill groove",
                    lines=3,
                    value="An upbeat lo-fi hip-hop track with a relaxed piano melody",
                )
                genre = gr.Dropdown(
                    label="Musical Genre & Texture",
                    choices=list(GENRE_PRESETS.keys()),
                    value="Lo-Fi Hip-Hop",
                )
                duration = gr.Slider(
                    minimum=5,
                    maximum=30,
                    value=10,
                    step=1,
                    label="Duration (Seconds)",
                )

                with gr.Accordion("Advanced Synthesis Tuning", open=False):
                    with gr.Row():
                        temperature = gr.Slider(
                            minimum=0.2,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="Sampling Temperature",
                        )
                        top_k = gr.Slider(
                            minimum=50,
                            maximum=500,
                            value=250,
                            step=10,
                            label="Top-K Audio Tokens",
                        )
                    with gr.Row():
                        guidance = gr.Slider(
                            minimum=1.0,
                            maximum=10.0,
                            value=3.0,
                            step=0.5,
                            label="Classifier-Free Guidance (CFG)",
                        )
                        seed_val = gr.Number(
                            label="Seed (-1 for Random)",
                            value=-1,
                            precision=0,
                        )
                    engine_choice = gr.Radio(
                        label="Synthesis Engine",
                        choices=[
                            "Auto (Smart Selection)",
                            "Local MusicGen Transformer (Meta AI)",
                            "Algorithmic Harmonic Synth (Instant Multi-Track)",
                            "Cloud AI Music Engine",
                        ],
                        value="Auto (Smart Selection)",
                    )

                with gr.Row():
                    generate_btn = gr.Button("Compose Track", variant="primary", size="lg")
                    clear_btn = gr.ClearButton(components=[prompt], value="Clear")

            with gr.Column(scale=1):
                audio_player = gr.Audio(
                    label="Synthesized Audio Player",
                    type="filepath",
                )
                status_box = gr.Textbox(label="Composition Diagnostics", interactive=False)

        generate_btn.click(
            fn=generate_music_gradio,
            inputs=[prompt, genre, duration, temperature, top_k, guidance, seed_val, engine_choice],
            outputs=[audio_player, status_box],
        )

        gr.Markdown(
            """
            ---
            ### Experiment 8 Technical Foundations:
            - **Neural Audio Codec (EnCodec)**: Discretizes continuous 32 kHz audio into multi-stream acoustic codebooks at 50 Hz.
            - **Text-Conditioned Autoregressive Transformer**: Cross-attends to frozen T5 text representations and predicts subsequent audio tokens ("autocomplete for sound").
            """
        )

    return demo


if __name__ == "__main__":
    app = build_gradio_ui()
    app.launch(server_name="127.0.0.1", server_port=7861, share=False)
