# Experiment 8: AI Music Generation with MusicGen

A state-of-the-art Text-to-Music Generation Application powered by Meta AI's **MusicGen** foundation model (`facebook/musicgen-small`). Built for the Generative AI Course Experiment 8 with a cyber-acoustic dark-mode Web Studio, real-time Web Audio API waveform visualizer, and multi-engine synthesis.

---

## 🌟 What is Experiment 8?

### Problem Statement
Generate **novel, continuous instrumental music compositions** conditioned on a **natural language description of genre and mood** (e.g., *"an upbeat lo-fi hip-hop track with a relaxed piano melody"*, *"driving 80s cyberpunk synthwave with energetic bass"*).

### Model Architecture: "MusicGPT"
MusicGen is a decoder-only, causal, autoregressive **Transformer-based Audio LLM**:
1. **Discrete Audio Tokenization**: Continuous audio is compressed into discrete token sequences by a neural audio codec (**EnCodec**, 32 kHz, 4 codebooks at 50 Hz).
2. **Text Conditioning**: A frozen **T5 text encoder** embeds the text prompt into continuous semantic representations.
3. **Autoregressive Audio LM**: The Transformer decoder cross-attends to the text embeddings and predicts subsequent audio tokens ("autocomplete for sound").

```
                                  +---------------------------+
                                  |     User Text Prompt      |
                                  +-------------+-------------+
                                                |
                                                v
                                  +---------------------------+
                                  |      T5 Text Encoder      |
                                  +-------------+-------------+
                                                |
                                                v (Cross-Attention)
+------------------------+        +---------------------------+
| Acoustic Token History | -----> | Autoregressive Audio LM   |
| (EnCodec 4 Codebooks)  |        | (300M Causal Transformer) |
+------------------------+        +-------------+-------------+
                                                |
                                                v
                                  +---------------------------+
                                  | EnCodec Neural Decoder    |
                                  | (Audio Tokens -> 32 kHz)  |
                                  +-------------+-------------+
                                                |
                                                v
                                  +---------------------------+
                                  |    Master WAV Audio File  |
                                  +---------------------------+
```

---

## 🚀 Quickstart Guide

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run the Sonora Music Studio (Recommended Web UI)
```bash
python run.py
```
Open **[http://127.0.0.1:5001](http://127.0.0.1:5001)** in your browser.

### 3. Run with Gradio UI (Alternative)
```bash
python run.py --ui gradio
```
Open **[http://127.0.0.1:7861](http://127.0.0.1:7861)** in your browser.

---

## 🎛️ Studio Capabilities & Features

1. **Real-Time HTML5 Canvas Audio Visualizer**:
   - Web Audio API integration with `AnalyserNode`
   - Real-time frequency spectrum bars and dancing oscilloscope waves synchronized with track playback
   - Vinyl disc rotation animation during active audio streams
2. **Custom Seekable Audio Player**:
   - Scrub seek bar with smooth handle dragging
   - Elapsed and total duration timer
   - Master volume slider with mute toggle
   - High-fidelity 32 kHz CD-quality WAV download
3. **Multi-Engine Synthesis**:
   - **Local MusicGen Transformer**: Runs `facebook/musicgen-small` locally on PyTorch with automatic token length calculation.
   - **Polyphonic Algorithmic Synthesizer**: Generates rich multi-track compositions (Chords, Bassline, Melody Lead, Kick, Snare, Hi-Hats, Vinyl Crackle) in 0.2s without downloading weights.
   - **Cloud AI Music Engine**: Remote audio synthesis fallback.
4. **Curated Genre Presets**:
   - Lo-Fi Hip-Hop (85 BPM, C Minor)
   - Cyberpunk Synthwave (125 BPM, A Minor)
   - Cinematic Orchestral (110 BPM, D Minor)
   - Smooth Jazz (95 BPM, F Major)
   - 8-Bit Chiptune (140 BPM, C Major)
   - Ambient Meditation (60 BPM, G Major)
   - Acoustic Folk (105 BPM, E Major)
   - EDM / Club Dance (128 BPM, G Minor)
   - Melancholic Piano (72 BPM, A Minor)
5. **Persistent Track Library**:
   - Slide-out drawer with history of previous compositions
   - Inline preview playback and single-click master export

---

## 🧪 Testing

Run the automated test suite:
```bash
python -m unittest tests/test_music_engine.py
```

---

## 📁 Directory Structure

```
EXP 8/
├── MusicGen_Finetuning_Novel_Compositions.ipynb  # LoRA fine-tuning notebook
├── Simple_MusicGen_Generation.ipynb             # Foundation model inference notebook
├── README.md                                    # Complete project documentation
├── requirements.txt                             # Python dependencies
├── run.py                                       # Unified CLI launcher
├── app.py                                       # Flask Web Server & REST API
├── gradio_app.py                                # Gradio UI application
├── music_engine.py                              # MusicGen & Algorithmic audio synthesis engine
├── templates/
│   └── index.html                               # Modern HTML5 studio template
├── static/
│   ├── css/
│   │   └── style.css                            # Custom audio-reactive visual styling
│   ├── js/
│   │   └── app.js                               # Web Audio API visualizer & controller
│   └── outputs/                                 # Generated WAV tracks & history
└── tests/
    └── test_music_engine.py                     # Unit & endpoint tests
```
