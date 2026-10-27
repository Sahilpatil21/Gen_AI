# Experiment 10: Generation of Videos from Text Prompts using a Pretrained Open-Source Video Generation Foundation Model

**Course:** Generative AI Laboratory (GenAI Lab Manual by Dr. Sachin Balawant Takmare)  
**Location:** `EXP 10/`

---

## 🎯 Aim
To use a pretrained, open-source video generation foundation model to generate short videos directly from text prompts, without collecting a dataset or training the model, and to preview and save the generated videos.

---

## 📌 Objectives
1. Understand the concept of video generation using foundation models.
2. Explore text-to-video generation using a pretrained open-source model.
3. Generate short videos from natural-language prompts.
4. Experiment with different prompts, scenes, camera motions, and visual styles.
5. Preview and save the generated videos in suitable video formats (MP4, GIF).

---

## ⚡ Architecture & Features
- **Tri-Engine Video Generation:**
  1. **Smart Auto Engine:** Detects GPU memory limits and chooses the optimal pipeline.
  2. **Kinematic Temporal Video Engine:** Procedural and optical flow synthesis with 3D camera pan, wave dynamics, and real-time MP4 encoding.
  3. **Diffusers / 3D U-Net Engine:** Local PyTorch foundation pipeline for GPU workstations.
  4. **Cloud AI Video Synthesis:** Remote photorealistic high-definition video generation.
- **Directorial Controls:**
  - **Styles:** Cinematic Photorealism, Cyberpunk Sci-Fi, Anime & Studio Ghibli, 3D Pixar Animation, Vintage Film 1920s, Nature Documentary.
  - **Camera Motion:** Cinematic Wide Pan, Dolly Zoom In, FPV Drone Soar, Orbital Rotation, Static Tripod, Slow Tracking Shot.
  - **Lighting:** Golden Hour Sunset, Volumetric Neon Fog, Studio Rim Light, Overcast Moody, Bioluminescent Glow.
  - **Resolutions:** 16:9 Cinema (`512x288`), 3:2 Standard (`384x256`), 1:1 Square (`256x256`), 9:16 Vertical Reel (`288x512`).
- **Interactive Scrubber:**
  - Full-featured HTML5 video player with loop and autoplay.
  - Filmstrip frame inspector: inspect every individual frame and click to jump video playback to that exact time.
- **Clip duration:** New clips default to 48 frames at 12 FPS (4 seconds). Adjust the frame and FPS controls to change the duration (`frames ÷ FPS`).
- **Export Containers:**
  - Master H.264 MP4 video container.
  - Animated GIF lightweight preview.
  - Individual high-resolution PNG frames.
- **Dual User Interfaces:**
  - Modern Obsidian glassmorphic Web Studio (`app.py`, default port 5002).
  - Gradio blocks interface (`gradio_app.py`, default port 7862).

---

## 🚀 Quickstart

### 1. Launch Modern Web Studio (Flask)
```bash
cd "EXP 10"
python run.py
# Studio accessible at: http://127.0.0.1:5002
```

### 2. Launch Gradio Interface
```bash
python run.py --ui gradio
# Gradio interface accessible at: http://127.0.0.1:7862
```

### 3. Run Automated Unit Tests
```bash
python -m unittest tests/test_video_engine.py
```

### 4. Open Jupyter Notebook
```bash
jupyter notebook Exp_10_Text_To_Video.ipynb
```

---

## 📂 Project Structure
```
EXP 10/
├── Exp_10_Text_To_Video.ipynb  # Interactive Jupyter Notebook for submission
├── README.md                   # Complete architectural and usage guide
├── app.py                      # Flask REST API backend
├── gradio_app.py               # Gradio interactive blocks UI
├── requirements.txt            # Python dependencies
├── run.py                      # Unified CLI launcher (web & gradio)
├── video_engine.py             # Spatio-temporal foundation generation pipeline
├── static/
│   ├── css/
│   │   └── style.css           # Modern cinema glassmorphic styling
│   ├── js/
│   │   └── app.js              # Video player, scrubber & API client
│   └── outputs/                # Generated MP4 videos, GIFs, and frames
├── templates/
│   └── index.html              # Modern cinema studio HTML5 template
└── tests/
    └── test_video_engine.py    # Unit test suite (7/7 tests passed)
```

---

## 🎓 Conclusion
The experiment demonstrates how a pretrained open-source video generation foundation model can generate short videos from natural-language prompts. Students learn to load a pretrained model, configure generation parameters, generate and preview video content, and save the output without collecting a dataset or training a model from scratch.
