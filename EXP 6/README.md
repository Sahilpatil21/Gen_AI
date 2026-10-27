# Experiment 6: Face Generation using GANs (DCGAN) on CelebA Dataset

**Course:** Generative AI Laboratory  
**Faculty / Author:** Dr. Sachin Balawant Takmare  
**Location:** `EXP 6/`

---

## 🎯 Aim
To build, train, and deploy a Deep Convolutional Generative Adversarial Network (DCGAN) capable of synthesizing realistic human celebrity faces from random Gaussian latent noise using the CelebA dataset, and to analyze latent space geometry through continuous identity morphing.

---

## 📌 Objectives
1. Understand the fundamental minimax adversarial formulation of Generative Adversarial Networks:
   $$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$
2. Implement transposed convolutional generator architectures with Batch Normalization.
3. Explore the latent noise space $z \sim \mathcal{N}(0, I_{100})$ and the effect of the truncation trick ($\psi$).
4. Compute continuous geodesic paths between different latent identities using Spherical Linear Interpolation (SLERP):
   $$z(t) = \frac{\sin((1-t)\theta)}{\sin\theta} z_A + \frac{\sin(t\theta)}{\sin\theta} z_B$$
5. Serve the trained Generator through an interactive modern Cinema Web Studio and Gradio application.

---

## ⚡ Key Features
- **Pretrained Generator (`generator.pth`):**
  - Trained on ~200,000 celebrity faces from the CelebFaces Attributes (CelebA) dataset.
  - Generates 64x64 native RGB faces, upscalable to 128x128 and 256x256 via Lanczos filtering.
- **Latent Space Morphing Lab:**
  - Real-time continuous interpolation between any two identities (Seed A and Seed B).
  - Exports smooth animated loops in both GIF and WebP formats.
- **Zero Heavy External Dependencies:**
  - Pure PyTorch, NumPy, and Pillow implementation (eliminates `torchvision` install hurdles).
- **Dual User Interfaces:**
  - Modern Obsidian & Amber-Gold Web Studio (`app.py`, port 5006).
  - Gradio blocks application (`gradio_app.py`, port 7866).

---

## 🚀 Quickstart

### 1. Launch Modern Web Studio (Flask)
```bash
cd "EXP 6"
python run.py
# Studio accessible at: http://127.0.0.1:5006
```

### 2. Launch Gradio Interface
```bash
python run.py --ui gradio
# Gradio interface accessible at: http://127.0.0.1:7866
```

### 3. Generate Faces via CLI
```bash
python generate.py --count 4 --scale 2 --seed 42
```

### 4. Run Automated Unit Tests
```bash
python -m unittest tests/test_face_engine.py
```

### 5. Open Jupyter Notebook
```bash
jupyter notebook GAN_Face_Generation_CelebA.ipynb
```

---

## 📂 Project Structure
```
EXP 6/
├── GAN_Face_Generation_CelebA.ipynb  # Complete student lab notebook
├── README.md                          # Architectural and operational guide
├── app.py                             # Flask REST API backend
├── face_engine.py                     # DCGAN inference and latent morphing engine
├── generate.py                        # Standalone CLI generation utility
├── generator.pth                      # Pretrained DCGAN generator weights
├── gradio_app.py                      # Gradio blocks user interface
├── model.py                           # DCGAN Generator PyTorch architecture
├── requirements.txt                   # Dependency specifications
├── run.py                             # Unified CLI launcher
├── static/
│   ├── css/
│   │   └── style.css                  # Modern obsidian amber-gold theme
│   └── js/
│       └── app.js                     # Interactive client & morph controller
├── templates/
│   └── index.html                     # Studio and morphing lab interface
└── tests/
    └── test_face_engine.py            # Automated unit tests (5/5 passed)
```

---

## 🎓 Conclusion
The experiment demonstrates how Deep Convolutional Generative Adversarial Networks (DCGAN) learn to synthesize high-dimensional human face distributions from low-dimensional random latent vectors. By exploring latent vector arithmetic and geodesic interpolation, students gain a deep understanding of continuous representation learning in Generative AI.
