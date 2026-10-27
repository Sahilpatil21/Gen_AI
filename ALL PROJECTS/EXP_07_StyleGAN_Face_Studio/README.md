# StyleGAN-style Face Generator (CelebA, 64×64) — Exp 7, Generative AI

A compact PyTorch implementation of a **style-based GAN** (StyleGAN-1 ideas) trained on
20k CelebA faces. This project was refactored from `notebook/Exp_no_7_gen_ai.ipynb`.

## Key ideas implemented
| Concept | Where |
|---|---|
| Mapping network Z → W (PixelNorm + 6 FC layers) | `models/mapping.py` |
| Learned constant 4×4 input (generator never sees z directly) | `models/generator.py` |
| AdaIN style injection at every block | `models/layers.py` |
| Per-pixel noise injection | `models/layers.py` |
| Non-saturating logistic loss + lazy R1 penalty | `losses.py` |
| Mapping net LR = 0.01 × synthesis LR | `train.py`, `config.py` |
| Truncation trick, style mixing, noise variation | `generate.py` |

## Project layout
```
stylegan_celeba/
├── config.py            # all hyper-parameters
├── data.py              # CelebA loading + transforms
├── losses.py            # D/G loss, R1 penalty
├── models/              # layers, mapping, generator, discriminator
├── train.py             # training loop (saves samples + checkpoint every epoch)
├── generate.py          # truncation / style-mixing / noise demos
├── app.py               # Streamlit gallery for saved results
├── utils.py
├── notebook/            # original Colab notebook
├── results/notebook_outputs/   # figures produced by the original run
├── samples/  checkpoints/      # created during training
└── requirements.txt
```

## Setup & run
```bash
pip install -r requirements.txt
streamlit run app.py                 # browse saved results in the browser
python train.py                      # ~15 epochs, GPU recommended (T4 was used)
python generate.py truncation
python generate.py mixing --crossover 2
python generate.py noise
```

## Setup used in the notebook
- Dataset: `tpremoli/CelebA-attrs-20k` (19,999 train images), 64×64, batch 64 (312 steps/epoch)
- 15 epochs, Adam (β=(0, 0.99)), LR 2e-3, R1 γ=10 every 16 steps
- Generator + mapping: 3,508,323 params · Discriminator: 695,137 params
- Final losses hovered near D ≈ 1.3, G ≈ 0.7–1.0 (balanced adversarial training)

## Results
See `results/notebook_outputs/` — per-epoch samples (`epoch_001.png … epoch_015.png`),
`loss_curves.png`, `truncation_trick.png`, `style_mixing_grid.png`, `noise_variation.png`.
