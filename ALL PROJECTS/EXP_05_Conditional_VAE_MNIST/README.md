# Conditional VAE MNIST Digit Generator

This project converts the original unconditional MNIST VAE into a Conditional VAE (CVAE).

The web application lets the user select a digit from 0 to 9. The selected digit is converted to a 10-class one-hot vector and passed to the conditional decoder together with a random 2-D latent vector.

## Project structure

```text
cvae_mnist_flask/
├── app.py
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── model/
│   └── cvae_decoder.keras       # generated after training
└── training/
    └── train_cvae.py
```

## 1. Install dependencies

Create a virtual environment if desired:

```bash
python -m venv .venv
```

Activate it.

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
```

## 2. Train the CVAE

Run:

```bash
python training/train_cvae.py
```

The script downloads MNIST through Keras, trains the conditional VAE, and saves:

```text
model/cvae_encoder.keras
model/cvae_decoder.keras
model/cvae_full.keras
```

The Flask application only needs `model/cvae_decoder.keras`.

Training may take a while depending on your CPU/GPU.

## 3. Start Flask

After training:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## How conditioning works

The decoder receives two inputs:

```text
z      = shape (batch, 2)
label  = shape (batch, 10)
```

The selected digit is represented as a one-hot vector:

```text
digit 0 -> [1,0,0,0,0,0,0,0,0,0]
digit 7 -> [0,0,0,0,0,0,0,1,0,0]
digit 9 -> [0,0,0,0,0,0,0,0,0,1]
```

The decoder combines the latent representation and digit condition and generates a 28x28 grayscale image.

## Important

The original `vae_decoder.keras` from the unconditional VAE cannot be used for digit conditioning because it was not trained with labels.

You must train the new CVAE and use:

```text
model/cvae_decoder.keras
```

instead.
