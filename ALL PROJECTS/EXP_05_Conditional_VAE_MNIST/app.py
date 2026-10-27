from flask import Flask, render_template, request, jsonify
import base64
import io
import os

import numpy as np
import tensorflow as tf
from PIL import Image


app = Flask(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "cvae_decoder.keras"
)

decoder = None


def load_decoder():
    global decoder

    if decoder is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}. "
                "Run training/train_cvae.py first."
            )

        decoder = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Conditional VAE decoder loaded.")
        print("Decoder input shapes:", decoder.input_shape)
        print("Decoder output shape:", decoder.output_shape)

    return decoder


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    try:
        model = load_decoder()
        return jsonify({
            "status": "ok",
            "model_loaded": True,
            "decoder_input_shape": [str(x) for x in model.input_shape],
            "decoder_output_shape": [str(x) for x in model.output_shape],
        })
    except Exception as exc:
        return jsonify({
            "status": "error",
            "model_loaded": False,
            "error": str(exc)
        }), 500


@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json(silent=True) or {}
        selected_digit = int(data.get("digit", 0))

        if selected_digit < 0 or selected_digit > 9:
            return jsonify({
                "success": False,
                "error": "Please select a number between 0 and 9."
            }), 400

        model = load_decoder()

        # Random point in the 2-D latent space.
        z = tf.random.normal(
            shape=(1, 2),
            mean=0.0,
            stddev=1.0
        )

        # One-hot condition for the selected MNIST digit.
        label = tf.one_hot(
            [selected_digit],
            depth=10,
            dtype=tf.float32
        )

        # The conditional decoder receives [z, label].
        generated_image = model.predict(
            [z, label],
            verbose=0
        )[0]

        generated_image = np.squeeze(generated_image)

        if generated_image.ndim != 2:
            raise ValueError(
                f"Unexpected decoder output shape: {generated_image.shape}"
            )

        generated_image = np.clip(generated_image, 0.0, 1.0)
        generated_image = (generated_image * 255).astype(np.uint8)

        image = Image.fromarray(generated_image, mode="L")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        image_base64 = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        return jsonify({
            "success": True,
            "selected_digit": selected_digit,
            "image": image_base64
        })

    except Exception as exc:
        print("ERROR:", repr(exc))
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5005))
    print(f"\n* Launching Conditional VAE Studio on http://127.0.0.1:{port} ...\n")
    app.run(
        debug=False,
        host="127.0.0.1",
        port=port
    )
