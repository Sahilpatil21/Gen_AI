"""
Train a Conditional Variational Autoencoder (CVAE) on MNIST.

The model uses:
    latent_dim = 2
    num_classes = 10
    image size = 28 x 28

The decoder receives:
    z:     (batch, 2)
    label: (batch, 10)

Outputs:
    (batch, 28, 28, 1)

Saved files:
    model/cvae_encoder.keras
    model/cvae_decoder.keras
    model/cvae_full.keras
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


LATENT_DIM = 2
NUM_CLASSES = 10
IMAGE_SIZE = 28
BATCH_SIZE = 128
EPOCHS = 30

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "model")
os.makedirs(MODEL_DIR, exist_ok=True)


def build_encoder():
    image_input = keras.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
        name="image"
    )

    label_input = keras.Input(
        shape=(NUM_CLASSES,),
        name="label"
    )

    x = layers.Flatten()(image_input)
    x = layers.Concatenate()([x, label_input])
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)

    z_mean = layers.Dense(
        LATENT_DIM,
        name="z_mean"
    )(x)

    z_log_var = layers.Dense(
        LATENT_DIM,
        name="z_log_var"
    )(x)

    encoder = keras.Model(
        [image_input, label_input],
        [z_mean, z_log_var],
        name="cvae_encoder"
    )

    return encoder


def build_decoder():
    z_input = keras.Input(
        shape=(LATENT_DIM,),
        name="z"
    )

    label_input = keras.Input(
        shape=(NUM_CLASSES,),
        name="label"
    )

    x = layers.Concatenate()([z_input, label_input])
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(
        IMAGE_SIZE * IMAGE_SIZE,
        activation="sigmoid"
    )(x)

    output = layers.Reshape(
        (IMAGE_SIZE, IMAGE_SIZE, 1)
    )(x)

    decoder = keras.Model(
        [z_input, label_input],
        output,
        name="cvae_decoder"
    )

    return decoder


class CVAE(keras.Model):
    def __init__(self, encoder, decoder, beta=1.0, **kwargs):
        super().__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta

        self.total_loss_tracker = keras.metrics.Mean(
            name="total_loss"
        )
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(
            name="kl_loss"
        )

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        images, labels = data

        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(
                [images, labels],
                training=True
            )

            epsilon = tf.random.normal(
                shape=tf.shape(z_mean)
            )

            z = (
                z_mean
                + tf.exp(0.5 * z_log_var)
                * epsilon
            )

            reconstruction = self.decoder(
                [z, labels],
                training=True
            )

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(
                        images,
                        reconstruction
                    ),
                    axis=(1, 2)
                )
            )

            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1
                    + z_log_var
                    - tf.square(z_mean)
                    - tf.exp(z_log_var),
                    axis=1
                )
            )

            total_loss = (
                reconstruction_loss
                + self.beta * kl_loss
            )

        grads = tape.gradient(
            total_loss,
            self.trainable_weights
        )

        self.optimizer.apply_gradients(
            zip(grads, self.trainable_weights)
        )

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(
            reconstruction_loss
        )
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss":
                self.reconstruction_loss_tracker.result(),
            "kl_loss":
                self.kl_loss_tracker.result(),
        }


def main():
    print("Loading MNIST...")

    (x_train, y_train), (x_test, y_test) = (
        keras.datasets.mnist.load_data()
    )

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train_onehot = keras.utils.to_categorical(
        y_train,
        NUM_CLASSES
    ).astype("float32")

    y_test_onehot = keras.utils.to_categorical(
        y_test,
        NUM_CLASSES
    ).astype("float32")

    train_ds = (
        tf.data.Dataset
        .from_tensor_slices(
            (x_train, y_train_onehot)
        )
        .shuffle(10000)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    encoder = build_encoder()
    decoder = build_decoder()

    print("\nEncoder:")
    encoder.summary()

    print("\nDecoder:")
    decoder.summary()

    cvae = CVAE(
        encoder,
        decoder,
        beta=1.0
    )

    cvae.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-3
        )
    )

    print("\nTraining CVAE...")

    cvae.fit(
        train_ds,
        epochs=EPOCHS
    )

    print("\nSaving models...")

    encoder.save(
        os.path.join(
            MODEL_DIR,
            "cvae_encoder.keras"
        )
    )

    decoder.save(
        os.path.join(
            MODEL_DIR,
            "cvae_decoder.keras"
        )
    )

    # Save the trained wrapper weights as well. The Flask app does not
    # need this file, but it is useful for keeping the full model.
    try:
        cvae.save_weights(
            os.path.join(
                MODEL_DIR,
                "cvae_full.weights.h5"
            )
        )
    except Exception as exc:
        print(f"Note: Full wrapper weights skipped: {exc}")

    print("\nTraining complete.")
    print(
        "Decoder saved to:",
        os.path.join(
            MODEL_DIR,
            "cvae_decoder.keras"
        )
    )


if __name__ == "__main__":
    main()
