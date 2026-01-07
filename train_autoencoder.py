import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model

DATA_DIR = Path("data/dataset")
OUT_DIR = Path("outputs/models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def make_autoencoder(input_shape):
    inp = layers.Input(shape=input_shape)

    # Encoder (simple)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
    x = layers.MaxPooling2D(2, padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2, padding="same")(x)

    # Embedding (latent)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    encoded = layers.MaxPooling2D(2, padding="same", name="embedding")(x)

    # Decoder
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(encoded)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(2)(x)

    out = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

    autoencoder = Model(inp, out, name="autoencoder")
    encoder = Model(inp, encoded, name="encoder")
    return autoencoder, encoder

def main():
    X = np.load(DATA_DIR / "X.npy")
    print("X:", X.shape, X.dtype, "min/max:", X.min(), X.max())

    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

    autoencoder, encoder = make_autoencoder(input_shape=X.shape[1:])
    autoencoder.compile(optimizer="adam", loss="mse")

    autoencoder.summary()

    history = autoencoder.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=15,
        batch_size=8
    )

    autoencoder.save(OUT_DIR / "autoencoder.keras")
    encoder.save(OUT_DIR / "encoder.keras")
    print("✅ Modèles sauvegardés dans", OUT_DIR)

if __name__ == "__main__":
    main()
