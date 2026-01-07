from pathlib import Path
import numpy as np
import tensorflow as tf

DATA_DIR = Path("data/dataset")
MODEL_DIR = Path("outputs/models")
OUT_DIR = Path("outputs/embeddings")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # Charger dataset
    X = np.load(DATA_DIR / "X.npy")  # (N, 256, 256, 1)
    print("X:", X.shape, X.dtype)

    # Charger encoder
    encoder = tf.keras.models.load_model(MODEL_DIR / "encoder.keras")

    # Produire embeddings (N, 32, 32, 128) dans ton cas
    Z = encoder.predict(X, batch_size=8, verbose=1)
    print("Z (raw):", Z.shape, Z.dtype)

    # Aplatir en vecteurs (N, D)
    Z_flat = Z.reshape((Z.shape[0], -1))
    print("Z_flat:", Z_flat.shape)

    np.save(OUT_DIR / "Z.npy", Z_flat)
    print("✅ Embeddings sauvegardés:", OUT_DIR / "Z.npy")

if __name__ == "__main__":
    main()
