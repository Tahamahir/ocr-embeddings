from pathlib import Path
import cv2
import numpy as np

IN_DIR = Path("data/processed/gray")
OUT_DIR = Path("data/dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Taille finale pour le modèle (simple et léger)
H, W = 256, 256

def list_images(folder: Path):
    return sorted([p for p in folder.rglob("*.png")])

def load_and_resize(path: Path, h=H, w=W):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Impossible de lire {path}")
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0  # normalisation 0..1
    return img

def main():
    paths = list_images(IN_DIR)
    print("Images trouvées:", len(paths))
    assert len(paths) > 0, "Aucune image dans data/processed/gray"

    X = np.zeros((len(paths), H, W), dtype=np.float32)

    for i, p in enumerate(paths):
        X[i] = load_and_resize(p)

    X = X[..., np.newaxis]  # (N, H, W, 1)

    np.save(OUT_DIR / "X.npy", X)
    (OUT_DIR / "paths.txt").write_text("\n".join(str(p) for p in paths), encoding="utf-8")

    print("✅ Dataset sauvegardé:")
    print(" -", OUT_DIR / "X.npy")
    print("Shape X:", X.shape)

if __name__ == "__main__":
    main()
