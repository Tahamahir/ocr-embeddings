from pathlib import Path
import cv2
import numpy as np

# ======================
# CONFIG
# ======================
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

# Deux sorties: grayscale propre (pour DL) + binaire (pour OCR classique)
OUT_GRAY_DIR = OUT_DIR / "gray"
OUT_BIN_DIR  = OUT_DIR / "binary"

PREVIEW_DIR = Path("outputs/previews")
SAVE_PREVIEW_COUNT = 12

TARGET_H = 800          # pour documents, garde plus de détails (tu peux baisser à 512 si lourd)
MAX_W = 1200            # largeur max après resize+padding/crop
USE_BINARY_OUTPUT = True

# ======================
# HELPERS
# ======================
def list_all_files(folder: Path):
    return sorted([p for p in folder.rglob("*") if p.is_file()])

def is_image_file(p: Path):
    # Ajoute des extensions fréquentes (Windows): jfif, webp
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".jfif", ".webp"}
    return p.suffix.lower() in exts

def read_gray_opencv(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return img

def resize_keep_ratio(gray: np.ndarray, target_h: int) -> np.ndarray:
    h, w = gray.shape[:2]
    if h == 0:
        return gray
    scale = target_h / float(h)
    new_w = int(round(w * scale))
    return cv2.resize(gray, (new_w, target_h), interpolation=cv2.INTER_AREA)

def pad_or_crop_width_uint8(img: np.ndarray, target_w: int, pad_value: int = 255) -> np.ndarray:
    h, w = img.shape[:2]
    if w > target_w:
        return img[:, :target_w]
    if w < target_w:
        pad = target_w - w
        return np.pad(img, ((0, 0), (0, pad)), mode="constant", constant_values=pad_value)
    return img

def illumination_correction(gray: np.ndarray) -> np.ndarray:
    """
    Correction simple du fond/ombre:
    - Estime un "fond" par un flou large
    - Enlève ce fond -> remet à l'échelle
    """
    # flou large (taille impaire)
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=25, sigmaY=25)
    # éviter division par zéro
    bg = np.clip(bg, 1, 255)
    norm = (gray.astype(np.float32) / bg.astype(np.float32)) * 128.0
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    return norm

def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    # CLAHE (très bien pour documents scannés)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def denoise(gray: np.ndarray) -> np.ndarray:
    # filtre léger, conserve les bords
    return cv2.bilateralFilter(gray, d=7, sigmaColor=30, sigmaSpace=30)

def unsharp_mask(gray: np.ndarray) -> np.ndarray:
    # rend le texte plus net (léger)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def make_binary(gray: np.ndarray) -> np.ndarray:
    # Binarisation adaptative (souvent meilleure que Otsu sur documents)
    # Si tu vois trop de bruit, augmente blockSize (ex: 51) ou C (ex: 15)
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41, 11
    )

def save_preview(before_u8: np.ndarray, after_gray_u8: np.ndarray, after_bin_u8: np.ndarray | None, idx: int):
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    # Met tout à même hauteur/largeur pour concat
    b = pad_or_crop_width_uint8(resize_keep_ratio(before_u8, TARGET_H), MAX_W, 255)
    g = pad_or_crop_width_uint8(after_gray_u8, MAX_W, 255)

    if after_bin_u8 is not None:
        bn = pad_or_crop_width_uint8(after_bin_u8, MAX_W, 255)
        combo = np.concatenate([b, g, bn], axis=1)
        out = PREVIEW_DIR / f"preview_{idx:03d}_before_gray_binary.png"
    else:
        combo = np.concatenate([b, g], axis=1)
        out = PREVIEW_DIR / f"preview_{idx:03d}_before_gray.png"

    cv2.imwrite(str(out), combo)

def preprocess_one(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Retourne:
    - gray_clean_u8 : uint8 0..255 (pour DL, on normalisera plus tard)
    - bin_u8 (optionnel)
    """
    x = resize_keep_ratio(gray, TARGET_H)
    x = denoise(x)
    x = illumination_correction(x)
    x = enhance_contrast(x)
    x = unsharp_mask(x)

    # largeur fixe
    x = pad_or_crop_width_uint8(x, MAX_W, 255)

    bin_img = make_binary(x) if USE_BINARY_OUTPUT else None
    return x, bin_img

def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Dossier introuvable: {RAW_DIR} (lance depuis la racine du projet)")

    OUT_GRAY_DIR.mkdir(parents=True, exist_ok=True)
    if USE_BINARY_OUTPUT:
        OUT_BIN_DIR.mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    all_files = list_all_files(RAW_DIR)
    image_files = [p for p in all_files if is_image_file(p)]

    print(f"📦 Fichiers totaux dans data/raw: {len(all_files)}")
    print(f"🖼️ Fichiers images reconnus: {len(image_files)}")

    if len(image_files) == 0:
        print("⚠️ Aucune image détectée. Vérifie extensions (png/jpg/jpeg/jfif/webp/...) ou si ce sont des PDF.")
        return

    ok_count = 0
    skipped = []

    for i, p in enumerate(image_files):
        gray = read_gray_opencv(p)
        if gray is None:
            skipped.append(str(p))
            continue

        gray_clean, bin_img = preprocess_one(gray)

        rel = p.relative_to(RAW_DIR)
        out_gray = (OUT_GRAY_DIR / rel).with_suffix(".png")
        out_gray.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_gray), gray_clean)

        if USE_BINARY_OUTPUT and bin_img is not None:
            out_bin = (OUT_BIN_DIR / rel).with_suffix(".png")
            out_bin.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_bin), bin_img)

        # previews avant/après
        if ok_count < SAVE_PREVIEW_COUNT:
            save_preview(gray, gray_clean, bin_img if USE_BINARY_OUTPUT else None, ok_count)

        ok_count += 1

    print(f"✅ Images traitées: {ok_count}")
    if skipped:
        print(f"⚠️ Images ignorées (non lisibles par OpenCV): {len(skipped)}")
        print("Exemples:")
        for s in skipped[:5]:
            print("  -", s)

    print(f"📁 Sorties grayscale (pour DL): {OUT_GRAY_DIR}")
    if USE_BINARY_OUTPUT:
        print(f"📁 Sorties binaires (optionnel): {OUT_BIN_DIR}")
    print(f"👀 Previews: {PREVIEW_DIR}")

if __name__ == "__main__":
    main()
