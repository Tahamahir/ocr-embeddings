from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

EMB_DIR = Path("outputs/embeddings")
OUT_FIG = Path("outputs/figures")
OUT_FIG.mkdir(parents=True, exist_ok=True)

def main():
    Z = np.load(EMB_DIR / "Z.npy")  # (N, D)
    print("Z:", Z.shape)

    # PCA 2D
    pca = PCA(n_components=2, random_state=42)
    Z_pca = pca.fit_transform(Z)

    plt.figure(figsize=(7, 6))
    plt.scatter(Z_pca[:, 0], Z_pca[:, 1], s=25)
    plt.title("Embeddings - PCA (2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "pca_2d.png", dpi=200)
    plt.show()

    # t-SNE 2D (plus lent)
    tsne = TSNE(n_components=2, random_state=42, perplexity=15, init="pca")
    Z_tsne = tsne.fit_transform(Z)

    plt.figure(figsize=(7, 6))
    plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], s=25)
    plt.title("Embeddings - t-SNE (2D)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "tsne_2d.png", dpi=200)
    plt.show()

    print("✅ Figures:", OUT_FIG / "pca_2d.png", "et", OUT_FIG / "tsne_2d.png")

if __name__ == "__main__":
    main()
