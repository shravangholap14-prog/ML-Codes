"""
Practical 2: Principal Component Analysis (PCA) using the Wine dataset.
Run: python 02_pca_wine.py
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def main():
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = wine.target

    print("Original dataset shape:", X.shape)
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=["Principal Component 1", "Principal Component 2"])
    pca_df["Wine Class"] = y

    print("\nReduced dataset (first 5 rows):\n", pca_df.head())
    print("\nExplained variance ratio:", pca.explained_variance_ratio_)
    print("Total variance retained:", round(pca.explained_variance_ratio_.sum() * 100, 2), "%")

    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", edgecolor="k")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA of Wine Dataset")
    plt.colorbar(scatter, label="Wine class")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
