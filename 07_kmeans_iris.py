"""
Practical 7: K-Means clustering on Iris dataset using the Elbow Method.
Run: python 07_kmeans_iris.py
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def main():
    iris = load_iris()
    X = StandardScaler().fit_transform(iris.data)

    inertias = []
    k_values = range(1, 11)
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        inertias.append(model.inertia_)

    plt.figure(figsize=(7, 5))
    plt.plot(list(k_values), inertias, marker="o")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia (WCSS)")
    plt.title("Elbow Method for Optimal K")
    plt.xticks(list(k_values))
    plt.tight_layout()
    plt.show()

    # Iris has three known species; use K=3 for the final illustrative clustering.
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    X_2d = PCA(n_components=2).fit_transform(X)

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="viridis", s=40)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("K-Means Clustering of Iris Dataset (K=3)")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.show()

    print("Cluster sizes:")
    for cluster in range(3):
        print(f"Cluster {cluster}: {(labels == cluster).sum()} samples")

if __name__ == "__main__":
    main()
