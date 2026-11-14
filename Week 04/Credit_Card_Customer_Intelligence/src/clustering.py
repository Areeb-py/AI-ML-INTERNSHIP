
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from config import OUTPUT, PLOTS, RANDOM_STATE

def run_clustering(X, ids, feature_names):
    rows = []
    for k in range(2, 9):
        m = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = m.fit_predict(X)
        rows.append({"k": k, "inertia": m.inertia_, "silhouette": silhouette_score(X, labels)})
    scores = pd.DataFrame(rows)
    scores.to_csv(OUTPUT / "k_selection_scores.csv", index=False)
    best_k = int(scores.loc[scores.silhouette.idxmax(), "k"])
    print("Best tested K-Means k:", best_k)

    models = {
        "KMeans": KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10),
        "Agglomerative": AgglomerativeClustering(n_clusters=best_k),
        "GaussianMixture": GaussianMixture(n_components=best_k, random_state=RANDOM_STATE),
        "DBSCAN": DBSCAN(eps=1.5, min_samples=10)
    }
    assignments = pd.DataFrame({"CUSTOMER_ID": ids})
    comparison = []
    for name, model in models.items():
        labels = model.fit_predict(X)
        assignments[name] = labels
        mask = labels != -1
        score = None
        if mask.sum() > 1 and len(set(labels[mask])) > 1 and len(set(labels[mask])) < mask.sum():
            score = silhouette_score(X[mask], labels[mask])
        comparison.append({"model": name, "clusters": len(set(labels)), "noise_points": int((labels == -1).sum()), "silhouette": score})
    assignments.to_csv(OUTPUT / "customer_segments.csv", index=False)
    pd.DataFrame(comparison).to_csv(OUTPUT / "model_comparison.csv", index=False)

    profile = pd.DataFrame(X, columns=feature_names)
    profile["cluster"] = assignments["KMeans"]
    profile.groupby("cluster").mean().to_csv(OUTPUT / "cluster_profiles_scaled.csv")

    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
    plt.figure(figsize=(9, 6))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=assignments["KMeans"], s=8, alpha=.6, cmap="tab10")
    plt.colorbar(sc, label="Cluster")
    plt.title("K-Means Segments - PCA Projection")
    plt.tight_layout()
    plt.savefig(PLOTS / "kmeans_pca.png", dpi=150)
    plt.close()
    plt.figure()
    plt.plot(scores.k, scores.inertia, marker="o")
    plt.xlabel("K"); plt.ylabel("Inertia"); plt.title("K-Means Elbow Curve")
    plt.tight_layout(); plt.savefig(PLOTS / "kmeans_elbow.png", dpi=150); plt.close()
    print("Clustering results saved.")
