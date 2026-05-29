from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from src.utils import RANDOM_STATE


def _cluster_metrics(X_scaled: np.ndarray, y: np.ndarray, k: int) -> dict:
    model = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
    labels = model.fit_predict(X_scaled)
    return {
        "k": k,
        "inertia": float(model.inertia_),
        "silhouette": float(silhouette_score(X_scaled, labels)),
        "adjusted_rand_index": float(adjusted_rand_score(y, labels)),
        "normalized_mutual_info": float(normalized_mutual_info_score(y, labels)),
        "labels": labels,
        "model": model,
    }


def run_kmeans_analysis(splits: dict, figures_dir: Path, metrics_dir: Path) -> dict:
    """
    K-Means no conjunto de treino apenas (rótulos ignorados no ajuste).
    Escala com o StandardScaler ajustado no treino em prepare_splits (sem refit).
    """
    X_train_scaled = splits["scaler"].transform(splits["X_train"])
    y_train = splits["y_train"]
    class_names = splits["label_encoder"].classes_
    n_classes = len(class_names)

    k_values = list(range(2, min(11, n_classes + 8)))
    rows = []
    for k in k_values:
        result = _cluster_metrics(X_train_scaled, y_train, k)
        rows.append({key: result[key] for key in ("k", "inertia", "silhouette", "adjusted_rand_index", "normalized_mutual_info")})

    metrics = pd.DataFrame(rows)
    metrics.to_csv(metrics_dir / "kmeans_metrics.csv", index=False)

    best_row = metrics.sort_values("silhouette", ascending=False).iloc[0]
    best_k = int(best_row["k"])
    k_n_classes = max(2, n_classes)
    k_classes_row = metrics.loc[metrics["k"] == k_n_classes].iloc[0] if k_n_classes in k_values else metrics.iloc[0]

    best_result = _cluster_metrics(X_train_scaled, y_train, best_k)
    k_classes_result = _cluster_metrics(X_train_scaled, y_train, k_n_classes)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.lineplot(data=metrics, x="k", y="inertia", marker="o", ax=ax)
    ax.set_title("Método do cotovelo - K-Means (conjunto de treino)")
    ax.set_xlabel("Número de clusters (K)")
    ax.set_ylabel("Inércia")
    fig.tight_layout()
    fig.savefig(figures_dir / "kmeans_elbow.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.lineplot(data=metrics, x="k", y="silhouette", marker="o", ax=ax)
    ax.set_title("Silhouette score por K (conjunto de treino)")
    ax.set_xlabel("Número de clusters (K)")
    ax.set_ylabel("Silhouette")
    fig.tight_layout()
    fig.savefig(figures_dir / "kmeans_silhouette.png", dpi=150)
    plt.close(fig)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    points = pca.fit_transform(X_train_scaled)
    y_train_labels = splits["label_encoder"].inverse_transform(y_train)

    pca_df = pd.DataFrame(
        {
            "PC1": points[:, 0],
            "PC2": points[:, 1],
            "cluster": best_result["labels"],
            "classe_real": y_train_labels,
        }
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="cluster", style="classe_real", palette="tab10", alpha=0.75, ax=ax)
    ax.set_title(f"PCA 2D dos clusters K-Means (K={best_k}, treino)")
    fig.tight_layout()
    fig.savefig(figures_dir / "kmeans_pca_clusters.png", dpi=150)
    plt.close(fig)

    crosstab = pd.crosstab(pca_df["cluster"], pca_df["classe_real"])
    crosstab.to_csv(metrics_dir / "kmeans_cluster_vs_class.csv")

    pca_k2 = pd.DataFrame(
        {
            "PC1": points[:, 0],
            "PC2": points[:, 1],
            "cluster": k_classes_result["labels"],
            "classe_real": y_train_labels,
        }
    )
    crosstab_k2 = pd.crosstab(pca_k2["cluster"], pca_k2["classe_real"])
    crosstab_k2.to_csv(metrics_dir / f"kmeans_cluster_vs_class_k{k_n_classes}.csv")

    return {
        "data_scope": "treino",
        "tested_k": k_values,
        "n_classes": n_classes,
        "best_k_silhouette": best_k,
        "best_silhouette": float(best_row["silhouette"]),
        "best_inertia": float(best_row["inertia"]),
        "best_k_ari": float(best_row["adjusted_rand_index"]),
        "best_k_nmi": float(best_row["normalized_mutual_info"]),
        "k_equals_n_classes": int(k_n_classes),
        "k_n_classes_silhouette": float(k_classes_row["silhouette"]),
        "k_n_classes_ari": float(k_classes_row["adjusted_rand_index"]),
        "k_n_classes_nmi": float(k_classes_row["normalized_mutual_info"]),
        "adjusted_rand_index": float(best_row["adjusted_rand_index"]),
        "normalized_mutual_info": float(best_row["normalized_mutual_info"]),
        "pca_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cluster_class_table": crosstab.to_dict(),
        "cluster_class_table_k_n_classes": crosstab_k2.to_dict(),
    }
