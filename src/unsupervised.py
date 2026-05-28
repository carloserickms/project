from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from src.utils import RANDOM_STATE


def run_kmeans_analysis(splits: dict, figures_dir: Path, metrics_dir: Path) -> dict:
    """
    Realiza uma análise de clustering usando o algoritmo K-Means, testando diferentes valores de K e avaliando o desempenho com métricas como inércia, silhouette score, adjusted rand index e normalized mutual information. O processo inclui a geração de gráficos para o método do cotovelo e o silhouette score, a visualização dos clusters usando PCA e a criação de uma tabela cruzada entre os clusters formados e as classes reais. Os resultados são salvos em arquivos e retornados em um dicionário.
    Args:
        splits (dict): Um dicionário contendo os dados de entrada e os rótulos verdadeiros para a análise de clustering.
        figures_dir (Path): O diretório onde as figuras são salvas.
        metrics_dir (Path): O diretório onde os arquivos de saída são salvos.

    Returns:
        dict: Um dicionário contendo os resultados da análise de clustering.
    """
    X_scaled = splits["scaler"].fit_transform(splits["X"])
    y = splits["y"]
    k_values = list(range(2, min(11, len(set(y)) + 8)))
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = model.fit_predict(X_scaled)
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(X_scaled, labels),
                "adjusted_rand_index": adjusted_rand_score(y, labels),
                "normalized_mutual_info": normalized_mutual_info_score(y, labels),
            }
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(metrics_dir / "kmeans_metrics.csv", index=False)
    best_row = metrics.sort_values("silhouette", ascending=False).iloc[0]
    best_k = int(best_row["k"])
    best_model = KMeans(n_clusters=best_k, n_init=20, random_state=RANDOM_STATE)
    best_labels = best_model.fit_predict(X_scaled)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.lineplot(data=metrics, x="k", y="inertia", marker="o", ax=ax)
    ax.set_title("Método do cotovelo - K-Means")
    ax.set_xlabel("Número de clusters (K)")
    ax.set_ylabel("Inércia")
    fig.tight_layout()
    fig.savefig(figures_dir / "kmeans_elbow.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.lineplot(data=metrics, x="k", y="silhouette", marker="o", ax=ax)
    ax.set_title("Silhouette score por K")
    ax.set_xlabel("Número de clusters (K)")
    ax.set_ylabel("Silhouette")
    fig.tight_layout()
    fig.savefig(figures_dir / "kmeans_silhouette.png", dpi=150)
    plt.close(fig)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    points = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({"PC1": points[:, 0], "PC2": points[:, 1], "cluster": best_labels, "classe_real": splits["label_encoder"].inverse_transform(y)})
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="cluster", style="classe_real", palette="tab10", alpha=0.75, ax=ax)
    ax.set_title(f"PCA 2D dos clusters K-Means (K={best_k})")
    fig.tight_layout()
    fig.savefig(figures_dir / "kmeans_pca_clusters.png", dpi=150)
    plt.close(fig)

    crosstab = pd.crosstab(pca_df["cluster"], pca_df["classe_real"])
    crosstab.to_csv(metrics_dir / "kmeans_cluster_vs_class.csv")

    return {
        "tested_k": k_values,
        "best_k_silhouette": best_k,
        "best_silhouette": float(best_row["silhouette"]),
        "best_inertia": float(best_row["inertia"]),
        "adjusted_rand_index": float(best_row["adjusted_rand_index"]),
        "normalized_mutual_info": float(best_row["normalized_mutual_info"]),
        "pca_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cluster_class_table": crosstab.to_dict(),
    }
