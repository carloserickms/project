from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def generate_eda_artifacts(df: pd.DataFrame, figures_dir: Path, metrics_dir: Path) -> dict:
    """Gera artefatos de exploração de dados (EDA) para um conjunto de dados.

    Args:
        df (pd.DataFrame): O DataFrame contendo os dados a serem explorados.
        figures_dir (Path): O diretório onde as figuras serão salvas.
        metrics_dir (Path): O diretório onde os arquivos de saída serão salvos.

    Returns:
        dict: Um dicionário contendo os resultados da exploração de dados, incluindo a variância explicada pela PCA e os caminhos para os arquivos de estatísticas descritivas e matriz de correlação.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    features = df.drop(columns=["target"])

    df.describe(include="all").to_csv(metrics_dir / "descriptive_statistics.csv")
    features.corr(numeric_only=True).to_csv(metrics_dir / "correlation_matrix.csv")
    class_distribution = df["target"].value_counts()
    class_distribution.to_csv(metrics_dir / "class_distribution.csv")

    fig = features.hist(figsize=(16, 12), bins=25)
    for axes in fig.flatten():
        axes.set_title(axes.get_title(), fontsize=8)
    plt.suptitle("Histogramas dos atributos", y=1.01)
    plt.tight_layout()
    plt.savefig(figures_dir / "histograms.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(16, 7))
    sns.boxplot(data=features, ax=ax, color="#86bf91", fliersize=2)
    ax.set_title("Boxplots dos atributos")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    fig.tight_layout()
    fig.savefig(figures_dir / "boxplots.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(features.corr(numeric_only=True), cmap="vlag", center=0, ax=ax)
    ax.set_title("Mapa de calor de correlação")
    fig.tight_layout()
    fig.savefig(figures_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=class_distribution.index, y=class_distribution.values, ax=ax, palette="Set2", hue=class_distribution.index, legend=False)
    ax.set_title("Distribuição das classes")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Frequência")
    fig.tight_layout()
    fig.savefig(figures_dir / "class_distribution.png", dpi=150)
    plt.close(fig)

    scaled = StandardScaler().fit_transform(features)
    pca = PCA(n_components=2, random_state=42)
    pca_points = pca.fit_transform(scaled)
    pca_df = pd.DataFrame({"PC1": pca_points[:, 0], "PC2": pca_points[:, 1], "target": df["target"].values})
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="target", s=35, alpha=0.75, ax=ax)
    ax.set_title("Visualização PCA 2D por classe real")
    fig.tight_layout()
    fig.savefig(figures_dir / "pca_initial_2d.png", dpi=150)
    plt.close(fig)

    return {
        "pca_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "descriptive_statistics_file": "results/metrics/descriptive_statistics.csv",
        "correlation_matrix_file": "results/metrics/correlation_matrix.csv",
    }
