from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_classifier(model, X, y_true, class_names: list[str], model_name: str, figures_dir: Path) -> dict:
    """Avalia um modelo de classificação, calculando métricas de desempenho, gerando a matriz de confusão e a curva ROC (se aplicável). As métricas incluem acurácia, precisão, recall, F1-score e AUC-ROC para problemas binários. Os resultados são salvos em arquivos e retornados em um dicionário.

    Args:
        model (object): O modelo de classificação a ser avaliado.
        X (array-like): Os dados de entrada para o modelo.
        y_true (array-like): Os rótulos verdadeiros correspondentes aos dados de entrada.
        class_names (list[str]): Uma lista de nomes das classes.
        model_name (str): O nome do modelo a ser avaliado.
        figures_dir (Path): O diretório onde as figuras serão salvas.

    Returns:
        dict: Um dicionário contendo as métricas de desempenho, a matriz de confusão e a curva ROC.
    """
    y_pred = model.predict(X)
    average = "binary" if len(class_names) == 2 else "weighted"
    pos_label = 1 if len(class_names) == 2 else None
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0),
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names, cmap="Blues", ax=ax)
    ax.set_title(f"Matriz de confusão - {model_name}")
    fig.tight_layout()
    fig.savefig(figures_dir / f"confusion_matrix_{model_name}.png", dpi=150)
    plt.close(fig)

    if len(class_names) == 2 and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y_true, probabilities)
        auc = roc_auc_score(y_true, probabilities)
        metrics["roc_auc"] = auc
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("Falso positivo")
        ax.set_ylabel("Verdadeiro positivo")
        ax.set_title(f"Curva ROC - {model_name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / f"roc_curve_{model_name}.png", dpi=150)
        plt.close(fig)
    return metrics


def save_comparison_table(results: list[dict], metrics_dir: Path, figures_dir: Path) -> pd.DataFrame:
    """Salva uma tabela comparativa dos resultados de avaliação dos modelos de classificação, incluindo métricas como acurácia, precisão, recall, F1-score e AUC-ROC (se aplicável). A tabela é salva em formato CSV e uma visualização gráfica das métricas é gerada.

    Args:
        results (list[dict]): Uma lista de dicionários contendo os resultados de avaliação de cada modelo de classificação.
        metrics_dir (Path): O diretório onde os arquivos de saída serão salvos.
        figures_dir (Path): O diretório onde as figuras serão salvas.

    Returns:
        pd.DataFrame: Uma tabela comparativa dos resultados de avaliação dos modelos de classificação.
    """
    rows = []
    for result in results:
        row = {
            "modelo": result["model_name"],
            "f1_validacao": result["validation_f1_weighted"],
            "accuracy": result["test_metrics"]["accuracy"],
            "precision": result["test_metrics"]["precision"],
            "recall": result["test_metrics"]["recall"],
            "f1": result["test_metrics"]["f1"],
            "tempo_treino_s": result["train_time"],
            "melhores_parametros": result["best_params"],
        }
        if "roc_auc" in result["test_metrics"]:
            row["roc_auc"] = result["test_metrics"]["roc_auc"]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    df.to_csv(metrics_dir / "supervised_comparison.csv", index=False)

    plot_df = df.melt(id_vars="modelo", value_vars=["accuracy", "precision", "recall", "f1"], var_name="metrica")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=plot_df, x="modelo", y="value", hue="metrica", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Comparação de métricas no conjunto de teste")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Valor")
    ax.legend(title="Métrica")
    fig.tight_layout()
    fig.savefig(figures_dir / "model_comparison_metrics.png", dpi=150)
    plt.close(fig)
    return df


def automatic_interpretation(comparison: pd.DataFrame, unsupervised_summary: dict) -> str:
    """
    Gera uma interpretação automática dos resultados de avaliação dos modelos de classificação, destacando o melhor e o pior desempenho com base nas métricas calculadas. A interpretação inclui insights sobre a diferença prática entre as hipóteses aprendidas e a qualidade da separação dos clusters no K-Means, com base na métrica silhouette.

    Args:
        comparison (pd.DataFrame): A tabela comparativa dos resultados de avaliação dos modelos de classificação.
        unsupervised_summary (dict): Um dicionário contendo o resumo dos resultados do modelo K-Means, incluindo o melhor K e a métrica silhouette correspondente.

    Returns:
        str: Uma interpretação automática dos resultados de avaliação dos modelos de classificação.
    """
    best = comparison.iloc[0]
    worst = comparison.iloc[-1]
    k_nc = unsupervised_summary.get("k_equals_n_classes", 2)
    return (
        f"O melhor desempenho supervisionado no teste foi obtido por {best['modelo']}, com F1={best['f1']:.3f} "
        f"e accuracy={best['accuracy']:.3f}. O menor F1 pertenceu a {worst['modelo']} "
        f"({worst['f1']:.3f}). No K-Means (treino), o melhor K por silhouette foi "
        f"{unsupervised_summary['best_k_silhouette']} (silhouette={unsupervised_summary['best_silhouette']:.3f}, "
        f"ARI={unsupervised_summary.get('best_k_ari', unsupervised_summary.get('adjusted_rand_index', 0)):.3f}); "
        f"com K={k_nc} (número de classes), ARI={unsupervised_summary.get('k_n_classes_ari', 0):.3f}."
    )
