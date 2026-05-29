from __future__ import annotations

import pandas as pd


REQUIRED_MODELS = {"KNN_padronizado", "Arvore_Decisao", "MLP_RNA"}


def format_classification_summary(result: dict, class_names: list[str]) -> str:
    """Gera markdown com métricas por classe a partir do classification_report."""
    report = result["test_metrics"]["classification_report"]
    lines = [f"#### {result['model_name']}", "", "| Classe | Precision | Recall | F1 | Suporte |", "|--------|-----------|--------|-----|---------|"]
    for cls in class_names:
        if cls not in report:
            continue
        row = report[cls]
        lines.append(
            f"| {cls} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1-score']:.3f} | {int(row['support'])} |"
        )
    lines.append(
        f"\nF1 validação (ponderado): {result['validation_f1_weighted']:.3f} | "
        f"F1 teste: {result['test_metrics']['f1']:.3f} | "
        f"Acurácia teste: {result['test_metrics']['accuracy']:.3f}"
    )
    lines.append(f"\nMatriz de confusão: `results/figures/confusion_matrix_{result['model_name']}.png`")
    if "roc_auc" in result["test_metrics"]:
        lines.append(f"Curva ROC: `results/figures/roc_curve_{result['model_name']}.png` (AUC={result['test_metrics']['roc_auc']:.3f})")
    return "\n".join(lines)


def interpret_kmeans(unsupervised: dict, crosstab: pd.DataFrame | None = None) -> str:
    """Interpretação qualitativa dos agrupamentos e comparação K=2 vs melhor K por silhouette."""
    best_k = unsupervised["best_k_silhouette"]
    k_nc = unsupervised["k_equals_n_classes"]
    parts = [
        f"O K-Means foi ajustado apenas no conjunto de **treino** ({unsupervised.get('data_scope', 'treino')}), "
        "sem usar o conjunto de teste e sem utilizar rótulos no ajuste dos centróides. "
        f"A escala dos atributos reutiliza o `StandardScaler` calibrado no treino supervisionado.",
        "",
        f"Por **silhouette**, o melhor K foi **{best_k}** (silhouette={unsupervised['best_silhouette']:.3f}, "
        f"ARI={unsupervised['best_k_ari']:.3f}, NMI={unsupervised['best_k_nmi']:.3f}). "
        f"Para alinhar ao problema **binário** (duas classes), K={k_nc} obteve "
        f"silhouette={unsupervised['k_n_classes_silhouette']:.3f}, "
        f"ARI={unsupervised['k_n_classes_ari']:.3f}, NMI={unsupervised['k_n_classes_nmi']:.3f}.",
    ]
    if unsupervised["k_n_classes_ari"] > unsupervised["best_k_ari"]:
        parts.append(
            f"\nEmbora K={best_k} maximize a separação geométrica (silhouette), **K={k_nc} apresenta maior ARI** "
            "em relação aos rótulos reais, o que é esperado quando o objetivo supervisionado é binário."
        )
    else:
        parts.append(
            f"\nMesmo com K={k_nc}, o **ARI permanece baixo** ({unsupervised['k_n_classes_ari']:.3f}), "
            "indicando que os agrupamentos globais por distância euclidiana não reproduzem bem RB/NRB."
        )

    if crosstab is not None and not crosstab.empty:
        parts.append("\n**Tabela cluster × classe (K por silhouette):**\n")
        parts.append(crosstab.to_markdown())
        dominant = []
        for cluster_id in crosstab.index:
            row = crosstab.loc[cluster_id]
            total = row.sum()
            if total == 0:
                continue
            majority = row.idxmax()
            purity = row.max() / total
            dominant.append(f"- Cluster {cluster_id}: predominância de **{majority}** ({purity:.0%} das {int(total)} amostras).")
        if dominant:
            parts.append("\n**Interpretação por cluster:**\n" + "\n".join(dominant))

    parts.append(
        "\nConclusão parcial: a classificação supervisionada explora fronteiras mais complexas do que "
        "partições esféricas do K-Means; baixo ARI não invalida os modelos supervisionados, mas mostra "
        "que a estrutura não supervisionada é fraca ou não coincide com os rótulos oficiais."
    )
    return "\n".join(parts)


def compare_knn_scaling(supervised_results: list[dict]) -> str:
    """Compara KNN com e sem padronização."""
    by_name = {r["model_name"]: r for r in supervised_results}
    scaled = by_name.get("KNN_padronizado")
    raw = by_name.get("KNN_sem_padronizacao")
    if not scaled or not raw:
        return ""
    diff = scaled["test_metrics"]["f1"] - raw["test_metrics"]["f1"]
    return (
        f"O KNN padronizado (F1={scaled['test_metrics']['f1']:.3f}) "
        f"{'superou' if diff > 0 else 'ficou abaixo de'} o KNN sem padronização "
        f"(F1={raw['test_metrics']['f1']:.3f}, Δ={diff:+.3f}), confirmando que atributos em escalas "
        "heterogêneas afetam algoritmos baseados em distância."
    )
