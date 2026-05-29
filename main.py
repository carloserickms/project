from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import pandas as pd

from src.console_output import (
    print_artifacts,
    print_banner,
    print_footer,
    print_kmeans_summary,
    print_step,
    print_supervised_table,
)
from src.data_loader import infer_dataset_profile, load_dataset
from src.evaluation import automatic_interpretation, save_comparison_table
from src.preprocessing import clean_dataset, prepare_splits
from src.reporting import (
    compare_knn_scaling,
    format_classification_summary,
    interpret_kmeans,
    REQUIRED_MODELS,
)
from src.supervised import train_supervised_models
from src.unsupervised import run_kmeans_analysis
from src.utils import as_serializable, ensure_directories, save_json, save_text, set_global_seed
from src.visualization import generate_eda_artifacts


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "biodeg.csv"
DOCS_DIR = ROOT / "docs"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = RESULTS_DIR / "models"
REPORT_DIR = ROOT / "report"


def _load_doc_fragment(filename: str) -> str:
    path = DOCS_DIR / filename
    if not path.exists():
        return f"_(Arquivo {filename} não encontrado.)_"
    return path.read_text(encoding="utf-8")


def build_report(
    profile: dict,
    eda: dict,
    split_info: dict,
    comparison: pd.DataFrame,
    supervised_results: list[dict],
    unsupervised: dict,
    interpretation: str,
    kmeans_interpretation: str,
    class_names: list[str],
) -> str:
    best = comparison.iloc[0]
    class_dist = ", ".join([f"{cls}: {count}" for cls, count in profile["distribuicao_classes"].items()])
    dataset_doc = _load_doc_fragment("dataset_biodeg.md")

    required = [r for r in supervised_results if r["model_name"] in REQUIRED_MODELS]
    optional_knn = next((r for r in supervised_results if r["model_name"] == "KNN_sem_padronizacao"), None)

    params_required = "\n".join(
        [
            f"- **{item['model_name']}**: hiperparâmetros `{item['best_params']}`; "
            f"F1 validação={item['validation_f1_weighted']:.3f}, F1 teste={item['test_metrics']['f1']:.3f}."
            for item in required
        ]
    )

    per_class_sections = "\n\n".join(format_classification_summary(r, class_names) for r in required)

    comparison_cols = [c for c in comparison.columns if c != "melhores_parametros"]
    comparison_md = comparison[comparison_cols].to_markdown(index=False)

    knn_scaling_text = compare_knn_scaling(supervised_results)
    optional_knn_block = ""
    if optional_knn:
        optional_knn_block = (
            f"\n\n### Experimento complementar: KNN sem padronização\n\n"
            f"{knn_scaling_text}\n\n"
            f"Este experimento não substitui o KNN obrigatório; ilustra o efeito da escala dos atributos."
        )

    balance_text = (
        f"A razão entre classes é {profile['razao_desbalanceamento']:.2f} "
        f"({'há desbalanceamento moderado' if profile['ha_desbalanceamento'] else 'distribuição equilibrada'}). "
        "Não foi aplicado oversampling/SMOTE: a divisão estratificada preserva proporções e as métricas "
        "ponderadas (F1 weighted no GridSearch, relatório por classe no teste) permitem comparar modelos "
        "sem alterar a distribuição original. Pesos de classe (`class_weight`) podem ser testados em "
        "experimentos futuros se a classe minoritária (RB) continuar com recall baixo."
    )

    return f"""# Relatório de Aprendizagem de Máquina — Biodegradabilidade (T2)

**Disciplina:** Inteligência Artificial   
**Grupo:** Grupo 1  
**Integrantes:** Carlos Erick;, Eduardo Américo; Elias Reis; Hugo Gabriel  


---

## a) Apresentação do dataset

{dataset_doc}

### Perfil observado nesta execução

- Instâncias após limpeza: **{profile['n_linhas']}**
- Atributos preditivos: **{profile['n_atributos']}**
- Classes: {', '.join(profile['classes'])} — distribuição: {class_dist}
- Duplicados removidos na limpeza: diferença entre perfil bruto ({profile['perfil_bruto']['duplicados']}) e limpo ({profile['duplicados']})
- Outliers potenciais (IQR, soma por atributo): **{profile['total_outliers_iqr']}** — mantidos por plausibilidade química

Figuras de EDA: `results/figures/histograms.png`, `boxplots.png`, `correlation_heatmap.png`, `class_distribution.png`, `pca_initial_2d.png`.

---

## b) Análise e preparação dos dados

1. **Coerção numérica** com `errors='coerce'` e imputação pela **mediana** (robusta a outliers).
2. **Remoção de duplicatas** para evitar vazamento de instâncias idênticas entre conjuntos.
3. **Codificação do alvo** com `LabelEncoder` (NRB/RB → inteiros).
4. **Padronização** (`StandardScaler`) no treino para KNN e MLP (via `Pipeline`); árvore usa atributos brutos.
5. **Balanceamento:** {balance_text}

---

## c) Protocolo experimental

| Conjunto | Proporção | Uso |
|----------|-----------|-----|
| Treino | ~{split_info['treino']:.0%} | `GridSearchCV` (5-fold) para hiperparâmetros; ajuste do K-Means |
| Validação | ~{split_info['validacao']:.0%} | Monitoramento (`validation_f1_weighted`); **não** usado para escolher hiperparâmetros |
| Teste | ~{split_info['teste']:.0%} | Avaliação final **única** dos classificadores |

- `random_state={split_info['random_state']}`, divisão **estratificada**.
- O conjunto de **teste não participa** do treino, da busca de hiperparâmetros nem do K-Means.
- K-Means: apenas **treino**, rótulos ignorados no `fit`; scaler do treino supervisionado (sem `fit` no teste).
- Métrica principal de busca: **F1 ponderado** (`f1_weighted`).

---

## d) Modelagem supervisionada

Modelos obrigatórios: **KNN**, **Árvore de Decisão**, **RNA (MLPClassifier)**.

{params_required}

### Justificativa das escolhas

- **KNN:** variação de `k`, pesos uniforme/distance e métricas euclidiana/manhattan; padronização via pipeline.
- **Árvore:** critérios Gini/entropy, profundidade, `min_samples_leaf`, poda `ccp_alpha` — controle de complexidade e interpretabilidade.
- **MLP:** arquiteturas pequenas (1–2 camadas), `relu`/`tanh`, L2 (`alpha`), `early_stopping` para reduzir overfitting.

{optional_knn_block}

---

## e) Modelagem não supervisionada (K-Means)

- Escopo: conjunto de **{unsupervised.get('data_scope', 'treino')}** apenas.
- Valores de K testados: {unsupervised['tested_k']}.
- Escolha de K por **silhouette**: K={unsupervised['best_k_silhouette']} (silhouette={unsupervised['best_silhouette']:.3f}).
- Comparação com número de classes: K={unsupervised['k_equals_n_classes']} (ARI={unsupervised['k_n_classes_ari']:.3f}, NMI={unsupervised['k_n_classes_nmi']:.3f}).

Figuras: `kmeans_elbow.png`, `kmeans_silhouette.png`, `kmeans_pca_clusters.png`.

{kmeans_interpretation}

---

## f) Resultados

### Tabela comparativa (teste e validação)

{comparison_md}

### Métricas por classe (conjunto de teste)

{per_class_sections}

### Figuras de avaliação

- Comparação de métricas: `results/figures/model_comparison_metrics.png`
- Matrizes de confusão e ROC: `results/figures/confusion_matrix_*.png`, `roc_curve_*.png`
- Árvore (profundidade limitada na figura): `results/figures/decision_tree.png`

Variância explicada PCA (EDA): {eda['pca_variance_ratio']}.

---

## g) Análise crítica comparativa

{interpretation}

{knn_scaling_text}

**Síntese:** O melhor modelo supervisionado no teste foi **{best['modelo']}** (F1={best['f1']:.3f}). A árvore oferece interpretabilidade por limiares; a MLP tem maior flexibilidade mas exigiu mais regularização; o KNN beneficiou-se da padronização. No não supervisionado, silhouette e ARI podem divergir (K ótimo geométrico ≠ alinhamento com RB/NRB), o que reforça que a fronteira de decisão supervisionada não coincide com clusters esféricos globais.

**Limitações:** partição única treino/val/teste; descritores sem nomenclatura química no CSV; possível desbalanceamento afetando recall de RB; K-Means assume clusters convexos e de variância similar.

---

## h) Conclusão

O trabalho cumpriu o protocolo de comparação entre KNN, árvore e RNA no problema de biodegradabilidade, com K-Means no mesmo conjunto de atributos (treino). As principais dificuldades foram interpretar baixo ARI no clustering e conciliar métricas globais com desbalanceamento moderado. Como evolução: validação cruzada aninhada, `class_weight`, seleção de atributos e análise química dos descritores mais relevantes na árvore.

---

## i) Referências

1. Mansouri, K., et al. (2013). Quantitative structure–activity relationship models for ready biodegradability of chemicals. *Journal of Chemical Information and Modeling*, 53(4), 867–878.
2. UCI Machine Learning Repository — datasets QSAR / biodegradation.
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
4. Documentação: [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/).

---

## Apêndice — Uso de IA generativa

IA generativa foi utilizada como apoio para estruturar código, documentar parâmetros e redigir seções iniciais do relatório. O grupo revisou e compreende as decisões metodológicas.

Prompts registrados em: [`docs/prompts_genai.md`](../docs/prompts_genai.md).

---

## Artefatos gerados

- Figuras: `results/figures/`
- Métricas: `results/metrics/`
- Modelos: `results/models/` (`.joblib`, `decision_tree.dot`)
"""


def main() -> None:
    import sys
    import time

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    t0 = time.perf_counter()
    total_steps = 8

    print_banner("Pipeline ML — Biodegradabilidade", f"Dataset: {DATASET_PATH.name}")

    set_global_seed()
    ensure_directories([FIGURES_DIR, METRICS_DIR, MODELS_DIR, REPORT_DIR])

    print_step(1, total_steps, "Carregando dataset", "run")
    raw_df = load_dataset(DATASET_PATH)
    raw_profile = infer_dataset_profile(raw_df)
    clean_df = clean_dataset(raw_df)
    profile = infer_dataset_profile(clean_df)
    profile["perfil_bruto"] = raw_profile
    print_step(1, total_steps, f"Dataset: {profile['n_linhas']} instâncias, {profile['n_atributos']} atributos")

    print_step(2, total_steps, "Salvando perfil do dataset", "run")
    save_json(as_serializable(profile), METRICS_DIR / "dataset_profile.json")
    print_step(2, total_steps, "Perfil salvo em results/metrics/")

    print_step(3, total_steps, "Gerando EDA (figuras e CSVs)", "run")
    eda = generate_eda_artifacts(clean_df, FIGURES_DIR, METRICS_DIR)
    print_step(3, total_steps, "EDA concluída")

    print_step(4, total_steps, "Dividindo treino / validação / teste", "run")
    splits = prepare_splits(clean_df)
    class_names = list(splits["label_encoder"].classes_)
    si = splits["split_info"]
    print_step(
        4,
        total_steps,
        f"Split {si['treino']:.0%} / {si['validacao']:.0%} / {si['teste']:.0%} (estratificado)",
    )

    print_step(5, total_steps, "Treinando modelos supervisionados (GridSearchCV)", "run")
    supervised_results = train_supervised_models(splits, class_names, FIGURES_DIR, MODELS_DIR)
    supervised_json = {
        item["model_name"]: {key: value for key, value in item.items() if key != "model"}
        for item in supervised_results
    }
    save_json(as_serializable(supervised_json), METRICS_DIR / "supervised_results.json")
    print_step(5, total_steps, "KNN, Árvore, MLP (+ KNN sem escala) treinados")

    print_step(6, total_steps, "Executando K-Means", "run")
    comparison = save_comparison_table(supervised_results, METRICS_DIR, FIGURES_DIR)
    unsupervised = run_kmeans_analysis(splits, FIGURES_DIR, METRICS_DIR)
    save_json(as_serializable(unsupervised), METRICS_DIR / "kmeans_summary.json")
    print_step(6, total_steps, f"K-Means: melhor K={unsupervised['best_k_silhouette']} (silhouette)")

    print_step(7, total_steps, "Gerando interpretações e relatório", "run")
    crosstab_path = METRICS_DIR / "kmeans_cluster_vs_class.csv"
    crosstab = pd.read_csv(crosstab_path, index_col=0) if crosstab_path.exists() else None

    interpretation = automatic_interpretation(comparison, unsupervised)
    kmeans_interpretation = interpret_kmeans(unsupervised, crosstab)
    save_text(interpretation, METRICS_DIR / "automatic_interpretation.txt")
    save_text(kmeans_interpretation, METRICS_DIR / "kmeans_interpretation.txt")

    report = build_report(
        profile,
        eda,
        splits["split_info"],
        comparison,
        supervised_results,
        unsupervised,
        interpretation,
        kmeans_interpretation,
        class_names,
    )
    save_text(report, REPORT_DIR / "relatorio.md")
    print_step(7, total_steps, "relatorio.md atualizado")

    print_supervised_table(comparison)
    print_kmeans_summary(unsupervised)
    print_artifacts()
    print_footer(time.perf_counter() - t0)


if __name__ == "__main__":
    main()
