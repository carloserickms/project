from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loader import infer_dataset_profile, load_dataset
from src.evaluation import automatic_interpretation, save_comparison_table
from src.preprocessing import clean_dataset, prepare_splits
from src.supervised import train_supervised_models
from src.unsupervised import run_kmeans_analysis
from src.utils import as_serializable, ensure_directories, save_json, save_text, set_global_seed
from src.visualization import generate_eda_artifacts


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "biodeg.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = RESULTS_DIR / "models"
REPORT_DIR = ROOT / "report"


def build_report(profile: dict, eda: dict, split_info: dict, comparison: pd.DataFrame, supervised_results: list[dict], unsupervised: dict, interpretation: str) -> str:
    best = comparison.iloc[0]
    class_dist = ", ".join([f"{cls}: {count}" for cls, count in profile["distribuicao_classes"].items()])
    params_text = "\n".join(
        [
            f"- **{item['model_name']}**: melhores hiperparâmetros `{item['best_params']}`, "
            f"F1 teste={item['test_metrics']['f1']:.3f}, tempo de treino={item['train_time']:.2f}s."
            for item in supervised_results
        ]
    )
    comparison_md = comparison.drop(columns=["melhores_parametros"], errors="ignore").to_markdown(index=False)
    return f"""# Relatório de Aprendizagem de Máquina - Biodegradabilidade

## 1. Introdução

Este trabalho implementa um pipeline clássico de Aprendizagem de Máquina para classificar amostras do dataset `biodeg.csv`. O objetivo é comparar algoritmos supervisionados com diferentes vieses indutivos e investigar, por K-Means, se os atributos apresentam agrupamentos naturais coerentes com os rótulos reais.

## 2. Dataset

O arquivo foi carregado sem cabeçalho, com separador `;`, e a última coluna foi tratada automaticamente como variável alvo. Foram identificadas {profile['n_linhas']} linhas, {profile['n_atributos']} atributos preditivos e {profile['n_colunas_total']} colunas totais.

As classes inferidas foram: {', '.join(profile['classes'])}. A distribuição observada foi {class_dist}. A razão entre a maior e a menor classe foi {profile['razao_desbalanceamento']:.2f}, portanto há {'indício de desbalanceamento' if profile['ha_desbalanceamento'] else 'distribuição relativamente equilibrada'}.

Todos os atributos preditivos foram avaliados como numéricos após coerção controlada. O perfil geral indica uma base tabular numérica, com escalas heterogêneas entre atributos, o que justifica padronização para algoritmos sensíveis a distância ou gradiente. Foram encontrados {profile['duplicados']} registros duplicados e {profile['valores_nulos_total']} valores nulos no carregamento bruto. A análise por IQR indicou {profile['total_outliers_iqr']} ocorrências potenciais de outliers, mantidas por poderem representar compostos quimicamente plausíveis.

## 3. Pré-processamento

O pré-processamento separou atributos e rótulo, codificou a classe por `LabelEncoder`, converteu atributos para valores numéricos, removeu duplicados e imputou eventuais ausências pela mediana. A divisão adotada foi estratificada, com proporções aproximadas de {split_info['treino']:.0%} treino, {split_info['validacao']:.0%} validação e {split_info['teste']:.0%} teste, usando `random_state={split_info['random_state']}`.

A padronização por `StandardScaler` foi aplicada nos modelos que dependem de magnitude. No KNN, atributos em escalas maiores dominam o cálculo de distância. Na MLP, escalas incompatíveis dificultam a otimização por gradiente e podem atrasar convergência.

## 4. Modelos Supervisionados

Foram avaliados KNN, Árvore de Decisão e uma Rede Neural Artificial do tipo MLP. O KNN testou diferentes valores de vizinhos, métricas de distância e pesos, incluindo comparação explícita com e sem padronização. A árvore testou critérios Gini/Entropy, profundidades máximas, folhas mínimas e `ccp_alpha`, permitindo controle de complexidade e poda. A MLP testou arquiteturas com uma ou duas camadas ocultas, funções `relu` e `tanh`, taxas de aprendizado e regularização L2.

Na MLP, a função de ativação introduz não linearidade. O `early_stopping` foi usado para reduzir overfitting, interrompendo o treino quando a validação interna deixa de melhorar. A convergência depende de escala, taxa de aprendizado e complexidade da arquitetura.

{params_text}

Tabela resumida:

{comparison_md}

## 5. Aprendizagem Não Supervisionada

O K-Means foi executado ignorando os rótulos. Foram testados K={unsupervised['tested_k']}, avaliando inércia pelo método do cotovelo e `silhouette score` como medida de separação geométrica. O melhor K por silhouette foi {unsupervised['best_k_silhouette']}, com silhouette={unsupervised['best_silhouette']:.3f}.

A comparação entre clusters e classes reais foi salva em `results/metrics/kmeans_cluster_vs_class.csv`. O ARI foi {unsupervised['adjusted_rand_index']:.3f} e o NMI foi {unsupervised['normalized_mutual_info']:.3f}. Esses indicadores quantificam quanto a estrutura não supervisionada coincide com a rotulagem conhecida.

## 6. Comparação de Resultados

{interpretation}

Em termos metodológicos, o KNN é simples e competitivo quando a geometria dos dados favorece vizinhança local, mas tem custo de predição maior e depende fortemente de escala. A Árvore de Decisão é mais interpretável e pouco sensível à normalização, porém tende a overfitting quando cresce sem restrição. A MLP possui maior flexibilidade funcional, mas exige mais cuidado com padronização, regularização, épocas e convergência.

A comparação entre KNN padronizado e não padronizado evidencia o impacto do pré-processamento. Quando há diferença relevante, ela confirma que a escala dos atributos influencia diretamente algoritmos baseados em distância. A árvore atua como contraponto interpretável porque divide atributos por limiares e não por distância euclidiana.

## 7. Conclusão

O melhor modelo neste experimento foi **{best['modelo']}**, com F1={best['f1']:.3f} no teste. A principal limitação é que o estudo depende de uma única partição treino/validação/teste; como melhoria futura, recomenda-se validação cruzada aninhada, seleção de atributos, análise química dos descritores e calibração probabilística.

Os resultados do K-Means ajudam a discutir se a separação supervisionada decorre de grupos naturalmente bem definidos. Caso ARI/NMI sejam baixos, a classificação depende de fronteiras mais complexas do que simples agrupamentos globulares.

## 8. Uso de IA Generativa

IA generativa foi utilizada como apoio para estruturar o código, organizar o relatório e redigir interpretações técnicas iniciais. O grupo declara compreender o conteúdo gerado, incluindo decisões de pré-processamento, funcionamento dos algoritmos, métricas calculadas e limitações metodológicas.

## Artefatos Gerados

- Figuras: `results/figures/`
- Métricas: `results/metrics/`
- Modelos: `results/models/`
- Variância explicada PCA inicial: {eda['pca_variance_ratio']}
"""


def main() -> None:
    set_global_seed()
    ensure_directories([FIGURES_DIR, METRICS_DIR, MODELS_DIR, REPORT_DIR])

    raw_df = load_dataset(DATASET_PATH)
    raw_profile = infer_dataset_profile(raw_df)
    clean_df = clean_dataset(raw_df)
    profile = infer_dataset_profile(clean_df)
    profile["perfil_bruto"] = raw_profile
    save_json(as_serializable(profile), METRICS_DIR / "dataset_profile.json")

    eda = generate_eda_artifacts(clean_df, FIGURES_DIR, METRICS_DIR)
    splits = prepare_splits(clean_df)
    class_names = list(splits["label_encoder"].classes_)

    supervised_results = train_supervised_models(splits, class_names, FIGURES_DIR, MODELS_DIR)
    supervised_json = {
        item["model_name"]: {key: value for key, value in item.items() if key != "model"}
        for item in supervised_results
    }
    save_json(as_serializable(supervised_json), METRICS_DIR / "supervised_results.json")

    comparison = save_comparison_table(supervised_results, METRICS_DIR, FIGURES_DIR)
    unsupervised = run_kmeans_analysis(splits, FIGURES_DIR, METRICS_DIR)
    save_json(as_serializable(unsupervised), METRICS_DIR / "kmeans_summary.json")

    interpretation = automatic_interpretation(comparison, unsupervised)
    save_text(interpretation, METRICS_DIR / "automatic_interpretation.txt")
    report = build_report(profile, eda, splits["split_info"], comparison, supervised_results, unsupervised, interpretation)
    save_text(report, REPORT_DIR / "relatorio.md")

    print("Pipeline concluído.")
    print(f"Melhor modelo: {comparison.iloc[0]['modelo']} | F1={comparison.iloc[0]['f1']:.3f}")
    print("Relatório: report/relatorio.md")


if __name__ == "__main__":
    main()
