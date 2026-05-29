# Relatório de Aprendizagem de Máquina — Biodegradabilidade (T2)

**Disciplina:** Inteligência Artificial   
**Grupo:** Grupo 1  
**Integrantes:** Carlos Erick;, Eduardo Américo; Elias Reis; Hugo Gabriel  


---

## a) Apresentação do dataset

# Dataset biodeg (biodegradabilidade)

## Problema

Classificar compostos químicos quanto à **biodegradabilidade** segundo critérios de prontidão para biodegradação. Trata-se de um problema de **classificação binária** em que cada instância é um composto descrito por descritores moleculares numéricos (abordagem QSAR).

## Rótulos

| Classe | Significado |
|--------|-------------|
| **RB** | Ready Biodegradable — composto considerado prontamente biodegradável |
| **NRB** | Not Ready Biodegradable — composto não classificado como prontamente biodegradável |

No arquivo `biodeg.csv`, a variável alvo está na **última coluna**, sem cabeçalho.

## Atributos

- **41 descritores moleculares** numéricos (colunas 1 a 41).
- No repositório local os atributos são nomeados `feature_01` … `feature_41`, pois o CSV não traz nomes químicos individuais.
- Representam propriedades estruturais/físico-químicas usadas em modelagem QSAR (escalas heterogêneas).

## Formato do arquivo

- Arquivo: `biodeg.csv`
- Separador: `;` (ponto e vírgula)
- Sem linha de cabeçalho
- Aproximadamente **1050+ instâncias** após limpeza (remoção de duplicatas)

## Origem e referência

Dataset amplamente usado em estudos de QSAR e disponível em repositórios de aprendizado de máquina (UCI Machine Learning Repository, conjunto relacionado à biodegradabilidade de compostos orgânicos).

**Referência sugerida para o relatório:**

- Mansouri, K., et al. (2013). *Quantitative structure–activity relationship models for ready biodegradability of chemicals.* Journal of Chemical Information and Modeling, 53(4), 867–878. (contexto QSAR de biodegradabilidade)
- UCI Machine Learning Repository — conjuntos QSAR / biodegradation (consultar a página do dataset utilizado pelo professor da disciplina)

## Uso neste projeto

O pipeline assume automaticamente que a última coluna é o alvo e que todas as demais são preditores numéricos. Não há variáveis categóricas brutas além do rótulo.


### Perfil observado nesta execução

- Instâncias após limpeza: **1052**
- Atributos preditivos: **41**
- Classes: NRB, RB — distribuição: NRB: 698, RB: 354
- Duplicados removidos na limpeza: diferença entre perfil bruto (3) e limpo (0)
- Outliers potenciais (IQR, soma por atributo): **1876** — mantidos por plausibilidade química

Figuras de EDA: `results/figures/histograms.png`, `boxplots.png`, `correlation_heatmap.png`, `class_distribution.png`, `pca_initial_2d.png`.

---

## b) Análise e preparação dos dados

1. **Coerção numérica** com `errors='coerce'` e imputação pela **mediana** (robusta a outliers).
2. **Remoção de duplicatas** para evitar vazamento de instâncias idênticas entre conjuntos.
3. **Codificação do alvo** com `LabelEncoder` (NRB/RB → inteiros).
4. **Padronização** (`StandardScaler`) no treino para KNN e MLP (via `Pipeline`); árvore usa atributos brutos.
5. **Balanceamento:** A razão entre classes é 1.97 (há desbalanceamento moderado). Não foi aplicado oversampling/SMOTE: a divisão estratificada preserva proporções e as métricas ponderadas (F1 weighted no GridSearch, relatório por classe no teste) permitem comparar modelos sem alterar a distribuição original. Pesos de classe (`class_weight`) podem ser testados em experimentos futuros se a classe minoritária (RB) continuar com recall baixo.

---

## c) Protocolo experimental

| Conjunto | Proporção | Uso |
|----------|-----------|-----|
| Treino | ~60% | `GridSearchCV` (5-fold) para hiperparâmetros; ajuste do K-Means |
| Validação | ~20% | Monitoramento (`validation_f1_weighted`); **não** usado para escolher hiperparâmetros |
| Teste | ~20% | Avaliação final **única** dos classificadores |

- `random_state=42`, divisão **estratificada**.
- O conjunto de **teste não participa** do treino, da busca de hiperparâmetros nem do K-Means.
- K-Means: apenas **treino**, rótulos ignorados no `fit`; scaler do treino supervisionado (sem `fit` no teste).
- Métrica principal de busca: **F1 ponderado** (`f1_weighted`).

---

## d) Modelagem supervisionada

Modelos obrigatórios: **KNN**, **Árvore de Decisão**, **RNA (MLPClassifier)**.

- **KNN_padronizado**: hiperparâmetros `{'model__metric': 'euclidean', 'model__n_neighbors': 11, 'model__weights': 'distance'}`; F1 validação=0.842, F1 teste=0.784.
- **Arvore_Decisao**: hiperparâmetros `{'ccp_alpha': 0.005, 'criterion': 'gini', 'max_depth': 5, 'min_samples_leaf': 1}`; F1 validação=0.821, F1 teste=0.760.
- **MLP_RNA**: hiperparâmetros `{'model__activation': 'tanh', 'model__alpha': 0.0001, 'model__hidden_layer_sizes': (32, 16), 'model__learning_rate_init': 0.01}`; F1 validação=0.869, F1 teste=0.738.

### Justificativa das escolhas

- **KNN:** variação de `k`, pesos uniforme/distance e métricas euclidiana/manhattan; padronização via pipeline.
- **Árvore:** critérios Gini/entropy, profundidade, `min_samples_leaf`, poda `ccp_alpha` — controle de complexidade e interpretabilidade.
- **MLP:** arquiteturas pequenas (1–2 camadas), `relu`/`tanh`, L2 (`alpha`), `early_stopping` para reduzir overfitting.



### Experimento complementar: KNN sem padronização

O KNN padronizado (F1=0.784) superou o KNN sem padronização (F1=0.755, Δ=+0.029), confirmando que atributos em escalas heterogêneas afetam algoritmos baseados em distância.

Este experimento não substitui o KNN obrigatório; ilustra o efeito da escala dos atributos.

---

## e) Modelagem não supervisionada (K-Means)

- Escopo: conjunto de **treino** apenas.
- Valores de K testados: [2, 3, 4, 5, 6, 7, 8, 9].
- Escolha de K por **silhouette**: K=3 (silhouette=0.186).
- Comparação com número de classes: K=2 (ARI=0.039, NMI=0.101).

Figuras: `kmeans_elbow.png`, `kmeans_silhouette.png`, `kmeans_pca_clusters.png`.

O K-Means foi ajustado apenas no conjunto de **treino** (treino), sem usar o conjunto de teste e sem utilizar rótulos no ajuste dos centróides. A escala dos atributos reutiliza o `StandardScaler` calibrado no treino supervisionado.

Por **silhouette**, o melhor K foi **3** (silhouette=0.186, ARI=0.013, NMI=0.090). Para alinhar ao problema **binário** (duas classes), K=2 obteve silhouette=0.181, ARI=0.039, NMI=0.101.

Embora K=3 maximize a separação geométrica (silhouette), **K=2 apresenta maior ARI** em relação aos rótulos reais, o que é esperado quando o objetivo supervisionado é binário.

**Tabela cluster × classe (K por silhouette):**

|   cluster |   NRB |   RB |
|----------:|------:|-----:|
|         0 |   210 |  181 |
|         1 |   178 |   29 |
|         2 |    30 |    2 |

**Interpretação por cluster:**
- Cluster 0: predominância de **NRB** (54% das 391 amostras).
- Cluster 1: predominância de **NRB** (86% das 207 amostras).
- Cluster 2: predominância de **NRB** (94% das 32 amostras).

Conclusão parcial: a classificação supervisionada explora fronteiras mais complexas do que partições esféricas do K-Means; baixo ARI não invalida os modelos supervisionados, mas mostra que a estrutura não supervisionada é fraca ou não coincide com os rótulos oficiais.

---

## f) Resultados

### Tabela comparativa (teste e validação)

| modelo               |   f1_validacao |   accuracy |   precision |   recall |       f1 |   tempo_treino_s |   roc_auc |
|:---------------------|---------------:|-----------:|------------:|---------:|---------:|-----------------:|----------:|
| KNN_padronizado      |       0.842391 |   0.843602 |    0.731707 | 0.84507  | 0.784314 |         3.25882  |  0.91826  |
| Arvore_Decisao       |       0.820508 |   0.829384 |    0.721519 | 0.802817 | 0.76     |         1.20847  |  0.875201 |
| KNN_sem_padronizacao |       0.841179 |   0.824645 |    0.7125   | 0.802817 | 0.754967 |         0.235887 |  0.900252 |
| MLP_RNA              |       0.868536 |   0.815166 |    0.705128 | 0.774648 | 0.738255 |         1.24327  |  0.884708 |

### Métricas por classe (conjunto de teste)

#### KNN_padronizado

| Classe | Precision | Recall | F1 | Suporte |
|--------|-----------|--------|-----|---------|
| NRB | 0.915 | 0.843 | 0.877 | 140 |
| RB | 0.732 | 0.845 | 0.784 | 71 |

F1 validação (ponderado): 0.842 | F1 teste: 0.784 | Acurácia teste: 0.844

Matriz de confusão: `results/figures/confusion_matrix_KNN_padronizado.png`
Curva ROC: `results/figures/roc_curve_KNN_padronizado.png` (AUC=0.918)

#### Arvore_Decisao

| Classe | Precision | Recall | F1 | Suporte |
|--------|-----------|--------|-----|---------|
| NRB | 0.894 | 0.843 | 0.868 | 140 |
| RB | 0.722 | 0.803 | 0.760 | 71 |

F1 validação (ponderado): 0.821 | F1 teste: 0.760 | Acurácia teste: 0.829

Matriz de confusão: `results/figures/confusion_matrix_Arvore_Decisao.png`
Curva ROC: `results/figures/roc_curve_Arvore_Decisao.png` (AUC=0.875)

#### MLP_RNA

| Classe | Precision | Recall | F1 | Suporte |
|--------|-----------|--------|-----|---------|
| NRB | 0.880 | 0.836 | 0.857 | 140 |
| RB | 0.705 | 0.775 | 0.738 | 71 |

F1 validação (ponderado): 0.869 | F1 teste: 0.738 | Acurácia teste: 0.815

Matriz de confusão: `results/figures/confusion_matrix_MLP_RNA.png`
Curva ROC: `results/figures/roc_curve_MLP_RNA.png` (AUC=0.885)

### Figuras de avaliação

- Comparação de métricas: `results/figures/model_comparison_metrics.png`
- Matrizes de confusão e ROC: `results/figures/confusion_matrix_*.png`, `roc_curve_*.png`
- Árvore (profundidade limitada na figura): `results/figures/decision_tree.png`

Variância explicada PCA (EDA): [0.18048658033105366, 0.12260365281672904].

---

## g) Análise crítica comparativa

O melhor desempenho supervisionado no teste foi obtido por KNN_padronizado, com F1=0.784 e accuracy=0.844. O menor F1 pertenceu a MLP_RNA (0.738). No K-Means (treino), o melhor K por silhouette foi 3 (silhouette=0.186, ARI=0.013); com K=2 (número de classes), ARI=0.039.

O KNN padronizado (F1=0.784) superou o KNN sem padronização (F1=0.755, Δ=+0.029), confirmando que atributos em escalas heterogêneas afetam algoritmos baseados em distância.

**Síntese:** O melhor modelo supervisionado no teste foi **KNN_padronizado** (F1=0.784). A árvore oferece interpretabilidade por limiares; a MLP tem maior flexibilidade mas exigiu mais regularização; o KNN beneficiou-se da padronização. No não supervisionado, silhouette e ARI podem divergir (K ótimo geométrico ≠ alinhamento com RB/NRB), o que reforça que a fronteira de decisão supervisionada não coincide com clusters esféricos globais.

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
