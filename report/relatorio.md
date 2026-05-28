# Relatório de Aprendizagem de Máquina - Biodegradabilidade

## 1. Introdução

Este trabalho implementa um pipeline clássico de Aprendizagem de Máquina para classificar amostras do dataset `biodeg.csv`. O objetivo é comparar algoritmos supervisionados com diferentes vieses indutivos e investigar, por K-Means, se os atributos apresentam agrupamentos naturais coerentes com os rótulos reais.

## 2. Dataset

O arquivo foi carregado sem cabeçalho, com separador `;`, e a última coluna foi tratada automaticamente como variável alvo. Foram identificadas 1052 linhas, 41 atributos preditivos e 42 colunas totais.

As classes inferidas foram: NRB, RB. A distribuição observada foi NRB: 698, RB: 354. A razão entre a maior e a menor classe foi 1.97, portanto há indício de desbalanceamento.

Todos os atributos preditivos foram avaliados como numéricos após coerção controlada. O perfil geral indica uma base tabular numérica, com escalas heterogêneas entre atributos, o que justifica padronização para algoritmos sensíveis a distância ou gradiente. Foram encontrados 0 registros duplicados e 0 valores nulos no carregamento bruto. A análise por IQR indicou 1876 ocorrências potenciais de outliers, mantidas por poderem representar compostos quimicamente plausíveis.

## 3. Pré-processamento

O pré-processamento separou atributos e rótulo, codificou a classe por `LabelEncoder`, converteu atributos para valores numéricos, removeu duplicados e imputou eventuais ausências pela mediana. A divisão adotada foi estratificada, com proporções aproximadas de 60% treino, 20% validação e 20% teste, usando `random_state=42`.

A padronização por `StandardScaler` foi aplicada nos modelos que dependem de magnitude. No KNN, atributos em escalas maiores dominam o cálculo de distância. Na MLP, escalas incompatíveis dificultam a otimização por gradiente e podem atrasar convergência.

## 4. Modelos Supervisionados

Foram avaliados KNN, Árvore de Decisão e uma Rede Neural Artificial do tipo MLP. O KNN testou diferentes valores de vizinhos, métricas de distância e pesos, incluindo comparação explícita com e sem padronização. A árvore testou critérios Gini/Entropy, profundidades máximas, folhas mínimas e `ccp_alpha`, permitindo controle de complexidade e poda. A MLP testou arquiteturas com uma ou duas camadas ocultas, funções `relu` e `tanh`, taxas de aprendizado e regularização L2.

Na MLP, a função de ativação introduz não linearidade. O `early_stopping` foi usado para reduzir overfitting, interrompendo o treino quando a validação interna deixa de melhorar. A convergência depende de escala, taxa de aprendizado e complexidade da arquitetura.

- **KNN_padronizado**: melhores hiperparâmetros `{'model__metric': 'euclidean', 'model__n_neighbors': 11, 'model__weights': 'distance'}`, F1 teste=0.784, tempo de treino=1.90s.
- **KNN_sem_padronizacao**: melhores hiperparâmetros `{'metric': 'manhattan', 'n_neighbors': 5, 'weights': 'distance'}`, F1 teste=0.755, tempo de treino=0.16s.
- **Arvore_Decisao**: melhores hiperparâmetros `{'ccp_alpha': 0.005, 'criterion': 'gini', 'max_depth': 5, 'min_samples_leaf': 1}`, F1 teste=0.760, tempo de treino=1.00s.
- **MLP_RNA**: melhores hiperparâmetros `{'model__activation': 'tanh', 'model__alpha': 0.0001, 'model__hidden_layer_sizes': (32, 16), 'model__learning_rate_init': 0.01}`, F1 teste=0.738, tempo de treino=1.21s.

Tabela resumida:

| modelo               |   accuracy |   precision |   recall |       f1 |   tempo_treino_s |   roc_auc |
|:---------------------|-----------:|------------:|---------:|---------:|-----------------:|----------:|
| KNN_padronizado      |   0.843602 |    0.731707 | 0.84507  | 0.784314 |         1.90455  |  0.91826  |
| Arvore_Decisao       |   0.829384 |    0.721519 | 0.802817 | 0.76     |         1.00356  |  0.875201 |
| KNN_sem_padronizacao |   0.824645 |    0.7125   | 0.802817 | 0.754967 |         0.155455 |  0.900252 |
| MLP_RNA              |   0.815166 |    0.705128 | 0.774648 | 0.738255 |         1.2051   |  0.884708 |

## 5. Aprendizagem Não Supervisionada

O K-Means foi executado ignorando os rótulos. Foram testados K=[2, 3, 4, 5, 6, 7, 8, 9], avaliando inércia pelo método do cotovelo e `silhouette score` como medida de separação geométrica. O melhor K por silhouette foi 4, com silhouette=0.202.

A comparação entre clusters e classes reais foi salva em `results/metrics/kmeans_cluster_vs_class.csv`. O ARI foi 0.003 e o NMI foi 0.132. Esses indicadores quantificam quanto a estrutura não supervisionada coincide com a rotulagem conhecida.

## 6. Comparação de Resultados

O melhor desempenho supervisionado foi obtido por KNN_padronizado, com F1=0.784 e accuracy=0.844. O menor F1 pertenceu a MLP_RNA (0.738), indicando diferença prática entre as hipóteses aprendidas. No K-Means, o melhor K por silhouette foi 4 com silhouette=0.202; valores próximos de zero sugerem separação fraca, enquanto valores mais altos indicam estrutura geométrica mais coerente.

Em termos metodológicos, o KNN é simples e competitivo quando a geometria dos dados favorece vizinhança local, mas tem custo de predição maior e depende fortemente de escala. A Árvore de Decisão é mais interpretável e pouco sensível à normalização, porém tende a overfitting quando cresce sem restrição. A MLP possui maior flexibilidade funcional, mas exige mais cuidado com padronização, regularização, épocas e convergência.

A comparação entre KNN padronizado e não padronizado evidencia o impacto do pré-processamento. Quando há diferença relevante, ela confirma que a escala dos atributos influencia diretamente algoritmos baseados em distância. A árvore atua como contraponto interpretável porque divide atributos por limiares e não por distância euclidiana.

## 7. Conclusão

O melhor modelo neste experimento foi **KNN_padronizado**, com F1=0.784 no teste. A principal limitação é que o estudo depende de uma única partição treino/validação/teste; como melhoria futura, recomenda-se validação cruzada aninhada, seleção de atributos, análise química dos descritores e calibração probabilística.

Os resultados do K-Means ajudam a discutir se a separação supervisionada decorre de grupos naturalmente bem definidos. Caso ARI/NMI sejam baixos, a classificação depende de fronteiras mais complexas do que simples agrupamentos globulares.

## 8. Uso de IA Generativa

IA generativa foi utilizada como apoio para estruturar o código, organizar o relatório e redigir interpretações técnicas iniciais. O grupo declara compreender o conteúdo gerado, incluindo decisões de pré-processamento, funcionamento dos algoritmos, métricas calculadas e limitações metodológicas.

## Artefatos Gerados

- Figuras: `results/figures/`
- Métricas: `results/metrics/`
- Modelos: `results/models/`
- Variância explicada PCA inicial: [0.1804865803310537, 0.122603652816729]
