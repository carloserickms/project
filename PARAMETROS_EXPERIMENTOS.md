# Guia de parâmetros para experimentos

Este arquivo resume os pontos do projeto que podem ser alterados depois para testar resultados diferentes. Serve como mapa de consulta para futuros experimentos.

**Nota (protocolo T2):** o K-Means em `src/unsupervised.py` usa apenas o conjunto de **treino**, com `scaler.transform` (sem refit no dataset completo), alinhado ao protocolo treino/validação/teste.

O projeto usa principalmente `pandas`, `numpy`, `matplotlib`, `seaborn`, `joblib` e `scikit-learn`. A maior parte dos parâmetros que mudam resultados está em `scikit-learn`, especialmente em divisão dos dados, padronização, busca de hiperparâmetros, KNN, Árvore de Decisão, MLP/RNA, K-Means e PCA.

## Fluxo geral do pipeline

O arquivo `main.py` executa esta sequência:

1. Carrega `biodeg.csv`.
2. Infere o perfil do dataset.
3. Limpa dados e remove duplicatas.
4. Gera gráficos e métricas de EDA.
5. Divide os dados em treino, validação e teste.
6. Treina modelos supervisionados.
7. Avalia modelos no teste.
8. Executa K-Means.
9. Gera relatório e salva artefatos.

Os arquivos mais importantes para alterar experimentos são:

- `src/preprocessing.py`: limpeza, `LabelEncoder`, divisão treino/validação/teste e `StandardScaler`.
- `src/supervised.py`: `GridSearchCV`, KNN, Árvore de Decisão e MLP.
- `src/unsupervised.py`: K-Means, intervalo de K, métricas de clusterização e PCA.
- `src/evaluation.py`: métricas de avaliação, matriz de confusão e curva ROC.
- `src/visualization.py`: gráficos exploratórios, histogramas, boxplots, heatmap e PCA inicial.
- `src/utils.py`: semente global `RANDOM_STATE`.

## Parâmetros globais

### `RANDOM_STATE`

Local: `src/utils.py`

Valor atual:

```python
RANDOM_STATE = 42
```

Controla a aleatoriedade em divisões de dados, Árvore de Decisão, MLP e K-Means. Alterar esse valor pode mudar:

- quais amostras caem em treino, validação e teste;
- inicialização da MLP;
- desempates ou escolhas internas da Árvore;
- inicialização dos centróides do K-Means.

Sugestões de teste:

```python
RANDOM_STATE = 0
RANDOM_STATE = 7
RANDOM_STATE = 123
RANDOM_STATE = 2026
```

Se o desempenho variar muito ao trocar a semente, o resultado depende bastante da partição dos dados. Nesse caso, o ideal seria usar validação cruzada repetida.

## Leitura do dataset

### `pd.read_csv`

Local: `src/data_loader.py`

Valor atual:

```python
pd.read_csv(path, sep=";", header=None)
```

Parâmetros relevantes:

- `sep`: separador das colunas. Atual: `";"`.
- `header`: indica se o arquivo tem cabeçalho. Atual: `None`, ou seja, sem cabeçalho.
- `encoding`: pode ser necessário se o CSV tiver caracteres especiais.
- `decimal`: útil quando números usam vírgula decimal, por exemplo `decimal=","`.

O projeto assume que a última coluna é o alvo (`target`) e todas as anteriores são atributos (`feature_01`, `feature_02`, etc.).

## Limpeza e pré-processamento

### Conversão numérica

Local: `src/preprocessing.py`, função `clean_dataset`.

Valor atual:

```python
pd.to_numeric(cleaned[col], errors="coerce")
```

O que faz:

- tenta converter cada feature para número;
- valores inválidos viram `NaN`;
- depois os `NaN` são preenchidos pela mediana.

Parâmetros/alternativas:

- `errors="coerce"`: transforma valores inválidos em `NaN`.
- `errors="raise"`: gera erro se encontrar valor inválido.
- `errors="ignore"`: manteria valores originais, mas não é adequado para modelos numéricos.

### Imputação de valores ausentes

Valor atual:

```python
fillna(cleaned.iloc[:, :-1].median())
```

O que faz:

- preenche valores ausentes com a mediana de cada coluna.

Alternativas para testar:

- média: mais sensível a outliers;
- mediana: mais robusta a outliers;
- valor fixo, como `0`;
- `SimpleImputer` do `scikit-learn`, com estratégias `mean`, `median`, `most_frequent` ou `constant`.

### Remoção de duplicatas

Valor atual:

```python
drop_duplicates()
```

O que faz:

- remove linhas repetidas.

Impacto:

- pode reduzir viés causado por amostras duplicadas;
- também pode remover repetições reais, caso elas tenham significado no dataset.

## Divisão treino/validação/teste

Local: `src/preprocessing.py`, função `prepare_splits`.

Valores atuais:

```python
test_size=0.20
validation_size=0.20
stratify=y
random_state=RANDOM_STATE
```

Com esses valores, o projeto usa aproximadamente:

- 60% treino;
- 20% validação;
- 20% teste.

### `test_size`

Define a proporção separada para teste final.

Sugestões:

```python
test_size=0.15
test_size=0.20
test_size=0.25
test_size=0.30
```

Efeito esperado:

- teste maior: avaliação final mais estável, mas menos dados para treinar;
- teste menor: mais dados para treinar, mas avaliação final mais sensível à sorte da divisão.

### `validation_size`

Define a proporção total desejada para validação.

Sugestões:

```python
validation_size=0.10
validation_size=0.15
validation_size=0.20
validation_size=0.25
```

Efeito esperado:

- validação maior: escolha de hiperparâmetros mais confiável;
- validação menor: mais dados para treino, mas seleção menos estável.

### `stratify`

Valor atual:

```python
stratify=y
```

O que faz:

- preserva a proporção das classes em treino, validação e teste.

Para datasets de classificação, normalmente deve ficar ligado. Remover `stratify` pode criar divisões desbalanceadas, principalmente em bases pequenas.

## Padronização

Local: `src/preprocessing.py` e pipelines em `src/supervised.py`.

Valor atual:

```python
StandardScaler()
```

O que faz:

- transforma cada atributo para média próxima de 0 e desvio padrão próximo de 1.

Impacto:

- muito importante para KNN, porque KNN usa distância;
- muito importante para MLP, porque redes neurais otimizam pesos por gradiente;
- pouco importante para Árvore de Decisão, porque árvore usa limiares de corte.

Alternativas do `scikit-learn`:

- `StandardScaler`: bom padrão para muitos modelos.
- `MinMaxScaler`: coloca valores em intervalo, geralmente `[0, 1]`.
- `RobustScaler`: usa mediana e intervalo interquartil; melhor quando há muitos outliers.
- `Normalizer`: normaliza cada linha/amostra, mais comum em texto ou vetores de direção.

Teste útil:

- comparar KNN com `StandardScaler`, `MinMaxScaler` e `RobustScaler`;
- comparar MLP com `StandardScaler` e `MinMaxScaler`.

## Busca de hiperparâmetros com `GridSearchCV`

Local: `src/supervised.py`, função `_fit_grid`.

Valor atual:

```python
GridSearchCV(estimator, params, scoring="f1_weighted", cv=5, n_jobs=-1)
```

### `estimator`

É o modelo ou pipeline que será treinado. No projeto:

- KNN com padronização;
- KNN sem padronização;
- Árvore de Decisão;
- MLP/RNA com padronização.

### `params`

É a grade de parâmetros testados. O `GridSearchCV` treina todas as combinações possíveis.

Exemplo:

```python
{
    "model__n_neighbors": [3, 5, 7],
    "model__weights": ["uniform", "distance"]
}
```

Nesse caso, ele testa 3 x 2 = 6 combinações.

### `scoring`

Valor atual:

```python
scoring="f1_weighted"
```

O que faz:

- escolhe os melhores hiperparâmetros usando F1 ponderado.

Alternativas comuns:

- `"accuracy"`: proporção de acertos.
- `"precision"` ou `"precision_weighted"`: foca em reduzir falsos positivos.
- `"recall"` ou `"recall_weighted"`: foca em reduzir falsos negativos.
- `"f1"` ou `"f1_weighted"`: equilíbrio entre precision e recall.
- `"roc_auc"`: bom para classificação binária com probabilidades.

Para classe desbalanceada, `f1_weighted`, `balanced_accuracy`, `recall_weighted` ou `roc_auc` podem ser mais informativos que `accuracy`.

### `cv`

Valor atual:

```python
cv=5
```

O que faz:

- divide o treino em 5 partes internas para validação cruzada.

Sugestões:

```python
cv=3
cv=5
cv=10
```

Efeito esperado:

- `cv=3`: mais rápido, menos estável;
- `cv=5`: bom equilíbrio;
- `cv=10`: mais caro, geralmente mais estável.

### `n_jobs`

Valor atual:

```python
n_jobs=-1
```

O que faz:

- usa todos os núcleos disponíveis para paralelizar a busca.

Alternativas:

- `n_jobs=1`: executa em um único processo, mais lento mas mais simples de depurar;
- `n_jobs=2`, `n_jobs=4`: limita uso de CPU.

## KNN com e sem padronização

Local: `src/supervised.py`, função `train_supervised_models`.

Modelos atuais:

```python
KNN_padronizado
KNN_sem_padronizacao
```

O KNN classifica uma amostra olhando os vizinhos mais próximos no espaço de atributos.

### `n_neighbors`

Valores atuais:

```python
[3, 5, 7, 9, 11, 15]
```

O que faz:

- define quantos vizinhos votam na classe final.

Efeito esperado:

- valores pequenos: modelo mais sensível, pode capturar padrões locais, mas pode overfitar;
- valores grandes: modelo mais suave, pode generalizar melhor, mas pode perder detalhes.

Sugestões:

```python
[1, 3, 5, 7, 9, 11, 15, 21, 31]
```

### `weights`

Valores atuais:

```python
["uniform", "distance"]
```

Opções:

- `"uniform"`: todos os vizinhos têm o mesmo peso.
- `"distance"`: vizinhos mais próximos têm mais peso.

Efeito esperado:

- `"distance"` pode ajudar quando vizinhos muito próximos são mais confiáveis;
- `"uniform"` pode ser mais estável quando há ruído.

### `metric`

Valores atuais:

```python
["euclidean", "manhattan"]
```

Opções úteis:

- `"euclidean"`: distância reta tradicional.
- `"manhattan"`: soma das diferenças absolutas.
- `"minkowski"`: generalização controlada por `p`.
- `"chebyshev"`: considera a maior diferença entre atributos.

Sugestão de grade:

```python
{
    "model__metric": ["euclidean", "manhattan", "minkowski"],
    "model__p": [1, 2, 3]
}
```

Observação: quando o KNN está dentro de `Pipeline`, os parâmetros precisam do prefixo `model__`. Sem pipeline, usam o nome direto, como `n_neighbors`.

### Outros parâmetros que a biblioteca oferece

`KNeighborsClassifier` também permite:

- `algorithm`: `"auto"`, `"ball_tree"`, `"kd_tree"`, `"brute"`.
- `leaf_size`: afeta desempenho de busca em árvores.
- `p`: expoente da métrica Minkowski.
- `n_jobs`: paralelismo na busca de vizinhos.

## Árvore de Decisão

Local: `src/supervised.py`.

Modelo atual:

```python
DecisionTreeClassifier(random_state=RANDOM_STATE)
```

Árvores dividem os dados por perguntas do tipo "atributo X <= valor Y". São interpretáveis, mas podem overfitar se crescerem demais.

### `criterion`

Valores atuais:

```python
["gini", "entropy"]
```

Opções:

- `"gini"`: mede impureza usando índice Gini.
- `"entropy"`: usa ganho de informação baseado em entropia.
- `"log_loss"`: alternativa moderna para classificação probabilística.

Na prática, `gini` e `entropy` costumam ter resultados parecidos, mas vale testar.

### `max_depth`

Valores atuais:

```python
[3, 5, 8, 12, None]
```

O que faz:

- limita a profundidade máxima da árvore.

Efeito esperado:

- menor profundidade: modelo mais simples, menor risco de overfitting;
- maior profundidade ou `None`: modelo mais flexível, maior risco de memorizar treino.

Sugestões:

```python
[2, 3, 4, 5, 8, 10, 12, 15, None]
```

### `min_samples_leaf`

Valores atuais:

```python
[1, 3, 5, 10]
```

O que faz:

- define o mínimo de amostras permitido em cada folha.

Efeito esperado:

- `1`: árvore pode criar folhas muito específicas;
- valores maiores: árvore fica mais conservadora e menos propensa a overfitting.

Sugestões:

```python
[1, 2, 3, 5, 10, 20]
```

### `ccp_alpha`

Valores atuais:

```python
[0.0, 0.001, 0.005, 0.01]
```

O que faz:

- controla poda por complexidade de custo.

Efeito esperado:

- `0.0`: sem poda;
- valores maiores: árvore menor e mais simples;
- valor alto demais: árvore pode ficar simples demais e perder desempenho.

Sugestões:

```python
[0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
```

### Outros parâmetros que a biblioteca oferece

`DecisionTreeClassifier` também permite:

- `splitter`: `"best"` ou `"random"`.
- `min_samples_split`: mínimo de amostras para dividir um nó.
- `max_features`: quantidade de atributos considerada em cada divisão.
- `class_weight`: pode usar `"balanced"` em dados desbalanceados.
- `max_leaf_nodes`: limita o número máximo de folhas.
- `min_impurity_decrease`: exige ganho mínimo para criar divisão.

Grade recomendada para testar overfitting:

```python
{
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [3, 5, 8, 12, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 3, 5, 10],
    "class_weight": [None, "balanced"],
    "ccp_alpha": [0.0, 0.001, 0.005, 0.01]
}
```

## MLP/RNA

Local: `src/supervised.py`.

Modelo atual:

```python
MLPClassifier(
    random_state=RANDOM_STATE,
    early_stopping=True,
    max_iter=700
)
```

A MLP é uma rede neural feedforward. Ela é mais flexível que KNN e árvore, mas costuma ser mais sensível a escala, taxa de aprendizado, regularização e número de épocas.

### `hidden_layer_sizes`

Valores atuais:

```python
[(16,), (32,), (32, 16), (64, 32)]
```

O que faz:

- define quantidade de camadas ocultas e neurônios por camada.

Exemplos:

- `(16,)`: uma camada oculta com 16 neurônios.
- `(32, 16)`: duas camadas ocultas, primeira com 32 e segunda com 16.
- `(64, 32, 16)`: três camadas ocultas.

Efeito esperado:

- redes pequenas: mais rápidas, menor risco de overfitting;
- redes grandes: maior capacidade, mais risco de overfitting e treino mais lento.

Sugestões:

```python
[(8,), (16,), (32,), (64,), (32, 16), (64, 32), (64, 32, 16)]
```

### `activation`

Valores atuais:

```python
["relu", "tanh"]
```

Opções:

- `"relu"`: padrão moderno, costuma treinar bem.
- `"tanh"`: pode funcionar bem com dados padronizados.
- `"logistic"`: sigmoid; pode saturar mais facilmente.
- `"identity"`: sem não linearidade; vira um modelo mais simples.

Sugestão:

```python
["relu", "tanh", "logistic"]
```

### `learning_rate_init`

Valores atuais:

```python
[0.001, 0.01]
```

O que faz:

- controla o tamanho inicial dos passos do otimizador.

Efeito esperado:

- valor baixo: treino mais lento, mas mais estável;
- valor alto: treino mais rápido, mas pode oscilar ou não convergir.

Sugestões:

```python
[0.0001, 0.0005, 0.001, 0.005, 0.01]
```

### `alpha`

Valores atuais:

```python
[0.0001, 0.001]
```

O que faz:

- regularização L2 dos pesos.

Efeito esperado:

- valores baixos: modelo mais livre, maior risco de overfitting;
- valores altos: pesos menores, modelo mais regularizado, possível underfitting.

Sugestões:

```python
[0.00001, 0.0001, 0.001, 0.01, 0.1]
```

### `early_stopping`

Valor atual:

```python
early_stopping=True
```

O que faz:

- separa internamente uma parte do treino para validação;
- interrompe o treino se a pontuação não melhorar.

Parâmetros relacionados:

- `validation_fraction`: fração do treino usada internamente para early stopping. Padrão do scikit-learn: `0.1`.
- `n_iter_no_change`: quantas épocas sem melhora antes de parar. Padrão: `10`.
- `tol`: melhora mínima considerada relevante.

### `max_iter`

Valor atual:

```python
max_iter=700
```

O que faz:

- define o número máximo de iterações/épocas.

Efeito esperado:

- baixo demais: rede pode parar antes de convergir;
- alto: permite convergir, mas aumenta tempo.

Sugestões:

```python
max_iter=300
max_iter=700
max_iter=1000
max_iter=1500
```

### Outros parâmetros que a biblioteca oferece

`MLPClassifier` também permite:

- `solver`: `"adam"`, `"sgd"`, `"lbfgs"`.
- `batch_size`: tamanho dos lotes no treino.
- `learning_rate`: `"constant"`, `"invscaling"`, `"adaptive"`; usado principalmente com `solver="sgd"`.
- `momentum`: usado com SGD.
- `beta_1`, `beta_2`, `epsilon`: parâmetros do Adam.
- `shuffle`: embaralha amostras a cada época.
- `class_weight`: o `MLPClassifier` não possui `class_weight` direto em algumas versões; para desbalanceamento, pode ser necessário usar reamostragem ou pesos via outras estratégias.

Grade recomendada para testar:

```python
{
    "model__hidden_layer_sizes": [(16,), (32,), (64,), (32, 16), (64, 32)],
    "model__activation": ["relu", "tanh"],
    "model__solver": ["adam", "sgd"],
    "model__learning_rate_init": [0.0005, 0.001, 0.005, 0.01],
    "model__alpha": [0.0001, 0.001, 0.01]
}
```

## K-Means

Local: `src/unsupervised.py`, função `run_kmeans_analysis`.

Modelo atual:

```python
KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
```

K-Means tenta separar os dados em grupos minimizando a distância entre pontos e centróides.

### `k_values`

Valor atual:

```python
list(range(2, min(11, len(set(y)) + 8)))
```

O que faz:

- testa valores de K começando em 2;
- vai até no máximo 10;
- também considera o número de classes reais mais 8.

Para um problema binário, por exemplo, tende a testar K de 2 até 9.

Sugestões:

```python
list(range(2, 11))
list(range(2, 16))
list(range(2, 21))
```

Efeito esperado:

- K baixo: clusters mais amplos;
- K alto: clusters menores, possivelmente fragmentados;
- silhouette ajuda a escolher separação geométrica melhor, mas não garante alinhamento com as classes reais.

### `n_clusters`

Valor usado em cada iteração:

```python
n_clusters=k
```

O que faz:

- define o número de clusters.

Para comparar diretamente com classes reais, pode ser interessante testar `n_clusters=2` se o problema for binário.

### `n_init`

Valor atual:

```python
n_init=20
```

O que faz:

- roda o K-Means várias vezes com inicializações diferentes e mantém a melhor solução.

Efeito esperado:

- valores maiores: resultado mais confiável, mas mais lento;
- valores menores: mais rápido, mas mais dependente da inicialização.

Sugestões:

```python
n_init=10
n_init=20
n_init=50
n_init="auto"
```

### Outros parâmetros que a biblioteca oferece

`KMeans` também permite:

- `init`: `"k-means++"` ou `"random"`.
- `max_iter`: máximo de iterações por inicialização.
- `tol`: tolerância para declarar convergência.
- `algorithm`: `"lloyd"` ou `"elkan"`.

Grade manual interessante:

```python
KMeans(
    n_clusters=k,
    init="k-means++",
    n_init=50,
    max_iter=500,
    random_state=RANDOM_STATE
)
```

## Métricas do K-Means

Local: `src/unsupervised.py`.

Métricas atuais:

- `inertia`: soma das distâncias quadradas até os centróides. Menor é melhor, mas sempre diminui quando K aumenta.
- `silhouette_score`: mede separação e coesão dos clusters. Varia de -1 a 1; maior é melhor.
- `adjusted_rand_score` ou ARI: compara clusters com classes reais. Próximo de 1 é melhor; próximo de 0 indica alinhamento parecido com acaso.
- `normalized_mutual_info_score` ou NMI: mede informação compartilhada entre clusters e classes. Varia de 0 a 1.

Interpretação prática:

- silhouette alto e ARI/NMI baixo: clusters são geométricos, mas não correspondem bem às classes reais;
- silhouette baixo e ARI/NMI baixo: K-Means provavelmente não encontrou estrutura útil;
- ARI/NMI alto: agrupamentos têm boa relação com os rótulos reais.

## PCA

Locais:

- `src/visualization.py`: PCA inicial por classe real.
- `src/unsupervised.py`: PCA para visualizar clusters do K-Means.

Valor atual:

```python
PCA(n_components=2, random_state=RANDOM_STATE)
```

### `n_components`

Valor atual:

```python
n_components=2
```

O que faz:

- reduz os atributos para 2 componentes principais para visualização.

Sugestões:

```python
n_components=2
n_components=3
n_components=0.95
```

Observações:

- `2` é bom para gráfico 2D;
- `3` permite gráfico 3D se o código de visualização for adaptado;
- `0.95` pede componentes suficientes para explicar 95% da variância, mas não serve diretamente para gráfico 2D.

### `explained_variance_ratio_`

O projeto salva a variância explicada pelo PCA. Isso indica quanta informação aproximada os componentes mantêm.

Se PC1 + PC2 explicam pouca variância, o gráfico 2D pode esconder separações importantes.

## Métricas supervisionadas

Local: `src/evaluation.py`.

Métricas atuais:

- `accuracy`: acertos totais.
- `precision`: entre as previsões positivas, quantas estavam corretas.
- `recall`: entre os positivos reais, quantos foram encontrados.
- `f1`: média harmônica entre precision e recall.
- `classification_report`: métricas por classe.
- `confusion_matrix`: matriz de confusão.
- `roc_auc`: área sob curva ROC, quando o problema é binário e o modelo tem `predict_proba`.

### `average`

Valor atual:

```python
average = "binary" if len(class_names) == 2 else "weighted"
```

O que faz:

- usa métrica binária para problema com 2 classes;
- usa média ponderada para multiclasse.

Alternativas:

- `"macro"`: média simples entre classes, dá peso igual para classes raras.
- `"weighted"`: média ponderada pelo suporte de cada classe.
- `"micro"`: agrega decisões globalmente.
- `"binary"`: usa classe positiva em problema binário.

Para classes desbalanceadas, `macro` pode revelar desempenho ruim em classe minoritária que o `weighted` esconde.

### `zero_division`

Valor atual:

```python
zero_division=0
```

O que faz:

- evita erro quando uma métrica envolve divisão por zero;
- nesses casos, retorna 0.

## Visualizações e parâmetros gráficos

Local: `src/visualization.py`, `src/evaluation.py` e `src/unsupervised.py`.

Parâmetros que não mudam o modelo, mas mudam leitura visual:

- `figsize`: tamanho das figuras.
- `dpi`: resolução ao salvar.
- `bins`: quantidade de barras nos histogramas.
- `palette` ou `cmap`: paleta de cores.
- `alpha`: transparência dos pontos.
- `s`: tamanho dos pontos no scatterplot.
- `max_depth` no `plot_tree`: profundidade exibida no gráfico da árvore, não necessariamente profundidade real do modelo.

Exemplo atual:

```python
plot_tree(..., max_depth=3, fontsize=7)
```

Isso limita apenas a visualização aos primeiros níveis. A árvore treinada pode ser mais profunda.

## Pipelines e prefixo `model__`

O projeto usa `Pipeline` em alguns modelos:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier())
])
```

Quando um estimador está dentro de um `Pipeline`, os parâmetros da grade precisam ser escritos com o nome da etapa seguido de `__`.

Exemplos:

```python
"model__n_neighbors"
"model__activation"
"model__learning_rate_init"
```

Se o modelo não está dentro de pipeline, o nome é direto:

```python
"n_neighbors"
"criterion"
"max_depth"
```

## Sugestões de experimentos futuros

### Experimento 1: impacto da divisão dos dados

Alterar em `prepare_splits`:

```python
test_size=0.15
validation_size=0.15
```

Depois testar:

```python
test_size=0.30
validation_size=0.20
```

Objetivo: verificar se o melhor modelo continua sendo o mesmo.

### Experimento 2: KNN mais amplo

Adicionar mais vizinhos:

```python
[1, 3, 5, 7, 9, 11, 15, 21, 31]
```

Objetivo: observar se o KNN melhora com modelos mais locais ou mais suaves.

### Experimento 3: Árvore com controle de overfitting

Adicionar:

```python
"min_samples_split": [2, 5, 10, 20]
"class_weight": [None, "balanced"]
```

Objetivo: verificar se uma árvore mais regularizada generaliza melhor.

### Experimento 4: MLP mais estável

Testar:

```python
"model__learning_rate_init": [0.0001, 0.0005, 0.001, 0.005]
"model__alpha": [0.0001, 0.001, 0.01]
```

Objetivo: reduzir instabilidade da rede e controlar overfitting.

### Experimento 5: K-Means com mais inicializações

Alterar:

```python
n_init=50
max_iter=500
```

Objetivo: reduzir dependência da inicialização dos centróides.

### Experimento 6: testar outro scaler

Substituir `StandardScaler` por:

```python
RobustScaler()
```

Objetivo: verificar se outliers estão prejudicando KNN, MLP ou K-Means.

## Cuidados ao comparar resultados

- Compare modelos usando o mesmo `RANDOM_STATE` quando quiser isolar o efeito dos hiperparâmetros.
- Troque `RANDOM_STATE` quando quiser testar estabilidade.
- Não escolha o melhor modelo olhando apenas o teste muitas vezes; isso vicia a avaliação final.
- Para comparação mais forte, use validação cruzada repetida ou nested cross-validation.
- Registre sempre a grade usada, a semente, o melhor parâmetro e a métrica final.

## Onde os resultados aparecem

Depois de rodar:

```bash
python main.py
```

os principais arquivos são:

- `results/metrics/supervised_comparison.csv`: comparação final dos modelos supervisionados.
- `results/metrics/supervised_results.json`: hiperparâmetros escolhidos e métricas detalhadas.
- `results/metrics/kmeans_metrics.csv`: métricas para cada K testado.
- `results/metrics/kmeans_summary.json`: resumo do melhor K-Means.
- `results/figures/`: gráficos gerados.
- `results/models/`: modelos salvos em `.joblib`.
- `report/relatorio.md`: relatório final gerado automaticamente.

