# Projeto de Aprendizagem de Máquina - Biodegradabilidade

Projeto acadêmico completo de Inteligência Artificial com análise exploratória, pré-processamento, modelos supervisionados e K-Means.

## Como executar

1. Crie um ambiente virtual, se desejar.
2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Execute o pipeline:

```bash
python main.py
```

Em alguns sistemas o comando pode ser:

```bash
python3 main.py
```

## Estrutura

- `biodeg.csv`: dataset sem cabeçalho, separado por `;`.
- `main.py`: orquestra todo o pipeline.
- `src/`: módulos de carga, pré-processamento, treino, avaliação e visualização.
- `results/figures/`: gráficos da EDA, matrizes de confusão, ROC, árvore e K-Means.
- `results/metrics/`: estatísticas, métricas, comparações e interpretações automáticas.
- `results/models/`: modelos treinados e árvore em DOT.
- `report/relatorio.md`: relatório acadêmico gerado automaticamente.

## Metodologia resumida

O código infere automaticamente a estrutura do dataset, assume a última coluna como classe e trata os atributos anteriores como preditores. A divisão treino/validação/teste é estratificada e usa `random_state=42`.

Modelos supervisionados:

- KNN com e sem padronização, múltiplos valores de `k` e métricas de distância.
- Árvore de Decisão com critérios Gini/Entropy, controle de profundidade e poda por `ccp_alpha`.
- MLPClassifier como Rede Neural Artificial, com busca sobre arquitetura, ativação, regularização e taxa de aprendizado.

Aprendizagem não supervisionada:

- K-Means com múltiplos valores de K.
- Método do cotovelo, silhouette score, PCA 2D e comparação com rótulos reais por ARI/NMI.
