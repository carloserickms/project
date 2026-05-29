# Projeto de Aprendizagem de Máquina - Biodegradabilidade

Projeto acadêmico (T2) com análise exploratória, pré-processamento, modelos supervisionados (KNN, árvore, MLP) e K-Means.

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

Em alguns sistemas o comando pode ser `python3 main.py`.

Isso regenera `results/figures/`, `results/metrics/`, `results/models/` e `report/relatorio.md`.

## Estrutura

| Caminho | Descrição |
|---------|-----------|
| `biodeg.csv` | Dataset (sem cabeçalho, separador `;`) |
| `main.py` | Orquestra o pipeline |
| `src/` | Módulos de carga, pré-processamento, treino, avaliação e visualização |
| `docs/` | Enunciado, critérios, descrição do dataset, apêndice GenAI |
| `results/figures/` | Gráficos (gerados ao executar) |
| `results/metrics/` | CSV/JSON de métricas |
| `results/models/` | Modelos `.joblib` e árvore em DOT |
| `report/relatorio.md` | Relatório acadêmico (seções a–i do enunciado) |
| `PARAMETROS_EXPERIMENTOS.md` | Guia de hiperparâmetros ajustáveis |

## Metodologia resumida

- Última coluna = classe (RB/NRB); atributos anteriores = descritores QSAR.
- Divisão estratificada treino/validação/teste (~60/20/20), `random_state=42`.
- `GridSearchCV` (5-fold) **somente no treino**; validação para monitoramento; **teste avaliado uma vez**.
- K-Means no **treino**, com scaler calibrado no treino (sem vazamento do teste).

## Checklist de entrega (grupo)

- [ ] Código-fonte (este repositório) com instruções de execução
- [ ] Relatório em **PDF** (`report/relatorio.pdf`)
- [ ] Slides da apresentação oral
- [ ] Apresentação (até 15 min) + arguição
- [ ] Documento de participação crítica (Teams), quando aplicável

### Gerar PDF do relatório

Com [Pandoc](https://pandoc.org/) instalado:

```bash
pandoc report/relatorio.md -o report/relatorio.pdf --resource-path=.:report
```

Revise antes da entrega: capa (nomes do grupo, data) e redação final em `report/relatorio.md`.

## Documentação complementar

- [docs/dataset_biodeg.md](docs/dataset_biodeg.md) — origem e significado do dataset


