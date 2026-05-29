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
