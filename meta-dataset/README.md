# Meta-dataset

Matrizes compartilhadas pelas práticas. Os 30 datasets de regressão originais ficam em [`../data/`](../data/). Em todas as matrizes, cada linha é um dataset, identificado pela coluna `dataset`.

| Arquivo | Conteúdo |
|---|---|
| [X.csv](X.csv) | Xbase: 13 meta-características dos 30 datasets |
| [X_complexidade.csv](X_complexidade.csv) | Medidas de complexidade por grupo (alvo inteiro, exemplos comuns e raros), usadas a partir da Prática 2 |
| [P.csv](P.csv) | SERA média dos seis regressores, sem reamostragem |
| [R.csv](R.csv) | Ranks derivados de P (1 é o melhor regressor) |
| [P_resampling.csv](P_resampling.csv) | SERA média com reamostragem (WERCS) |
| [R_resampling.csv](R_resampling.csv) | Ranks derivados de P_resampling |
| [resultados_por_fold.csv](resultados_por_fold.csv) | SERA por dataset, regressor e fold, de onde saem P e R |
| [resultados_por_fold_resampling.csv](resultados_por_fold_resampling.csv) | O mesmo, com reamostragem |

Xestendido não tem cópia aqui. Ele junta os 13 descritores de Xbase, 36 medidas de `X_complexidade.csv` e T2 dos exemplos comuns e dos raros, num total de 51 colunas. A versão usada está em [`../pratica-03/resultados/Xestendido_usado.csv`](../pratica-03/resultados/).
