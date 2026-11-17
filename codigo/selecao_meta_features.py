"""Selecao de meta-caracteristicas DENTRO de cada particao de treino.

O regime e 29 linhas contra ate 49 colunas. A aula -- e o enunciado, ao avisar
que "dobrar o numero de colunas de X nao e um gesto neutro" -- diz que nesse
cenario a selecao deixa de ser opcional.

A defesa contra vazamento e estrutural, nao uma promessa no comentario: a
funcao recebe SO dados de treino. Nao existe parametro por onde o dataset de
teste entre. Foi exatamente aqui que a primeira tentativa falhou -- as oito
medidas dela foram escolhidas por uma analise de redundancia que usou os 30
datasets, incluindo o de teste de cada round.

Metodo: filtro univariado por |rho de Spearman| contra o ranking de treino, com
k fixado a priori. k = 13 iguala a dimensionalidade da base, de modo que
`base` contra `selecionado` compara o MESMO numero de colunas, colunas
diferentes -- sem isso, qualquer diferenca confundiria "colunas melhores" com
"mais colunas".

Eliminacao reversa ficou de fora: com 29 linhas e 49 colunas e instavel e cara,
e a instabilidade viraria variancia entre folds, nao sinal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

K_PADRAO = 13


def pontuar(X_train: pd.DataFrame, R_train: pd.DataFrame) -> pd.Series:
    """Forca da associacao de cada coluna com o ranking dos algoritmos.

    Maximo de |rho| sobre os seis algoritmos: uma coluna util para prever a
    posicao de UM algoritmo ja e util, e a media diluiria isso.
    """
    pontos = {}
    for coluna in X_train.columns:
        x = X_train[coluna].to_numpy(dtype=float)
        if not np.isfinite(x).all() or np.allclose(x, x[0]):
            pontos[coluna] = 0.0
            continue
        rhos = [abs(spearmanr(x, R_train[alvo]).statistic) for alvo in R_train.columns]
        rhos = [r for r in rhos if np.isfinite(r)]
        pontos[coluna] = max(rhos) if rhos else 0.0
    return pd.Series(pontos).sort_values(ascending=False)


def selecionar_filtro(
    X_train: pd.DataFrame, R_train: pd.DataFrame, k: int = K_PADRAO
) -> list[str]:
    """As k colunas mais associadas ao meta-alvo, vistas so no treino.

    Empates e colunas constantes resolvem-se pela ordem de `X_train`, que e
    determinista, para que a selecao seja reprodutivel.
    """
    if k >= X_train.shape[1]:
        return list(X_train.columns)
    return list(pontuar(X_train, R_train).head(k).index)


def estabilidade(selecoes: pd.DataFrame) -> pd.DataFrame:
    """Quantas vezes cada coluna foi escolhida ao longo dos folds.

    Se os conjuntos escolhidos forem quase disjuntos entre folds, isso e um
    resultado por si: nao ha sinal estavel para selecionar, e essa e a
    explicacao mais provavel para um efeito nulo na recomendacao.
    """
    contagem = selecoes["coluna"].value_counts()
    n_folds = selecoes["dataset_teste"].nunique()
    return pd.DataFrame({
        "vezes": contagem,
        "frequencia": contagem / n_folds,
    }).sort_values("vezes", ascending=False)
