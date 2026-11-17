"""Recomendação da MtL (Abordagem 2 da Prática 1) sem acesso ao teste do fold externo.

Para o dataset alvo d e o fold f:
  * o meta-modelo (RF sobre ranks) é treinado com os outros 29 datasets
    (Xbase e ranks R calculados offline; o alvo fica de fora: leave-one-dataset-out);
  * o alvo é descrito pelo Xbase calculado só em X[treino_f], y[treino_f]
    (resultados/meta_por_fold.jsonl).
"""
from __future__ import annotations

import sys
import time

import pandas as pd

import comum as c

sys.path.insert(0, str(c.BASE / "codigo"))
from abordagens_meta_modelo import abordagem2_regressor_sobre_R  # noqa: E402

COLUNAS_XBASE = pd.read_csv(c.META / "X.csv", index_col=0).columns.tolist()


def recomendar(Xbase: pd.DataFrame, R: pd.DataFrame, alvo: str, descritor_alvo: pd.Series) -> str:
    """Algoritmo top-1 para `alvo`; nenhuma linha do alvo entra no treino do meta-modelo."""
    X_alvo = pd.DataFrame([descritor_alvo[COLUNAS_XBASE].astype(float)], index=[alvo])
    ranking = abordagem2_regressor_sobre_R(Xbase.drop(index=alvo), R.drop(index=alvo), X_alvo).loc[alvo]
    return ranking.idxmin()


def recomendacoes_por_fold(Xbase: pd.DataFrame, R: pd.DataFrame, meta_fold: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for (alvo, fold), desc in meta_fold.iterrows():
        t0 = time.perf_counter()
        alg = recomendar(Xbase, R, alvo, desc)
        linhas.append(dict(dataset=alvo, fold=fold, alg=alg, segundos_meta_modelo=time.perf_counter() - t0,
                           segundos_meta_caracteristicas=desc["segundos_xbase"]))
    return pd.DataFrame(linhas)
