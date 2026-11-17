"""DESReg com cache: o pool é ajustado uma vez e só a etapa de seleção muda.

A DESRegression original (DESReg 0.1.1) refaz o kNN sobre o DSEL e chama o
predict de cada regressor do pool a cada instância de teste. Para avaliar
centenas de configurações isso é inviável. Aqui a DESRegression continua
responsável pelo fit (bagging do pool e sorteio do DSEL); depois guardamos:

  - os k_max vizinhos de cada instância de teste no DSEL;
  - as predições do pool no DSEL e no teste (matrizes n x m).

Com isso, cada configuração de seleção vira só álgebra de arrays. A semântica
segue a da DESReg linha a linha, inclusive duas peculiaridades dela:

  1. em all_errors, o erro quadrático é ponderado pelas distâncias BRUTAS
     (sum_squared_error(y_true, y_pred, dists)), e não pelas normalizadas;
  2. os erros são escalados por MinMax ENTRE regressores, por instância, e a
     competência é a média das medidas escaladas (menor = mais competente).

testes/test_desreg_cache.py confere a equivalência com DESRegression.predict.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
from desReg.des.DESRegression import DESRegression
from desReg.utils import measures as em
from sklearn.neighbors import NearestNeighbors

import comum  # noqa: F401  (coloca a lib/ das práticas anteriores no path)
from phi import phi
from phi_ctrl_pts import phi_ctrl_pts

SELECAO = {
    "mean": np.mean,
    "median": np.median,
    "p25": partial(np.percentile, q=25),
}
AGREGACAO = {
    "mean": np.mean,
    "weighted": np.average,
    "median": np.median,
}
COMPETENCIAS = ("all_errors", "sum_squared_error", "sum_absolute_error",
                "closest_squared_error", "phi_sse")


# ---------------------------------------------------------------- relevância

def parametros_phi(y_treino) -> dict:
    """Pontos de controle de phi estimados só no alvo de treino do fold."""
    return phi_ctrl_pts(pd.Series(np.asarray(y_treino, dtype=float)))


def relevancia(y, params_phi: dict) -> np.ndarray:
    return np.asarray(phi(pd.Series(np.asarray(y, dtype=float)), params_phi), dtype=float)


def fazer_phi_sse(params_phi: dict):
    """Erro quadrático local ponderado pela distância e por (1 + phi(y)).

    O peso 1 + phi mantém os vizinhos normais na conta (com phi = 0 a medida
    zeraria para todos os regressores) e dobra, no máximo, o peso dos raros.
    Os nomes dos argumentos importam: a DESReg os injeta pelo nome.
    """
    def phi_sse(y_true, y_pred, normalized_distances):
        peso = 1.0 + relevancia(y_true, params_phi)
        return np.dot(peso * np.square(y_true - y_pred), normalized_distances)
    return phi_sse


def medida_desreg(nome: str, params_phi: dict | None = None):
    """A função de competência no formato que a DESRegression original espera."""
    if nome == "phi_sse":
        return fazer_phi_sse(params_phi)
    return getattr(em, nome)


# ------------------------------------------------------------------ o cache

class DESRegCache:
    """Guarda pool, DSEL e vizinhança de um fold; prevê qualquer configuração de seleção."""

    def __init__(self, des: DESRegression, X_teste, k_max: int, params_phi: dict):
        self.des = des
        P_dsel = des._predict_base(des.DSEL_data_)
        self.P_teste = des._predict_base(np.asarray(X_teste))
        self.y_dsel = np.asarray(des.DSEL_target_, dtype=float)

        k_max = min(k_max, len(self.y_dsel))
        nn = NearestNeighbors(n_neighbors=k_max, metric="euclidean").fit(des.DSEL_data_)
        dists, idxs = nn.kneighbors(np.asarray(X_teste))
        self.dists = np.where(dists == 0, 1e-10, dists)  # mesmo truque da DESReg
        self.idxs = idxs

        # (n_teste, k_max) e (n_teste, k_max, m): tudo o que a seleção precisa
        self.y_viz = self.y_dsel[idxs]
        self.P_viz = P_dsel[idxs]
        self.phi_viz = relevancia(self.y_dsel, params_phi)[idxs]

    # medidas de erro locais, uma matriz (n_teste, m) por medida
    def _medidas(self, nome: str, k: int) -> list[np.ndarray]:
        d = self.dists[:, :k]
        nd = (1.0 / d) / (1.0 / d).sum(axis=1, keepdims=True)
        yt = self.y_viz[:, :k, None]
        yp = self.P_viz[:, :k, :]
        erro2 = np.square(yt - yp)

        sse = np.einsum("nkm,nk->nm", erro2, nd)
        mais_proximo = erro2[:, 0, :]  # vizinhos já vêm ordenados por distância
        if nome == "sum_squared_error":
            return [sse]
        if nome == "sum_absolute_error":
            return [np.einsum("nkm,nk->nm", np.abs(yt - yp), nd)]
        if nome == "closest_squared_error":
            return [mais_proximo]
        if nome == "phi_sse":
            peso = 1.0 + self.phi_viz[:, :k]
            return [np.einsum("nkm,nk->nm", erro2, nd * peso)]
        if nome == "all_errors":
            pond = erro2 * nd[:, :, None]
            return [
                yp.var(axis=1),
                np.einsum("nkm,nk->nm", np.abs(yt - yp), nd),
                np.einsum("nkm,nk->nm", erro2, d),  # distâncias brutas, como na DESReg
                pond.min(axis=1),
                pond.max(axis=1),
                np.einsum("nkm,nk->nm", np.square(yt - self.P_teste[:, None, :]), nd),
                np.sqrt(sse),
                mais_proximo,
            ]
        raise ValueError(f"medida de competência desconhecida: {nome}")

    def competencia(self, nome: str, k: int) -> np.ndarray:
        """Nível de competência (n_teste, m): média das medidas escaladas por MinMax."""
        escaladas = []
        for v in self._medidas(nome, k):
            # mesma aritmética do MinMaxScaler (x * escala + deslocamento), inclusive
            # a regra de que faixa < 10*eps conta como constante: erros que deveriam
            # ser zero e saem 1e-30 não podem virar diferença de competência
            vmin = v.min(axis=1, keepdims=True)
            faixa = v.max(axis=1, keepdims=True) - vmin
            escala = 1.0 / np.where(faixa < 10 * np.finfo(float).eps, 1.0, faixa)
            escaladas.append(v * escala + (0.0 - vmin * escala))
        return np.mean(np.stack(escaladas, axis=2), axis=2)

    def prever(self, ensemble_type: str, k: int, competence_level: str,
               regressor_selection: str = "mean", aggregation_method: str = "mean") -> np.ndarray:
        c = self.competencia(competence_level, k)
        linhas = np.arange(len(c))

        if ensemble_type == "DRS":
            return self.P_teste[linhas, np.argmin(c, axis=1)]  # argmin = o primeiro empatado

        limiar = {
            "mean": lambda a: a.mean(axis=1),
            "median": lambda a: np.median(a, axis=1),
            "p25": lambda a: np.percentile(a, 25, axis=1),
        }[regressor_selection](c)
        sel = c <= limiar[:, None]

        if aggregation_method == "mean":
            return (self.P_teste * sel).sum(axis=1) / sel.sum(axis=1)
        if aggregation_method == "weighted":
            w = sel / np.where(c == 0, 1e-10, c)
            return (self.P_teste * w).sum(axis=1) / w.sum(axis=1)
        if aggregation_method == "median":
            return np.nanmedian(np.where(sel, self.P_teste, np.nan), axis=1)
        raise ValueError(f"agregação desconhecida: {aggregation_method}")
