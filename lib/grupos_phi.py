"""Separacao dos grupos por relevancia e as replicas de subamostragem.

A funcao de relevancia phi(y) e a do Projeto 1 (spline monotona de Ribeiro e
Moniz, pontos de controle por boxplot de Tukey, ambas as caudas). Raro e
phi(y) >= LIMIAR_RELEVANCIA, o mesmo limiar que produz `pct_relevante` em
`X.csv` -- e a coerencia entre os dois e verificada por teste.

A ordem das operacoes e fixa e nao negociavel:

    phi no y BRUTO  ->  subamostragem  ->  min-max no conjunto (sub)amostrado
    inteiro  ->  divisao em mascaras.

Normalizar cada grupo a parte poria os tres em escalas diferentes e a
comparacao entre eles perderia sentido (ficha 4.7).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from phi import phi
from phi_ctrl_pts import phi_ctrl_pts

LIMIAR_RELEVANCIA = 0.8
MAX_INSTANCIAS = 1500
N_REPLICAS = 5
SEED = 42

GRUPOS = ("all", "normal", "raro")


def carregar_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Le um CSV de `data/`: alvo na primeira coluna, preditores no resto."""
    df = pd.read_csv(csv_path)
    y = df.iloc[:, 0].to_numpy(dtype=float)
    Z = df.iloc[:, 1:].to_numpy(dtype=float)
    return Z, y


def relevancia(y: np.ndarray) -> np.ndarray:
    """phi(y) no alvo BRUTO, antes de qualquer normalizacao."""
    serie = pd.Series(y)
    return np.asarray(phi(serie, phi_ctrl_pts(y)), dtype=float)


def mascara_raro(y: np.ndarray, limiar: float = LIMIAR_RELEVANCIA) -> np.ndarray:
    return relevancia(y) >= limiar


def mascaras_dos_grupos(raro: np.ndarray) -> dict[str, np.ndarray]:
    return {"all": np.ones_like(raro, dtype=bool), "normal": ~raro, "raro": raro}


@dataclass(frozen=True)
class Diagnostico:
    """O que precisa acompanhar cada valor para que ele seja interpretavel."""

    pct_raro: float
    n_raro: int
    n_normal: int
    raro_alta: int
    raro_baixa: int
    bimodalidade_raro: float
    gap_y_raro: float
    dispersao_relativa_raro: float

    def como_dict(self) -> dict:
        return {
            "pct_raro": self.pct_raro,
            "n_raro": self.n_raro,
            "n_normal": self.n_normal,
            "raro_alta": self.raro_alta,
            "raro_baixa": self.raro_baixa,
            "bimodalidade_raro": self.bimodalidade_raro,
            "gap_y_raro": self.gap_y_raro,
            "dispersao_relativa_raro": self.dispersao_relativa_raro,
        }


def diagnosticar(Z: np.ndarray, y: np.ndarray, raro: np.ndarray) -> Diagnostico:
    """Mede o quanto o grupo raro e bimodal e disperso.

    phi com duas caudas poe extremos altos e baixos no mesmo grupo, que fica
    disperso por construcao -- e a dispersao infla justamente S1 e S3. Sem
    estes numeros nao da para distinguir "os raros sao dificeis" de "o grupo
    raro reune dois blocos opostos de y".
    """
    mediana = float(np.median(y))
    alta = int(np.count_nonzero(raro & (y > mediana)))
    baixa = int(np.count_nonzero(raro & (y < mediana)))
    n_raro = int(raro.sum())

    bimodalidade = min(alta, baixa) / n_raro if n_raro else np.nan
    if alta and baixa:
        amplitude = y.max() - y.min()
        gap = abs(y[raro & (y > mediana)].mean() - y[raro & (y < mediana)].mean())
        gap = float(gap / amplitude) if amplitude else np.nan
    else:
        gap = 0.0

    return Diagnostico(
        pct_raro=float(raro.mean()) if len(raro) else np.nan,
        n_raro=n_raro,
        n_normal=int((~raro).sum()),
        raro_alta=alta,
        raro_baixa=baixa,
        bimodalidade_raro=bimodalidade,
        gap_y_raro=gap,
        dispersao_relativa_raro=_dispersao_relativa(Z, raro),
    )


def _dispersao_relativa(Z: np.ndarray, raro: np.ndarray, maximo: int = 400) -> float:
    """Distancia media dentro do grupo raro sobre a do dataset inteiro.

    Acima de 1 o grupo raro ocupa uma regiao mais esparsa do que o dataset --
    que e a condicao geometrica em que a MST do grupo tem arestas longas sem
    que a regiao seja mais dificil. Amostrada para nao custar O(n^2).
    """
    from scipy.spatial.distance import pdist

    n_raro = int(raro.sum())
    if n_raro < 3 or len(raro) < 3:
        return np.nan
    rng = np.random.default_rng(SEED)

    def media_dist(indices):
        if len(indices) > maximo:
            indices = rng.choice(indices, size=maximo, replace=False)
        return float(pdist(Z[indices]).mean())

    todos = media_dist(np.arange(len(raro)))
    return media_dist(np.flatnonzero(raro)) / todos if todos else np.nan


def replicas_subamostra(
    raro: np.ndarray,
    max_instancias: int = MAX_INSTANCIAS,
    n_replicas: int = N_REPLICAS,
    seed: int = SEED,
) -> list[np.ndarray]:
    """Mascaras de subamostragem estratificada pela propria condicao de raro.

    As medidas com matriz de distancia sao O(n^2) em memoria, o que forca um
    teto. Acima dele sorteiam-se `n_replicas` subconjuntos independentes em vez
    de um so: na primeira tentativa era um unico sorteio com semente fixa, e os
    valores dos datasets grandes entraram no CSV como se fossem exatos, sem
    variancia quantificada. X recebe a media entre replicas e o desvio fica
    gravado ao lado.
    """
    n = len(raro)
    if n <= max_instancias:
        return [np.ones(n, dtype=bool)]

    sementes = np.random.SeedSequence(seed).spawn(n_replicas)
    saida = []
    for semente in sementes:
        rng = np.random.default_rng(semente)
        manter = np.zeros(n, dtype=bool)
        for classe in (True, False):
            idx = np.flatnonzero(raro == classe)
            if not len(idx):
                continue
            k = max(1, round(len(idx) * max_instancias / n))
            manter[rng.choice(idx, size=min(k, len(idx)), replace=False)] = True
        saida.append(manter)
    return saida
