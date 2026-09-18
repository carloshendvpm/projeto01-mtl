"""Medidas de complexidade por grupo: todos, normais e raros.

Para cada dataset, aplica a funcao de relevancia phi ao alvo, separa os
exemplos raros (phi >= 0.8) dos normais e calcula as medidas de complexidade
tres vezes: no conjunto inteiro, so nos normais e so nos raros.

A normalizacao min-max e feita UMA vez, no dataset inteiro, antes da divisao:
assim os valores dos tres grupos ficam na mesma escala e sao comparaveis.

Saidas:
  complexidade.csv        formato longo (dataset, grupo, medidas...)
  X_complexidade.csv      formato largo (uma linha por dataset, C1_all, C1_normal, ...)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).parent
sys.path.insert(0, str(RAIZ / "lib"))

from complexidade import MEDIDAS, MIN_INSTANCIAS, _minmax, calcular_todas  # noqa: E402
from phi import phi  # noqa: E402
from phi_ctrl_pts import phi_ctrl_pts  # noqa: E402

DATA_DIR = RAIZ / "data"
LIMIAR_RELEVANCIA = 0.8
GRUPOS = ("all", "normal", "raro")
MAX_INSTANCIAS = 1500  # medidas com matriz de distancia sao O(n^2)
SEED = 42


def _subamostrar(raro: np.ndarray) -> np.ndarray:
    """Amostra estratificada (mantem a proporcao de raros) se o dataset e grande."""
    n = len(raro)
    if n <= MAX_INSTANCIAS:
        return np.ones(n, dtype=bool)
    rng = np.random.default_rng(SEED)
    manter = np.zeros(n, dtype=bool)
    for classe in (True, False):
        idx = np.flatnonzero(raro == classe)
        k = max(1, round(len(idx) * MAX_INSTANCIAS / n))
        manter[rng.choice(idx, size=min(k, len(idx)), replace=False)] = True
    return manter


def por_grupo(csv_path: Path) -> list[dict]:
    df = pd.read_csv(csv_path)
    y_serie = df.iloc[:, 0]
    y = y_serie.to_numpy(dtype=float)
    X = df.iloc[:, 1:].to_numpy(dtype=float)

    relevancia = np.array(phi(y_serie, phi_ctrl_pts(y)))
    raro = relevancia >= LIMIAR_RELEVANCIA
    pct_raro = float(raro.mean())

    amostra = _subamostrar(raro)
    X, y, raro = X[amostra], y[amostra], raro[amostra]

    Xn = _minmax(X)
    yn = _minmax(y.reshape(-1, 1)).ravel()

    mascaras = {
        "all": np.ones(len(y), dtype=bool),
        "normal": ~raro,
        "raro": raro,
    }

    linhas = []
    for grupo, m in mascaras.items():
        medidas = calcular_todas(Xn[m], yn[m], normalizar=False)
        linhas.append({
            "dataset": csv_path.stem,
            "grupo": grupo,
            "n": int(m.sum()),
            "pct_raro": pct_raro,
            "subamostrado": bool(not amostra.all()),
            **medidas,
        })
    return linhas


def main():
    linhas = []
    for p in sorted(DATA_DIR.glob("*.csv")):
        print(f"... {p.stem}", flush=True)
        linhas.extend(por_grupo(p))

    longo = pd.DataFrame(linhas)
    longo.to_csv(RAIZ / "complexidade.csv", index=False)

    largo = longo.pivot(index="dataset", columns="grupo",
                        values=["n", *MEDIDAS])
    largo.columns = [f"{med}_{grupo}" for med, grupo in largo.columns]
    largo = largo[[f"{med}_{g}" for med in ("n", *MEDIDAS) for g in GRUPOS]]
    largo.insert(0, "pct_raro", longo.groupby("dataset")["pct_raro"].first())
    largo.to_csv(RAIZ / "X_complexidade.csv")

    n_raro = longo[longo.grupo == "raro"]
    print(f"\n{len(longo.dataset.unique())} datasets, {len(longo)} linhas.")
    print(f"Grupo raro com menos de {MIN_INSTANCIAS} exemplos (medidas NaN): "
          f"{(n_raro['n'] < MIN_INSTANCIAS).sum()} datasets")
    print("Salvo: complexidade.csv, X_complexidade.csv")


if __name__ == "__main__":
    main()
