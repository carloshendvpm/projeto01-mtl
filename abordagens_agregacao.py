from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

BASE_DIR = Path(__file__).parent
ALPHA = 0.05


def rank_medio(R: pd.DataFrame) -> pd.Series:
    return R.mean(axis=0).rank(method="average", ascending=True)


def rank_mediano(R: pd.DataFrame) -> pd.Series:
    return R.median(axis=0).rank(method="average", ascending=True)


def vitorias_significativas(resultados_por_fold: pd.DataFrame, alpha: float = ALPHA) -> pd.Series:
    algoritmos = sorted(resultados_por_fold["algoritmo"].unique())
    vitorias = {a: 0 for a in algoritmos}

    for _, grupo in resultados_por_fold.groupby("dataset"):
        pivot = grupo.pivot(index="fold", columns="algoritmo", values="sera")
        for a, b in combinations(algoritmos, 2):
            x, y = pivot[a].to_numpy(), pivot[b].to_numpy()
            try:
                _, p = wilcoxon(x, y)
            except ValueError:
                continue
            if p < alpha:
                vencedor = a if x.mean() < y.mean() else b
                vitorias[vencedor] += 1

    contagem = pd.Series(vitorias)
    return contagem.rank(method="average", ascending=False)


def main():
    R = pd.read_csv(BASE_DIR / "R.csv", index_col=0)
    resultados = pd.read_csv(BASE_DIR / "resultados_por_fold.csv")

    ar = rank_medio(R)
    mr = rank_mediano(R)
    vs = vitorias_significativas(resultados)

    consenso = pd.DataFrame({"AR": ar, "MR": mr, "Vitorias_Significativas": vs})
    print(consenso.sort_values("AR"))


if __name__ == "__main__":
    main()
