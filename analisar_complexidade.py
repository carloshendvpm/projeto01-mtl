"""Analise das medidas de complexidade por grupo (todos / normais / raros).

Responde a duas perguntas:

1. Os valores diferem entre os grupos? Teste de Wilcoxon pareado entre grupos,
   sobre os datasets, para cada medida.
2. A caracterizacao separada traz informacao NOVA? Se a medida no grupo raro
   fosse apenas uma copia da medida no conjunto todo, a correlacao entre elas
   seria ~1 e nada se ganharia. Correlacao de Spearman baixa = informacao nova.

Saidas:
  complexidade_comparacao.csv   medias por grupo, testes e redundancia
  complexidade_grupos.png       um painel por medida (normal x raro)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

RAIZ = Path(__file__).parent
MEDIDAS = ["C1", "C2", "C3", "C4", "L1", "L2", "L3", "S1", "S2", "S3", "S4", "T2"]


def _wilcoxon(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """Wilcoxon pareado; devolve (estatistica, p) ou (nan, nan) se nao aplicavel."""
    par = pd.concat([a, b], axis=1).dropna()
    if len(par) < 5 or np.allclose(par.iloc[:, 0], par.iloc[:, 1]):
        return np.nan, np.nan
    r = wilcoxon(par.iloc[:, 0], par.iloc[:, 1])
    return float(r.statistic), float(r.pvalue)


def comparar(longo: pd.DataFrame) -> pd.DataFrame:
    largo = longo.pivot(index="dataset", columns="grupo", values=MEDIDAS)
    linhas = []
    for med in MEDIDAS:
        g = {grupo: largo[(med, grupo)] for grupo in ("all", "normal", "raro")}
        _, p_rn = _wilcoxon(g["raro"], g["normal"])
        _, p_ra = _wilcoxon(g["raro"], g["all"])
        rho_ra = spearmanr(g["raro"], g["all"], nan_policy="omit").statistic
        rho_rn = spearmanr(g["raro"], g["normal"], nan_policy="omit").statistic
        rho_na = spearmanr(g["normal"], g["all"], nan_policy="omit").statistic
        linhas.append({
            "medida": med,
            "media_all": g["all"].mean(),
            "media_normal": g["normal"].mean(),
            "media_raro": g["raro"].mean(),
            "n_datasets_raro": int(g["raro"].notna().sum()),
            "p_raro_vs_normal": p_rn,
            "p_raro_vs_all": p_ra,
            "rho_raro_all": rho_ra,
            "rho_raro_normal": rho_rn,
            "rho_normal_all": rho_na,
        })
    return pd.DataFrame(linhas).set_index("medida")


def graficos(longo: pd.DataFrame, saida: Path):
    largo = longo.pivot(index="dataset", columns="grupo", values=MEDIDAS)
    fig, eixos = plt.subplots(3, 4, figsize=(16, 11))
    for ax, med in zip(eixos.ravel(), MEDIDAS):
        dados = [largo[(med, g)].dropna() for g in ("all", "normal", "raro")]
        ax.boxplot(dados, tick_labels=["todos", "normais", "raros"])
        ax.set_title(med)
        ax.tick_params(labelsize=8)
    fig.suptitle("Medidas de complexidade por grupo (30 datasets)")
    fig.tight_layout()
    fig.savefig(saida, dpi=140)
    print(f"Salvo: {saida.name}")


def main():
    longo = pd.read_csv(RAIZ / "complexidade.csv")
    tabela = comparar(longo)
    tabela.to_csv(RAIZ / "complexidade_comparacao.csv")

    pd.set_option("display.width", 200)
    print("\n=== Medias por grupo e testes pareados (Wilcoxon) ===")
    print(tabela[["media_all", "media_normal", "media_raro",
                  "n_datasets_raro", "p_raro_vs_normal", "p_raro_vs_all"]].round(4))
    print("\n=== Redundancia: Spearman entre a medida nos grupos ===")
    print("(rho baixo em rho_raro_all = a versao separada traz informacao nova)")
    print(tabela[["rho_raro_all", "rho_raro_normal", "rho_normal_all"]].round(3))

    graficos(longo, RAIZ / "complexidade_grupos.png")
    print("Salvo: complexidade_comparacao.csv")


if __name__ == "__main__":
    main()
