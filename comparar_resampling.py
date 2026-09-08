from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from abordagens_agregacao import rank_medio

BASE_DIR = Path(__file__).parent
SAIDA = BASE_DIR / "assets" / "relatorio"


def decimal(valor):
    return f"{valor:.3f}".replace(".", "{,}")


def main():
    R = pd.read_csv(BASE_DIR / "R.csv", index_col=0)
    R_rs = pd.read_csv(BASE_DIR / "R_resampling.csv", index_col=0)[R.columns]

    spearmans = []
    venceu_igual = 0
    for dataset in R.index:
        rho, _ = spearmanr(R.loc[dataset], R_rs.loc[dataset])
        spearmans.append(rho)
        if R.loc[dataset].idxmin() == R_rs.loc[dataset].idxmin():
            venceu_igual += 1

    spearman_medio = float(np.mean(spearmans))
    pct_igual = venceu_igual / len(R.index)

    ar_sem = rank_medio(R).rename("Sem resampling")
    ar_com = rank_medio(R_rs).rename("Com resampling (WERCS)")
    tabela_ar = pd.concat([ar_sem, ar_com], axis=1).sort_values("Sem resampling")

    print(f"Spearman médio entre R e R_resampling: {spearman_medio:.3f}")
    print(f"Datasets em que o algoritmo de rank 1 não muda: {venceu_igual}/{len(R.index)} ({pct_igual:.0%})")
    print("\nAR (rank médio) sem vs. com resampling:")
    print(tabela_ar)

    SAIDA.mkdir(parents=True, exist_ok=True)
    linhas = [
        rf"Spearman médio entre os dois rankings & ${decimal(spearman_medio)}$ \\",
        rf"Datasets com o mesmo algoritmo em 1\textordmasculine{{}} lugar & {venceu_igual}/{len(R.index)} ({pct_igual:.0%}) \\",
    ]
    corpo = (
        "\\begin{tabular}{lr}\n\\toprule\n"
        + "\n\\midrule\n".join(linhas)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (SAIDA / "tabela_resampling.tex").write_text(corpo, encoding="utf-8")

    linhas_ar = []
    for algo in tabela_ar.index:
        linhas_ar.append(
            f"{algo} & {tabela_ar.loc[algo, 'Sem resampling']:.1f} & "
            f"{tabela_ar.loc[algo, 'Com resampling (WERCS)']:.1f} \\\\"
        )
    corpo_ar = (
        "\\begin{tabular}{lrr}\n\\toprule\n"
        "Algoritmo & AR sem resampling & AR com resampling \\\\\n\\midrule\n"
        + "\n".join(linhas_ar)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (SAIDA / "tabela_resampling_ar.tex").write_text(corpo_ar, encoding="utf-8")
    print(f"\nSalvo em {SAIDA}")


if __name__ == "__main__":
    main()
