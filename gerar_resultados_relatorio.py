"""Gera tabelas e figuras do relatório a partir dos resultados já armazenados.

Execute antes de compilar relatorio.tex. Não retreina modelos nem altera os CSVs.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from matplotlib.ticker import FuncFormatter
from scipy.stats import friedmanchisquare

BASE_DIR = Path(__file__).parent
SAIDA = BASE_DIR / "assets" / "relatorio"
NOMES = {
    "AR": "AR",
    "MR": "MR",
    "Vitorias_Significativas": "Vitórias significativas",
    "Abordagem_1": "Abordagem 1",
    "Abordagem_2": "Abordagem 2",
    "HARRIS": "HARRIS",
}


def decimal(valor):
    return f"{valor:.3f}".replace(".", "{,}")


def cientifico(valor):
    mantissa, expoente = f"{valor:.3e}".split("e")
    return mantissa.replace(".", "{,}") + rf"\times 10^{{{int(expoente)}}}"


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    P = pd.read_csv(BASE_DIR / "P.csv", index_col=0)
    avaliacao = pd.read_csv(BASE_DIR / "avaliacao_por_dataset.csv")
    colunas = [f"perda_t{t}" for t in range(1, 7)]
    faixa = avaliacao["dataset"].map(P.max(axis=1) - P.min(axis=1))
    assert faixa.notna().all(), "Dataset da avaliação ausente em P"
    assert np.allclose(avaliacao[colunas].mean(axis=1), avaliacao["auc_perda"])
    assert np.allclose(avaliacao["perda_t6"], 0)
    assert (np.diff(avaliacao[colunas].to_numpy(), axis=1) <= 1e-12).all()

    # Desfaz a normalização por dataset antes de calcular a média entre datasets.
    original = avaliacao[colunas].mul(faixa, axis=0)
    avaliacao["auc_original"] = original.mean(axis=1)
    resumo = avaliacao.groupby("abordagem")[["spearman", "auc_original", "auc_perda"]].mean()
    oficiais = resumo.loc[list(NOMES)].sort_values("spearman", ascending=False)
    melhor_por_coluna = {
        "spearman": oficiais["spearman"].idxmax(),
        "auc_original": oficiais["auc_original"].idxmin(),
        "auc_perda": oficiais["auc_perda"].idxmin(),
    }
    linhas = []
    for nome, row in oficiais.iterrows():
        formatadores = [decimal, cientifico, decimal]
        colunas_tabela = ["spearman", "auc_original", "auc_perda"]
        valores = []
        for coluna, formatar in zip(colunas_tabela, formatadores):
            v = formatar(row[coluna])
            if melhor_por_coluna[coluna] == nome:
                v = rf"\mathbf{{{v}}}"
            valores.append(v)
        linhas.append(NOMES[nome] + " & " + " & ".join(f"${v}$" for v in valores) + r" \\")
    cabecalho = (
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        + r"Abordagem & Spearman médio $\uparrow$ & Área original $\downarrow$ & Área normalizada $\downarrow$ \\"
        + "\n\\midrule\n"
    )
    fim = "\n\\bottomrule\n\\end{tabular}\n"
    (SAIDA / "tabela_abordagens.tex").write_text(cabecalho + "\n".join(linhas) + fim, encoding="utf-8")

    linhas = []
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        nome = "HARRIS" if lam == 0.5 else f"HARRIS_lambda={lam}"
        row = resumo.loc[nome]
        valores = [decimal(lam), decimal(row.spearman), cientifico(row.auc_original), decimal(row.auc_perda)]
        linhas.append(" & ".join(f"${v}$" for v in valores) + r" \\")
    (SAIDA / "tabela_harris.tex").write_text(
        cabecalho.replace("{lrrr}", "{rrrr}").replace("Abordagem &", r"$\lambda$ &")
        + "\n".join(linhas) + fim, encoding="utf-8"
    )

    curvas_norm = avaliacao.groupby("abordagem")[colunas].mean()
    curvas_orig = original.groupby(avaliacao["abordagem"]).mean()
    estilos = [("o", "-"), ("s", "--"), ("x", ":"), ("^", "-."), ("D", "-"), ("v", "--")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55))
    for ax, curvas, titulo in zip(axes, [curvas_orig, curvas_norm], ["Perda original (SERA)", "Perda normalizada"]):
        for (nome, rotulo), (marcador, estilo) in zip(NOMES.items(), estilos):
            ax.plot(range(1, 7), curvas.loc[nome], marker=marcador, linestyle=estilo,
                    markersize=4, linewidth=1.2, label=rotulo)
        ax.set(title=titulo, xlabel="Número de algoritmos testados (t)", xticks=range(1, 7))
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.2)
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}".replace(".", ",")))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    fig.savefig(SAIDA / "curvas_perda.pdf", bbox_inches="tight")
    plt.close(fig)

    matriz = avaliacao.pivot(index="dataset", columns="abordagem", values="spearman")[list(NOMES)]
    assert matriz.shape == (30, 6) and matriz.notna().all().all()
    estatistica, p = friedmanchisquare(*[matriz[c] for c in matriz.columns])
    pvalores = sp.posthoc_nemenyi_friedman(matriz.to_numpy())
    pvalores.index = pvalores.columns = [NOMES[c] for c in matriz.columns]
    ranks = matriz.rank(axis=1, ascending=False).mean().rename(index=NOMES)
    fig, ax = plt.subplots(figsize=(7.0, 2.1))
    sp.critical_difference_diagram(ranks, pvalores, ax=ax, label_fmt_left="{label} ({rank:.2f})",
                                 label_fmt_right="({rank:.2f}) {label}")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}".replace(".", ",")))
    for texto in ax.texts:
        texto.set_text(texto.get_text().replace(".", ","))
    fig.tight_layout()
    fig.savefig(SAIDA / "diferenca_critica.pdf", bbox_inches="tight")
    plt.close(fig)
    print(resumo.to_string())
    print(f"Friedman sobre Spearman: chi2={estatistica:.6f}, p={p:.6f}")
    print(f"Artefatos gerados em {SAIDA}")


if __name__ == "__main__":
    main()
