"""Friedman/Nemenyi para sete representações/tratamentos em cada meta-modelo."""
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, studentized_range

CONFIGURACOES = [
    ("base", "sem_selecao", "Base"),
    ("base_mais_todos", "sem_selecao", "Todos"),
    ("base_mais_normais_raros", "sem_selecao", "Normais + raros"),
    ("base_mais_todos_normais_raros", "sem_selecao", "Três grupos"),
    ("base_mais_todos", "selecao_13", "Todos / seleção"),
    ("base_mais_normais_raros", "selecao_13", "Normais + raros / seleção"),
    ("base_mais_todos_normais_raros", "selecao_13", "Três grupos / seleção"),
]
ABORDAGENS = {"Abordagem_1": "RF sobre desempenhos",
              "Abordagem_2": "RF sobre ranks", "HARRIS": "HARRIS"}


def grupos_maximais(ranks, cd):
    """Intervalos maximais em que toda diferença de ranks é <= CD."""
    candidatos = [(i, j) for i in range(len(ranks))
                  for j in range(i+1, len(ranks)) if ranks[j]-ranks[i] <= cd]
    return [(i, j) for i, j in candidatos
            if not any(a <= i and j <= b and (a, b) != (i, j)
                       for a, b in candidatos)]


def desenhar_cd(ax, ranks, cd, titulo, p, alpha):
    ranks = ranks.sort_values(kind="stable")
    k = len(ranks)
    ax.set_xlim(-2.6, k+3.6)
    ax.set_ylim(-1.65, 1.1)
    ax.axis("off")
    ax.set_title(f"{titulo} — Friedman p={p:.4g}; α por abordagem={alpha:.4f}", fontsize=11)
    ax.plot([1, k], [0.5, 0.5], color="black", lw=1)
    for pos in range(1, k+1):
        ax.plot([pos, pos], [0.46, 0.54], color="black", lw=1)
        ax.text(pos, 0.59, str(pos), ha="center", fontsize=9)
    ax.plot([1, 1+cd], [0.9, 0.9], color="black", marker="|", markersize=8)
    ax.text(1+cd/2, 0.96, f"CD = {cd:.3f}", ha="center", fontsize=9)
    for i, (nome, rank) in enumerate(ranks.items()):
        if i < (k+1)//2:
            y = -0.55-0.25*i
            ax.plot([rank, rank, 0.8], [0.5, y, y], color="0.65", lw=0.8)
            ax.text(0.7, y, f"{nome} ({rank:.2f})", ha="right", va="center", fontsize=9)
        else:
            y = -0.55-0.25*(k-1-i)
            ax.plot([rank, rank, k+0.2], [0.5, y, y], color="0.65", lw=0.8)
            ax.text(k+0.3, y, f"({rank:.2f}) {nome}", ha="left", va="center", fontsize=9)
    for nivel, (i, j) in enumerate(grupos_maximais(ranks.to_numpy(), cd)):
        y = 0.32-0.12*nivel
        ax.plot([ranks.iloc[i], ranks.iloc[j]], [y, y], color="black", lw=3)
    ax.text((k+1)/2, -1.5, "Menor rank médio = menor área de perda", ha="center", fontsize=9)


def comparar_representacoes(resultados, pasta="resultados/avaliacao_final",
                            pasta_figuras="figuras", alpha_global=0.05):
    pasta, pasta_figuras = Path(pasta), Path(pasta_figuras)
    pasta.mkdir(parents=True, exist_ok=True)
    pasta_figuras.mkdir(parents=True, exist_ok=True)
    # Nemenyi controla as comparações dentro de cada abordagem;
    # Bonferroni adicional contempla as três famílias de comparações.
    alpha = alpha_global / len(ABORDAGENS)
    omnibus, pares, registros_ranks = [], [], []
    fig, eixos = plt.subplots(3, 1, figsize=(14, 10.5))
    for ax, (abordagem, titulo) in zip(eixos, ABORDAGENS.items()):
        series = []
        for representacao, tratamento, nome in CONFIGURACOES:
            bloco = resultados.loc[
                (resultados["abordagem"] == abordagem)
                & (resultados["representacao"] == representacao)
                & (resultados["tratamento"] == tratamento)]
            assert bloco["dataset"].is_unique, "Dataset duplicado."
            series.append(bloco.set_index("dataset")["auc_perda"].rename(nome))
        matriz = pd.concat(series, axis=1).sort_index()
        assert len(matriz) == 30 and np.isfinite(matriz.to_numpy()).all()
        matriz.to_csv(pasta / f"areas_pareadas_{abordagem}.csv")
        n, k = matriz.shape
        if (matriz.nunique(axis=1) == 1).all():
            estatistica, p = 0.0, 1.0
        else:
            estatistica, p = friedmanchisquare(*[matriz[c] for c in matriz])
        ranks = matriz.rank(axis=1, method="average", ascending=True).mean()
        erro_padrao = np.sqrt(k*(k+1)/(6*n))
        cd = float(studentized_range.ppf(1-alpha, k, np.inf)/np.sqrt(2)*erro_padrao)
        omnibus.append({"abordagem": abordagem, "n_datasets": n, "k": k,
                        "friedman": float(estatistica), "p_friedman": float(p),
                        "p_friedman_bonferroni": min(1., p*3),
                        "alpha_por_abordagem": alpha, "CD": cd,
                        "rejeita_friedman": p < alpha})
        for nome in ranks.index:
            registros_ranks.append({"abordagem": abordagem, "configuracao": nome,
                                    "rank_medio_area": ranks[nome],
                                    "area_media": matriz[nome].mean()})
        for a, b in combinations(ranks.index, 2):
            diferenca = abs(ranks[a]-ranks[b])
            p_nem = float(studentized_range.sf(np.sqrt(2)*diferenca/erro_padrao, k, np.inf))
            pares.append({"abordagem": abordagem, "configuracao_a": a, "configuracao_b": b,
                          "diferenca_rank": diferenca, "p_nemenyi": p_nem,
                          "p_nemenyi_bonferroni": min(1., 3*p_nem),
                          "significativo": bool(p < alpha and diferenca > cd)})
        desenhar_cd(ax, ranks, cd, titulo, p, alpha)
    fig.suptitle("Diferença crítica — área da curva de perda", fontsize=14)
    fig.text(0.5, 0.012,
             "Barras: grupos sem diferença detectada por Nemenyi; não demonstram equivalência.\n"
             "Conclusões por pares exigem também rejeição de Friedman. α global = 0,05; três abordagens.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.055, 1, 0.97])
    fig.savefig(pasta_figuras / "diferenca_critica_area.pdf", bbox_inches="tight")
    fig.savefig(pasta_figuras / "diferenca_critica_area.png", dpi=200, bbox_inches="tight")
    omnibus, pares, registros_ranks = map(pd.DataFrame, (omnibus, pares, registros_ranks))
    omnibus.to_csv(pasta / "friedman_area.csv", index=False)
    pares.to_csv(pasta / "nemenyi_area.csv", index=False)
    registros_ranks.to_csv(pasta / "ranks_medios_area.csv", index=False)
    return omnibus, pares, registros_ranks, fig
