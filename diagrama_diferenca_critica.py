from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare

BASE_DIR = Path(__file__).parent
ABORDAGENS_OFICIAIS = ["AR", "MR", "Vitorias_Significativas", "Abordagem_1", "Abordagem_2", "HARRIS"]


def main():
    avaliacao = pd.read_csv(BASE_DIR / "avaliacao_por_dataset.csv")
    avaliacao = avaliacao[avaliacao["abordagem"].isin(ABORDAGENS_OFICIAIS)]

    matriz = avaliacao.pivot(index="dataset", columns="abordagem", values="spearman")
    matriz = matriz[ABORDAGENS_OFICIAIS]

    estatistica, p_valor = friedmanchisquare(*[matriz[c] for c in matriz.columns])
    print(f"Friedman: estatistica={estatistica:.4f}, p-valor={p_valor:.4f}")

    p_valores_nemenyi = sp.posthoc_nemenyi_friedman(matriz.to_numpy())
    p_valores_nemenyi.columns = matriz.columns
    p_valores_nemenyi.index = matriz.columns
    print("\nP-valores do pos-teste de Nemenyi:")
    print(p_valores_nemenyi.round(3))

    ranks_medios = matriz.rank(axis=1, ascending=False).mean()

    plt.figure(figsize=(8, 3))
    sp.critical_difference_diagram(ranks_medios, p_valores_nemenyi)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "diagrama_diferenca_critica.png", dpi=150)
    print("\nSalvo em diagrama_diferenca_critica.png")


if __name__ == "__main__":
    main()
