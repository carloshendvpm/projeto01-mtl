"""Analise 1 do enunciado: como as colunas novas se relacionam com o meta-alvo.

O enunciado pede explicitamente que, havendo mais de um algoritmo base, se
examine SE O PERFIL DE IMPORTANCIA MUDA DE UM PARA OUTRO, como em De Amorim et
al. (2024), onde cada classificador base depende de familias diferentes de
meta-atributos. E o que a figura principal e a matriz 6x6 respondem.

Escolha de alvo. Nao se usa SERA bruto de `P.csv`: a escala e incomparavel
entre datasets (Linear vai de 794 mil a 4 milhoes, RF fica em milhares), e a
importancia acabaria medindo a ESCALA do problema, nao a adequacao do
algoritmo. O alvo principal e o rank de cada algoritmo, em 1..6.

Duas importancias, e a distincao importa:

  impureza     de graca, como o enunciado diz, e vai como manchete. Mas ela
               SOMA 1 por construcao, logo dobrar o numero de colunas dilui
               mecanicamente o bloco base. Por isso sai sempre acompanhada da
               importancia POR COLUNA. Os numeros da primeira tentativa (base
               0,470 / all 0,371 / raro 0,159) nao podiam ser lidos como "a
               base vale tres vezes o bloco raro", e foram.
  permutacao   fora da amostra, um round leave-one-out por vez. Mais lenta, mas
               e a unica sem o vies da impureza e sem ajustar e avaliar no
               mesmo dado. Vai como robustez.

Firewall: esta analise usa os 30 datasets, o que e legitimo porque ela e
descritiva e o enunciado a pede assim. Mas NENHUMA coluna do experimento e
escolhida com base nela -- o bloco novo entra inteiro, e qualquer reducao
acontece dentro do fold, em `selecao_meta_features.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "lib"))

from abordagens_meta_modelo import CONFIGS  # noqa: E402
from complexidade import MEDIDAS_META  # noqa: E402
from montar_x import BLOCO_NOVO, carregar, origem_da_coluna  # noqa: E402

SEED = 42
CONFIG = CONFIGS["regularizada"]


def familia_da_coluna(coluna: str) -> str:
    """Familia da medida (correlacao/linearidade/suavidade) ou `base`."""
    origem = origem_da_coluna(coluna)
    if origem == "base":
        return "base"
    medida = coluna.rsplit(f"_{origem}", 1)[0]
    return MEDIDAS_META[medida].familia if medida in MEDIDAS_META else "base"


def bloco_da_coluna(coluna: str) -> str:
    origem = origem_da_coluna(coluna)
    return "base" if origem == "base" else f"{familia_da_coluna(coluna)}_{origem}"


def importancia_por_impureza(X: pd.DataFrame, R: pd.DataFrame) -> pd.DataFrame:
    """Uma floresta por algoritmo base. Descritiva, ajustada no conjunto todo."""
    saida = {}
    for algoritmo in R.columns:
        modelo = RandomForestRegressor(random_state=SEED, n_jobs=-1, **CONFIG)
        modelo.fit(X, R[algoritmo])
        saida[algoritmo] = pd.Series(modelo.feature_importances_, index=X.columns)
    return pd.DataFrame(saida)


def importancia_por_permutacao(X: pd.DataFrame, R: pd.DataFrame, repeticoes: int = 5) -> pd.DataFrame:
    """Degradacao ao permutar cada coluna, medida FORA da amostra.

    Em cada round leave-one-dataset-out a floresta e ajustada nos 29 e o erro
    e medido no que ficou de fora, com e sem a coluna embaralhada. Sem o vies
    de cardinalidade da impureza e sem vazamento.
    """
    rng = np.random.default_rng(SEED)
    acumulado = {a: pd.Series(0.0, index=X.columns) for a in R.columns}

    for dataset_teste in X.index:
        X_train, X_test = X.drop(index=dataset_teste), X.loc[[dataset_teste]]
        for algoritmo in R.columns:
            y_train = R.drop(index=dataset_teste)[algoritmo]
            y_test = R.loc[dataset_teste, algoritmo]
            modelo = RandomForestRegressor(random_state=SEED, n_jobs=-1, **CONFIG)
            modelo.fit(X_train, y_train)
            erro_base = (modelo.predict(X_test)[0] - y_test) ** 2
            for coluna in X.columns:
                perdas = []
                for _ in range(repeticoes):
                    perturbado = X_test.copy()
                    perturbado[coluna] = rng.choice(X_train[coluna].to_numpy())
                    perdas.append((modelo.predict(perturbado)[0] - y_test) ** 2)
                acumulado[algoritmo][coluna] += np.mean(perdas) - erro_base

    return pd.DataFrame(acumulado) / len(X.index)


def similaridade_entre_perfis(importancias: pd.DataFrame) -> pd.DataFrame:
    """Matriz 6x6 de rho entre os vetores de importancia de cada algoritmo.

    E o numero que responde ao ponto de De Amorim: perto de 0,8-0,9 significa
    que o perfil e essencialmente o mesmo para todos os algoritmos; perto de
    0,2-0,4 significa que cada um depende de familias diferentes.
    """
    algoritmos = list(importancias.columns)
    M = pd.DataFrame(index=algoritmos, columns=algoritmos, dtype=float)
    for a in algoritmos:
        for b in algoritmos:
            M.loc[a, b] = spearmanr(importancias[a], importancias[b]).statistic
    return M


def por_bloco(importancias: pd.DataFrame) -> pd.DataFrame:
    """Soma e media por coluna dentro de cada bloco.

    A media por coluna e a leitura honesta: a soma favorece automaticamente o
    bloco com mais colunas.
    """
    blocos = importancias.index.map(bloco_da_coluna)
    soma = importancias.groupby(blocos).sum()
    tamanho = pd.Series(blocos).value_counts()
    media = soma.div(tamanho, axis=0)
    return soma, media


def associacao_direta(X: pd.DataFrame, R: pd.DataFrame) -> pd.DataFrame:
    """|rho| de cada coluna com o rank de cada algoritmo, com BH sobre tudo."""
    linhas = []
    for coluna in X.columns:
        for algoritmo in R.columns:
            r = spearmanr(X[coluna], R[algoritmo])
            linhas.append({"coluna": coluna, "algoritmo": algoritmo,
                           "rho": r.statistic, "p": r.pvalue,
                           "origem": origem_da_coluna(coluna)})
    df = pd.DataFrame(linhas)
    validos = df["p"].notna()
    df["p_bh"] = np.nan
    df.loc[validos, "p_bh"] = multipletests(df.loc[validos, "p"], method="fdr_bh")[1]
    df["abs_rho"] = df["rho"].abs()
    return df.sort_values("abs_rho", ascending=False)


def figura_perfis(media_por_bloco: pd.DataFrame, saida: Path):
    dados = media_por_bloco.T
    dados = dados.div(dados.sum(axis=1), axis=0)
    fig, eixo = plt.subplots(figsize=(11, 4.5))
    im = eixo.imshow(dados.to_numpy(), aspect="auto", cmap="viridis")
    eixo.set_xticks(range(dados.shape[1]), dados.columns, rotation=45, ha="right", fontsize=9)
    eixo.set_yticks(range(dados.shape[0]), dados.index, fontsize=9)
    for i in range(dados.shape[0]):
        for j in range(dados.shape[1]):
            eixo.text(j, i, f"{dados.iat[i, j]:.2f}", ha="center", va="center",
                      color="w", fontsize=7)
    fig.colorbar(im, ax=eixo, label="importancia media por coluna (normalizada)")
    eixo.set_title("O perfil de importancia muda de um algoritmo base para outro?")
    fig.tight_layout()
    fig.savefig(saida, dpi=140)
    plt.close(fig)


def main():
    X_base, Xc = carregar()
    X = X_base.join(Xc[BLOCO_NOVO])
    R = pd.read_csv(BASE_DIR / "meta-dataset" / "R.csv", index_col=0).loc[X.index]
    P = pd.read_csv(BASE_DIR / "meta-dataset" / "P.csv", index_col=0).loc[X.index]

    impureza = importancia_por_impureza(X, R)
    permutacao = importancia_por_permutacao(X, R)
    soma, media = por_bloco(impureza)
    similaridade = similaridade_entre_perfis(impureza)
    associacao = associacao_direta(X, R)

    impureza.to_csv(BASE_DIR / "pratica-02" / "resultados" / "importancia_por_algoritmo.csv")
    permutacao.to_csv(BASE_DIR / "pratica-02" / "resultados" / "importancia_permutacao.csv")
    similaridade.to_csv(BASE_DIR / "pratica-02" / "resultados" / "similaridade_perfis.csv")
    associacao.to_csv(BASE_DIR / "pratica-02" / "resultados" / "associacao_direta.csv", index=False)
    figura_perfis(media, BASE_DIR / "pratica-02" / "resultados" / "importancia_por_algoritmo.png")

    pd.set_option("display.width", 200)
    print("importancia por bloco -- soma (favorece blocos grandes):")
    print(soma.round(3).to_string())
    print("\nimportancia por bloco -- media POR COLUNA (leitura honesta):")
    print(media.round(4).to_string())

    print("\nsimilaridade entre os perfis dos 6 algoritmos (rho de Spearman):")
    print(similaridade.round(2).to_string())
    fora_diagonal = similaridade.to_numpy()[~np.eye(len(similaridade), dtype=bool)]
    print(f"\n  media {fora_diagonal.mean():.2f} | minimo {fora_diagonal.min():.2f}")
    print("  perto de 0,8-0,9: mesmo perfil para todos, ao contrario de De Amorim")
    print("  perto de 0,2-0,4: cada algoritmo depende de familias diferentes")

    print("\ntop-10 colunas por importancia media (impureza):")
    top = impureza.mean(axis=1).sort_values(ascending=False).head(10)
    for coluna, valor in top.items():
        em_top5 = [a for a in R.columns if coluna in impureza[a].nlargest(5).index]
        print(f"  {coluna:28s} {valor:.4f}  top-5 de: {', '.join(em_top5) or '-'}")

    print("\ntop-10 por permutacao fora da amostra:")
    print(permutacao.mean(axis=1).sort_values(ascending=False).head(10).round(4).to_string())

    vencedor = P.idxmin(axis=1).value_counts()
    print(f"\nmelhor algoritmo por dataset (alvo categorico): {vencedor.to_dict()}")
    if vencedor.max() >= 0.8 * len(P):
        print("  um algoritmo domina: o alvo categorico nao tem o que discriminar")

    print(f"\nassociacoes com p_bh < 0.05: {(associacao['p_bh'] < 0.05).sum()}"
          f" de {len(associacao)}")
    print(associacao.head(8)[["coluna", "algoritmo", "rho", "p_bh", "origem"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
