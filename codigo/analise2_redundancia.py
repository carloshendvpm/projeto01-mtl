"""Analise 2 do enunciado: as colunas novas trazem informacao nova?

O enunciado oferece tres instrumentos -- correlacao entre meta-caracteristicas,
agrupamento das redundantes, ou eliminacao reversa -- e pede que se escolha um.

Escolha: AGRUPAMENTO HIERARQUICO sobre a distancia 1 - |rho de Spearman|,
ligacao media, corte em |rho| >= 0,90.

Por que este e nao os outros dois. Correlacao pura produz uma matriz 49x49 que
nao se le e nao da veredito. Eliminacao reversa exige um alvo, logo exigiria
rodar dentro do fold para nao vazar -- e ai seria selecao, nao analise de
redundancia; ela tem lugar, mas no experimento. O agrupamento e o unico que
responde na forma pedida: cada cluster e so-novo, so-classico ou MISTO, e
cluster misto significa que a coluna nova nao acrescenta nada as classicas.

Ponto que a primeira tentativa deixou de fora: ela so correlacionou cada medida
consigo mesma em outro grupo. O enunciado pede as novas CONTRA as classicas.

Instrumento secundario: R^2 de validacao cruzada leave-one-out com Ridge, de
cada coluna nova regredida nas 13 classicas -- porque correlacao par a par nao
captura que uma coluna nova pode ser combinacao de varias classicas. R^2
ordinario nao serve: com 13 preditores e 30 linhas ele chegaria a 0,6-0,7 para
ruido puro. O R^2-LOO pode ser negativo, e o seu zero significa "as classicas
nao predizem esta coluna".
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.model_selection import LeaveOneOut, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "lib"))

from montar_x import BLOCO_NOVO, carregar, origem_da_coluna  # noqa: E402

CORTE_RHO = 0.90
ALFAS = np.logspace(-3, 3, 25)
CORES = {"base": "#1f77b4", "all": "#2ca02c", "normal": "#ff7f0e", "raro": "#d62728"}


def matriz_de_distancia(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """1 - |rho|, com as colunas constantes removidas antes."""
    util = df.loc[:, df.std(numeric_only=True) > 0].dropna(axis=1, how="any")
    corr = util.corr(method="spearman").abs().fillna(0.0)
    D = 1.0 - corr.to_numpy()
    np.fill_diagonal(D, 0.0)
    return np.clip((D + D.T) / 2, 0, 2), list(util.columns)


def agrupar(df: pd.DataFrame, corte: float = 1 - CORTE_RHO):
    D, colunas = matriz_de_distancia(df)
    Z = linkage(squareform(D, checks=False), method="average")
    rotulos = fcluster(Z, t=corte, criterion="distance")
    return Z, colunas, pd.Series(rotulos, index=colunas, name="cluster")


def classificar_clusters(clusters: pd.Series) -> pd.DataFrame:
    """Cada cluster vira uma linha, com o veredito que interessa ao enunciado."""
    linhas = []
    for rotulo, membros in clusters.groupby(clusters):
        colunas = list(membros.index)
        origens = {origem_da_coluna(c) for c in colunas}
        tem_base = "base" in origens
        tem_nova = bool(origens - {"base"})
        tipo = (
            "misto" if tem_base and tem_nova
            else "so_classico" if tem_base
            else "so_novo"
        )
        linhas.append({
            "cluster": rotulo, "tamanho": len(colunas), "tipo": tipo,
            "colunas": ", ".join(sorted(colunas)),
        })
    return pd.DataFrame(linhas).set_index("cluster").sort_values("tamanho", ascending=False)


def r2_loo_contra_classicas(X: pd.DataFrame, novas: pd.DataFrame) -> pd.DataFrame:
    """Quanto das colunas novas as 13 classicas ja predizem."""
    modelo = make_pipeline(StandardScaler(), RidgeCV(alphas=ALFAS))
    base = X.to_numpy()
    linhas = []
    for coluna in novas.columns:
        alvo = novas[coluna]
        if alvo.isna().any() or alvo.std() == 0:
            linhas.append({"coluna": coluna, "r2_loo": np.nan, "max_abs_rho_classica": np.nan})
            continue
        y = alvo.to_numpy()
        pred = cross_val_predict(modelo, base, y, cv=LeaveOneOut())
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        rho = X.corrwith(alvo, method="spearman").abs().max()
        linhas.append({"coluna": coluna, "r2_loo": float(r2),
                       "max_abs_rho_classica": float(rho)})
    return pd.DataFrame(linhas).set_index("coluna").sort_values("r2_loo", ascending=False)


def teste_de_sanidade(clusters: pd.Series, longo: pd.DataFrame, X: pd.DataFrame) -> str:
    """T2_all e o inverso de `instances_per_feature`, ja na base.

    Tem de cair no mesmo cluster. Se nao cair, o instrumento esta errado e
    nenhuma conclusao de redundancia vale. T2 nao entra em X -- entra aqui so
    para validar o metodo.
    """
    t2 = (longo[longo["grupo"] == "all"]
          .groupby("dataset")["T2"].mean()
          .reindex(X.index).rename("T2_all"))
    aumentado = X.join(t2)
    _, _, clusters2 = agrupar(aumentado)
    if "T2_all" not in clusters2 or "instances_per_feature" not in clusters2:
        return "INCONCLUSIVO: T2_all ou instances_per_feature fora do agrupamento"
    mesmo = clusters2["T2_all"] == clusters2["instances_per_feature"]
    return ("OK: T2_all agrupa com instances_per_feature"
            if mesmo else
            "FALHOU: T2_all nao agrupou com instances_per_feature -- instrumento suspeito")


def figura_dendrograma(Z, colunas, saida: Path):
    fig, eixo = plt.subplots(figsize=(15, 7))
    dend = dendrogram(Z, labels=colunas, ax=eixo, leaf_rotation=90, leaf_font_size=7,
                      color_threshold=1 - CORTE_RHO)
    eixo.axhline(1 - CORTE_RHO, color="crimson", ls="--", lw=1,
                 label=f"corte em |rho| = {CORTE_RHO}")
    for rotulo in eixo.get_xticklabels():
        rotulo.set_color(CORES[origem_da_coluna(rotulo.get_text())])
    eixo.set_ylabel("1 - |rho de Spearman|")
    eixo.set_title("Redundancia entre meta-caracteristicas\n"
                   "azul: base do Projeto 1 | verde: todos | laranja: normais | vermelho: raros")
    eixo.legend()
    fig.tight_layout()
    fig.savefig(saida, dpi=140)
    plt.close(fig)
    return dend


def main():
    X, Xc = carregar()
    novas = Xc[BLOCO_NOVO]
    completo = X.join(novas)

    Z, colunas, clusters = agrupar(completo)
    tabela = classificar_clusters(clusters)
    r2 = r2_loo_contra_classicas(X, novas)

    tabela.to_csv(BASE_DIR / "pratica-02" / "resultados" / "redundancia_clusters.csv")
    r2.to_csv(BASE_DIR / "pratica-02" / "resultados" / "redundancia_r2_loo.csv")
    clusters.to_frame().assign(origem=[origem_da_coluna(c) for c in clusters.index]).to_csv(
        BASE_DIR / "pratica-02" / "resultados" / "redundancia_coluna_cluster.csv"
    )
    figura_dendrograma(Z, colunas, BASE_DIR / "pratica-02" / "resultados" / "redundancia_dendrograma.png")

    longo = pd.read_csv(BASE_DIR / "pratica-02" / "resultados" / "complexidade.csv")
    sanidade = teste_de_sanidade(clusters, longo, X)

    novas_agrupadas = clusters[[c for c in clusters.index if origem_da_coluna(c) != "base"]]
    tipo_por_coluna = novas_agrupadas.map(tabela["tipo"])
    em_misto = int((tipo_por_coluna == "misto").sum())
    singletons = int((clusters.value_counts()[novas_agrupadas] == 1).sum())

    print(f"teste de sanidade do instrumento -> {sanidade}\n")
    print(f"colunas no agrupamento: {len(colunas)} "
          f"({len(X.columns)} classicas + {len(novas.columns)} novas)")
    print(f"clusters ao corte |rho| >= {CORTE_RHO}: {clusters.nunique()} "
          f"-- dimensionalidade efetiva")
    print(f"colunas novas em cluster MISTO (nao acrescentam): {em_misto}/{len(novas.columns)}")
    print(f"colunas novas isoladas (informacao nova): {singletons}/{len(novas.columns)}")
    print("\nclusters por tipo:")
    print(tabela["tipo"].value_counts().to_string())
    print("\nmenos previsiveis pelas classicas (mais informacao nova):")
    print(r2.head(8).round(3).to_string())
    print("\nmais previsiveis pelas classicas (redundantes):")
    print(r2.tail(8).round(3).to_string())


if __name__ == "__main__":
    main()
