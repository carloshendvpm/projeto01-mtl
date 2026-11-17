"""Compara as medidas entre os tres grupos: todos, normais e raros.

Responde a pergunta da professora -- a caracterizacao separada traz informacao
adicional? -- e, antes disso, se a diferenca observada entre raros e normais e
mesmo diferenca de dificuldade.

As familias de hipoteses sao PRE-REGISTRADAS aqui, antes de olhar qualquer
p-valor, porque a correcao para multiplas comparacoes so faz sentido sobre uma
familia declarada:

  F1  primaria      z do grupo raro contra a nula `normal`, testado contra 0.
                    E "os raros sao mais dificeis?" ja controlado por tamanho:
                    compara o grupo raro com regioes normais DO MESMO TAMANHO.
  F2  fidelidade    Wilcoxon raro x normal bruto. E o teste da primeira
                    tentativa, mantido para exibir o contraste com F1, e
                    rotulado como CONFUNDIDO POR TAMANHO.
  F3  contexto      Wilcoxon raro x todos bruto.
  F4  redundancia   rho entre as versoes da mesma medida, SEM teste. A
                    pergunta ali e "quanta informacao nova", que rho responde
                    direto; nao testar reduz a familia de 42 hipoteses a zero.

Correcao de Benjamini-Hochberg como primaria, dentro de cada familia, com Holm
ao lado. Efeito e intervalo sempre junto do p: um p alto com intervalo estreito
diz "o efeito e pequeno"; com intervalo largo diz "nao sabemos".
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "lib"))

from complexidade import MEDIDAS_EM_X, MEDIDAS_META  # noqa: E402
from grupos_phi import GRUPOS  # noqa: E402

ALFA = 0.05
N_BOOTSTRAP = 2000
SEED = 42


def chave_canonica(medida: str) -> str:
    """A coluna que representa a medida: `_sq` se agregativa, bruta se nao."""
    return f"{medida}_sq" if MEDIDAS_META[medida].agregativa else medida


# ------------------------------------------------------------------- efeitos


def rank_biserial(diferencas: np.ndarray) -> float:
    """Tamanho de efeito do Wilcoxon pareado, em [-1, 1]."""
    d = diferencas[diferencas != 0]
    if not len(d):
        return 0.0
    postos = pd.Series(np.abs(d)).rank().to_numpy()
    mais, menos = postos[d > 0].sum(), postos[d < 0].sum()
    return float((mais - menos) / (mais + menos))


def hodges_lehmann(d: np.ndarray) -> float:
    """Mediana das medias de Walsh: o deslocamento tipico, robusto."""
    if not len(d):
        return np.nan
    i, j = np.triu_indices(len(d), k=0)
    return float(np.median((d[i] + d[j]) / 2))


def ic_bootstrap(d: np.ndarray, seed: int = SEED) -> tuple[float, float]:
    """IC de 95% da mediana das diferencas, reamostrando DATASETS."""
    if len(d) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    amostras = rng.choice(d, size=(N_BOOTSTRAP, len(d)), replace=True)
    medianas = np.median(amostras, axis=1)
    return tuple(float(v) for v in np.percentile(medianas, [2.5, 97.5]))


def _testar_pareado(a: pd.Series, b: pd.Series) -> dict:
    """Wilcoxon pareado bilateral com efeito e intervalo."""
    par = pd.concat([a, b], axis=1).dropna()
    if len(par) < 5:
        return {"n": len(par), "p": np.nan, "efeito": np.nan,
                "desloc": np.nan, "ic_baixo": np.nan, "ic_alto": np.nan}
    d = (par.iloc[:, 0] - par.iloc[:, 1]).to_numpy()
    p = np.nan if np.allclose(d, 0) else float(wilcoxon(d).pvalue)
    baixo, alto = ic_bootstrap(d)
    return {"n": len(par), "p": p, "efeito": rank_biserial(d),
            "desloc": hodges_lehmann(d), "ic_baixo": baixo, "ic_alto": alto}


def corrigir(df: pd.DataFrame, coluna: str = "p") -> pd.DataFrame:
    """BH (primaria) e Holm (conservadora), dentro da familia recebida."""
    df = df.copy()
    validos = df[coluna].notna()
    df["p_bh"] = np.nan
    df["p_holm"] = np.nan
    if validos.any():
        p = df.loc[validos, coluna].to_numpy()
        df.loc[validos, "p_bh"] = multipletests(p, alpha=ALFA, method="fdr_bh")[1]
        df.loc[validos, "p_holm"] = multipletests(p, alpha=ALFA, method="holm")[1]
    df["sig_bh"] = df["p_bh"] < ALFA
    df["sig_holm"] = df["p_holm"] < ALFA
    return df


# -------------------------------------------------------------------- dados


def carregar(base_dir: Path = BASE_DIR):
    longo = pd.read_csv(base_dir / "pratica-02" / "resultados" / "complexidade.csv")
    # Media entre replicas de subamostragem: um valor por dataset e grupo.
    valores = longo.groupby(["dataset", "grupo"]).mean(numeric_only=True).reset_index()
    nulo = pd.read_csv(base_dir / "pratica-02" / "resultados" / "complexidade_nulo.csv")
    Xc = pd.read_csv(base_dir / "meta-dataset" / "X_complexidade.csv", index_col=0)
    X = pd.read_csv(base_dir / "meta-dataset" / "X.csv", index_col=0)
    return valores, nulo, Xc, X


def por_grupo(valores: pd.DataFrame, chave: str) -> pd.DataFrame:
    """Uma coluna por grupo, indexada por dataset."""
    return valores.pivot(index="dataset", columns="grupo", values=chave)


# ------------------------------------------------------------------ familias


def familia_1(nulo: pd.DataFrame, tipo: str = "normal") -> pd.DataFrame:
    """z do grupo raro contra a nula, testado contra zero.

    H0: o grupo raro e indistinguivel de um subconjunto do mesmo tamanho. Este
    e o teste que a primeira tentativa deveria ter feito.
    """
    linhas = []
    sub = nulo[nulo["nulo"] == tipo]
    for medida in MEDIDAS_EM_X:
        z = sub[sub["medida"] == chave_canonica(medida)].set_index("dataset")["z"].dropna()
        if len(z) < 5:
            linhas.append({"medida": medida, "n": len(z), "p": np.nan})
            continue
        d = z.to_numpy()
        baixo, alto = ic_bootstrap(d)
        # A MEDIANA e a leitura valida. A distribuicao de z entre datasets e
        # fortemente assimetrica: em datasets de alvo muito concentrado, como
        # `meta`, a nula tem variancia minuscula e o z passa de 10^4, o que
        # sozinho destroi qualquer media. O teste ja e de postos, logo imune;
        # so o resumo precisava mudar.
        linhas.append({
            "medida": medida, "n": len(z),
            "z_mediano": float(np.median(d)), "z_medio": float(d.mean()),
            "z_max": float(d.max()),
            "p": float(wilcoxon(d).pvalue) if not np.allclose(d, 0) else np.nan,
            "efeito": rank_biserial(d), "desloc": hodges_lehmann(d),
            "ic_baixo": baixo, "ic_alto": alto,
        })
    return corrigir(pd.DataFrame(linhas).set_index("medida"))


def familia_bruta(valores: pd.DataFrame, contra: str) -> pd.DataFrame:
    """Wilcoxon raro x `contra`, sem controle de tamanho."""
    linhas = []
    for medida in MEDIDAS_EM_X:
        tabela = por_grupo(valores, chave_canonica(medida))
        resultado = _testar_pareado(tabela["raro"], tabela[contra])
        linhas.append({
            "medida": medida,
            **{f"media_{g}": tabela[g].mean() for g in GRUPOS},
            **resultado,
        })
    return corrigir(pd.DataFrame(linhas).set_index("medida"))


def familia_4_redundancia(valores: pd.DataFrame) -> pd.DataFrame:
    """rho entre as versoes da mesma medida. Descritiva, sem teste.

    rho(normal, all) alto significa que a versao dos normais nao acrescenta
    nada -- eram a maioria dos exemplos. rho(raro, all) baixo significa que a
    versao rara reordena os datasets, que e a condicao para ser meta-
    caracteristica util.
    """
    linhas = []
    for medida in MEDIDAS_EM_X:
        t = por_grupo(valores, chave_canonica(medida)).dropna()
        def rho(a, b):
            return float(spearmanr(t[a], t[b]).statistic) if len(t) >= 5 else np.nan
        linhas.append({
            "medida": medida, "n": len(t),
            "rho_raro_all": rho("raro", "all"),
            "rho_raro_normal": rho("raro", "normal"),
            "rho_normal_all": rho("normal", "all"),
        })
    return pd.DataFrame(linhas).set_index("medida")


def estratificar_por_cauda(nulo: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """F1 separada em datasets de uma e de duas caudas relevantes.

    phi bimodal poe extremos altos e baixos no mesmo grupo, disperso por
    construcao. Se o efeito so aparece nos bimodais, e dispersao, nao
    dificuldade.
    """
    duas = (X["pct_relevante_alta"] > 0) & (X["pct_relevante_baixa"] > 0)
    rotulo = duas.map({True: "duas_caudas", False: "uma_cauda"})
    sub = nulo[nulo["nulo"] == "normal"].copy()
    sub["cauda"] = sub["dataset"].map(rotulo)

    linhas = []
    for medida in MEDIDAS_EM_X:
        chave = chave_canonica(medida)
        linha = {"medida": medida}
        for cauda in ("uma_cauda", "duas_caudas"):
            z = sub[(sub["medida"] == chave) & (sub["cauda"] == cauda)]["z"].dropna()
            linha[f"n_{cauda}"] = len(z)
            linha[f"z_medio_{cauda}"] = float(z.mean()) if len(z) else np.nan
            linha[f"p_{cauda}"] = (
                float(wilcoxon(z.to_numpy()).pvalue)
                if len(z) >= 5 and not np.allclose(z, 0) else np.nan
            )
        linhas.append(linha)
    return pd.DataFrame(linhas).set_index("medida")


def confundimento_com_tamanho(valores: pd.DataFrame, Xc: pd.DataFrame) -> pd.DataFrame:
    """rho entre a medida do grupo raro e |R|.

    Evidencia direta do confundimento: se a medida do grupo raro so acompanha
    o tamanho do grupo, ela nao esta medindo dificuldade.
    """
    linhas = []
    for medida in MEDIDAS_EM_X:
        t = por_grupo(valores, chave_canonica(medida))["raro"]
        par = pd.concat([t, Xc["n_raro"]], axis=1).dropna()
        linhas.append({
            "medida": medida,
            "rho_com_n_raro": float(spearmanr(par.iloc[:, 0], par.iloc[:, 1]).statistic)
            if len(par) >= 5 else np.nan,
        })
    return pd.DataFrame(linhas).set_index("medida")


# ------------------------------------------------------------------ figuras


def figura_grupos(valores: pd.DataFrame, saida: Path):
    n = len(MEDIDAS_EM_X)
    fig, eixos = plt.subplots(3, 4, figsize=(14, 9))
    for eixo, medida in zip(eixos.ravel(), MEDIDAS_EM_X):
        t = por_grupo(valores, chave_canonica(medida))
        eixo.boxplot([t[g].dropna() for g in GRUPOS], tick_labels=["todos", "normais", "raros"])
        eixo.set_title(medida, fontsize=10)
        eixo.tick_params(labelsize=8)
    for eixo in eixos.ravel()[n:]:
        eixo.axis("off")
    fig.suptitle("Medidas por grupo nos 30 datasets (escala squashed)")
    fig.tight_layout()
    fig.savefig(saida, dpi=140)
    plt.close(fig)


def figura_z(nulo: pd.DataFrame, saida: Path, tipo: str = "normal"):
    """A figura principal: o que sobra do contraste depois do controle."""
    sub = nulo[nulo["nulo"] == tipo]
    dados = [
        sub[sub["medida"] == chave_canonica(m)]["z"].dropna().to_numpy()
        for m in MEDIDAS_EM_X
    ]
    fig, eixo = plt.subplots(figsize=(11, 5))
    eixo.axhspan(-2, 2, color="0.92", zorder=0)
    eixo.axhline(0, color="0.4", lw=1, zorder=1)
    eixo.boxplot(dados, tick_labels=list(MEDIDAS_EM_X), zorder=2)
    eixo.set_ylabel(f"z do grupo raro contra a nula '{tipo}'")
    eixo.set_title(
        "Controle de tamanho: o grupo raro contra subconjuntos de mesmo |R|\n"
        "dentro da faixa cinza, a diferenca e compativel com o tamanho do grupo"
    )
    fig.tight_layout()
    fig.savefig(saida, dpi=140)
    plt.close(fig)


def figura_vs_n_raro(valores: pd.DataFrame, Xc: pd.DataFrame, saida: Path):
    fig, eixos = plt.subplots(3, 4, figsize=(14, 9))
    for eixo, medida in zip(eixos.ravel(), MEDIDAS_EM_X):
        t = por_grupo(valores, chave_canonica(medida))["raro"]
        par = pd.concat([t.rename("v"), Xc["n_raro"]], axis=1).dropna()
        eixo.scatter(par["n_raro"], par["v"], s=14)
        eixo.set_xscale("log")
        rho = spearmanr(par["n_raro"], par["v"]).statistic if len(par) >= 5 else np.nan
        eixo.set_title(f"{medida}  (rho={rho:.2f})", fontsize=10)
        eixo.tick_params(labelsize=8)
    for eixo in eixos.ravel()[len(MEDIDAS_EM_X) :]:
        eixo.axis("off")
    fig.suptitle("Medida do grupo raro contra o tamanho do grupo")
    fig.tight_layout()
    fig.savefig(saida, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------- main


def main():
    valores, nulo, Xc, X = carregar()

    f1 = familia_1(nulo, "normal")
    f1_qualquer = familia_1(nulo, "qualquer")
    f1_janela = familia_1(nulo, "janela_y")
    f2 = familia_bruta(valores, "normal")
    f3 = familia_bruta(valores, "all")
    f4 = familia_4_redundancia(valores)
    caudas = estratificar_por_cauda(nulo, X)
    tamanho = confundimento_com_tamanho(valores, Xc)

    comparacao = pd.concat(
        [
            f2[[f"media_{g}" for g in GRUPOS]],
            f1[["z_mediano", "p", "p_bh", "p_holm", "sig_bh"]].add_prefix("F1_"),
            f1_qualquer[["z_mediano", "p_bh"]].add_prefix("F1qualquer_"),
            f1_janela[["z_mediano", "p_bh"]].add_prefix("F1janela_"),
            f2[["p", "p_bh", "efeito"]].add_prefix("F2_"),
            f3[["p", "p_bh", "efeito"]].add_prefix("F3_"),
            f4[["rho_raro_all", "rho_raro_normal", "rho_normal_all"]],
            tamanho,
        ],
        axis=1,
    )
    comparacao.to_csv(BASE_DIR / "pratica-02" / "resultados" / "complexidade_comparacao.csv")
    caudas.to_csv(BASE_DIR / "pratica-02" / "resultados" / "complexidade_caudas.csv")

    figura_grupos(valores, BASE_DIR / "pratica-02" / "resultados" / "complexidade_grupos.png")
    figura_z(nulo, BASE_DIR / "pratica-02" / "resultados" / "complexidade_z_nulo.png")
    figura_vs_n_raro(valores, Xc, BASE_DIR / "pratica-02" / "resultados" / "complexidade_vs_n_raro.png")

    pd.set_option("display.width", 200)
    print("F1 -- raros contra regioes normais do MESMO TAMANHO (primaria):")
    print(f1[["n", "z_mediano", "p", "p_bh", "sig_bh", "efeito"]].round(4).to_string())
    print("\nOs tres nulos lado a lado (mediana de z entre os 30 datasets):")
    tres = pd.DataFrame({
        "qualquer": f1_qualquer["z_mediano"], "normal": f1["z_mediano"],
        "janela_y": f1_janela["z_mediano"],
        "sig_normal": f1["sig_bh"], "sig_janela": f1_janela["sig_bh"],
    })
    print(tres.round(2).to_string())
    print("\nF2 -- raro x normal bruto (CONFUNDIDO POR TAMANHO, so para contraste):")
    print(f2[["media_normal", "media_raro", "p", "p_bh", "sig_bh"]].round(4).to_string())
    print("\nF4 -- redundancia entre as versoes (sem teste):")
    print(f4.round(3).to_string())

    sobrevivem = f1.index[f1["sig_bh"].fillna(False)].tolist()
    caem = f2.index[f2["sig_bh"].fillna(False) & ~f1["sig_bh"].fillna(False)].tolist()
    print(f"\nsignificativas com controle de tamanho: {sobrevivem or '-'}")
    print(f"significativas SO sem o controle (candidatas a artefato): {caem or '-'}")


if __name__ == "__main__":
    main()
