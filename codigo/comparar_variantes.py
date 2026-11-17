"""Protocolo estatistico da Analise 3, com o calculo de poder que o decide.

Uma hipotese primaria, declarada ANTES de rodar, sem correcao porque e uma so:

    `+ambos_sel13` contra `base`, metrica Spearman, abordagem `Abordagem_2`
    (a de melhor area de perda no Projeto 1), configuracao regularizada.

Todas as demais comparacoes braco-contra-base formam a familia exploratoria,
com Benjamini-Hochberg a q = 0,10 e os p brutos tabelados ao lado. BH e nao
Bonferroni porque os testes sao fortemente correlacionados -- mesmos 30
datasets, bracos encaixados -- e o objetivo e triagem, nao confirmacao.

A parte que decide a leitura do trabalho e o poder. Com 30 datasets pareados
so se detecta uma diferenca de 0,05 a 0,12 em Spearman medio, e diferencas
realistas entre variantes de X sao de ordem 0,01. Um p alto aqui NAO e
evidencia de ausencia de efeito; e ausencia de evidencia. Por isso a conclusao
e enquadrada por EQUIVALENCIA (TOST), que e uma afirmacao positiva: "o efeito,
qualquer que seja o sinal, e menor que a margem".

Agravante a registrar: com k = 6 algoritmos o rho de Spearman assume poucos
valores discretos (passo ~0,086), o que gera empates e retira poder adicional
do Wilcoxon. A granularidade da metrica e parte do teto.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, norm, wilcoxon
from statsmodels.stats.multitest import multipletests

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "lib"))

ALFA = 0.05
Q_EXPLORATORIA = 0.10
MARGEM_TOST = 0.05
PODER_ALVO = 0.80
N_BOOTSTRAP = 2000
SEED = 42

PRIMARIA = {"braco": "+ambos_sel13", "abordagem": "Abordagem_2",
            "config": "regularizada", "metrica": "spearman"}


def pares(avaliacao: pd.DataFrame, abordagem, config, braco, metrica):
    """Serie pareada por dataset: braco e base, sob a mesma abordagem/config."""
    sub = avaliacao[(avaliacao["abordagem"] == abordagem) & (avaliacao["config"] == config)]
    tabela = sub.pivot(index="dataset", columns="braco", values=metrica)
    if braco not in tabela or "base" not in tabela:
        return None
    return tabela[[braco, "base"]].dropna()


def ic_bootstrap(d: np.ndarray, seed: int = SEED):
    rng = np.random.default_rng(seed)
    amostras = rng.choice(d, size=(N_BOOTSTRAP, len(d)), replace=True)
    return tuple(float(v) for v in np.percentile(amostras.mean(axis=1), [2.5, 97.5]))


def efeito_minimo_detectavel(sd: float, n: int, poder=PODER_ALVO, alfa=ALFA) -> float:
    """Diferenca que um teste pareado com este n e este sd detecta.

    Aproximacao normal do teste t pareado: (z_{1-a/2} + z_{poder}) * sd / sqrt(n).
    Serve para dizer o que o meta-dataset e capaz de ver, antes de interpretar
    qualquer p-valor.
    """
    return float((norm.ppf(1 - alfa / 2) + norm.ppf(poder)) * sd / np.sqrt(n))


def tost(d: np.ndarray, margem: float = MARGEM_TOST) -> float:
    """Dois testes unilaterais: p de equivalencia dentro de +-margem.

    p < alfa permite AFIRMAR que o efeito e menor que a margem -- que e uma
    conclusao positiva, ao contrario de "nao deu significativo".
    """
    if len(d) < 5:
        return np.nan
    p1 = wilcoxon(d - (-margem), alternative="greater").pvalue
    p2 = wilcoxon(d - margem, alternative="less").pvalue
    return float(max(p1, p2))


def comparar_todos(avaliacao: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    com_x = avaliacao[avaliacao["braco"] != "teto"]
    for (abordagem, config), sub in com_x.groupby(["abordagem", "config"]):
        for braco in sub["braco"].unique():
            if braco == "base":
                continue
            for metrica in ("spearman", "auc_perda"):
                par = pares(avaliacao, abordagem, config, braco, metrica)
                if par is None or len(par) < 5:
                    continue
                d = (par[braco] - par["base"]).to_numpy()
                baixo, alto = ic_bootstrap(d)
                sd = float(d.std(ddof=1))
                linhas.append({
                    "abordagem": abordagem, "config": config, "braco": braco,
                    "metrica": metrica, "n": len(d),
                    "media_braco": float(par[braco].mean()),
                    "media_base": float(par["base"].mean()),
                    "diferenca": float(d.mean()),
                    "sd_pareado": sd,
                    "ic_baixo": baixo, "ic_alto": alto,
                    "p": np.nan if np.allclose(d, 0) else float(wilcoxon(d).pvalue),
                    "efeito_min_detectavel": efeito_minimo_detectavel(sd, len(d)),
                    "p_tost": tost(d) if metrica == "spearman" else np.nan,
                })
    df = pd.DataFrame(linhas)

    # A hipotese primaria sai da familia exploratoria: ela nao e corrigida.
    e_primaria = (
        (df["braco"] == PRIMARIA["braco"]) & (df["abordagem"] == PRIMARIA["abordagem"])
        & (df["config"] == PRIMARIA["config"]) & (df["metrica"] == PRIMARIA["metrica"])
    )
    df["primaria"] = e_primaria
    df["p_bh"] = np.nan
    exploratoria = ~e_primaria & df["p"].notna()
    if exploratoria.any():
        df.loc[exploratoria, "p_bh"] = multipletests(
            df.loc[exploratoria, "p"], alpha=Q_EXPLORATORIA, method="fdr_bh"
        )[1]
    df["sig_bh"] = df["p_bh"] < Q_EXPLORATORIA
    return df


def holm_contra_base(avaliacao: pd.DataFrame, metrica="spearman") -> pd.DataFrame:
    """Comparacao com controle: k-1 hipoteses, nao k(k-1)/2.

    Tem poder substancialmente maior que Nemenyi todos-contra-todos, que com
    muitos bracos e N=30 nao rejeitaria nada -- e isso seria informacao sobre o
    teste, nao sobre os dados.
    """
    linhas = []
    com_x = avaliacao[avaliacao["braco"] != "teto"]
    for (abordagem, config), sub in com_x.groupby(["abordagem", "config"]):
        ps, bracos = [], []
        for braco in sorted(set(sub["braco"]) - {"base"}):
            par = pares(avaliacao, abordagem, config, braco, metrica)
            if par is None or len(par) < 5:
                continue
            d = (par[braco] - par["base"]).to_numpy()
            if np.allclose(d, 0):
                continue
            ps.append(wilcoxon(d).pvalue)
            bracos.append(braco)
        if not ps:
            continue
        ajustados = multipletests(ps, alpha=ALFA, method="holm")[1]
        for braco, p, p_holm in zip(bracos, ps, ajustados):
            linhas.append({"abordagem": abordagem, "config": config, "braco": braco,
                           "p": p, "p_holm": p_holm, "sig_holm": p_holm < ALFA})
    return pd.DataFrame(linhas)


def friedman_por_abordagem(avaliacao: pd.DataFrame, metrica="spearman") -> pd.DataFrame:
    linhas = []
    com_x = avaliacao[avaliacao["braco"] != "teto"]
    for (abordagem, config), sub in com_x.groupby(["abordagem", "config"]):
        tabela = sub.pivot(index="dataset", columns="braco", values=metrica).dropna()
        if tabela.shape[1] < 3:
            continue
        est, p = friedmanchisquare(*[tabela[c] for c in tabela.columns])
        linhas.append({"abordagem": abordagem, "config": config,
                       "k_bracos": tabela.shape[1], "n": len(tabela),
                       "estatistica": est, "p": p})
    return pd.DataFrame(linhas)


def linha_de_teto(avaliacao: pd.DataFrame) -> pd.DataFrame:
    """O que um ranking fixo, igual para os 30 datasets, ja consegue."""
    teto = avaliacao[avaliacao["braco"] == "teto"]
    return teto.groupby("abordagem")[["spearman", "auc_perda"]].mean().sort_values(
        "spearman", ascending=False
    )


def main():
    avaliacao = pd.read_csv(BASE_DIR / "pratica-02" / "resultados" / "avaliacao_x_estendido.csv")

    comparacao = comparar_todos(avaliacao)
    holm = holm_contra_base(avaliacao)
    friedman = friedman_por_abordagem(avaliacao)
    teto = linha_de_teto(avaliacao)

    comparacao.to_csv(BASE_DIR / "pratica-02" / "resultados" / "comparacao_x_estendido.csv", index=False)
    holm.to_csv(BASE_DIR / "pratica-02" / "resultados" / "holm_contra_base.csv", index=False)
    friedman.to_csv(BASE_DIR / "pratica-02" / "resultados" / "friedman_por_abordagem.csv", index=False)
    poder = comparacao[["abordagem", "config", "braco", "metrica", "sd_pareado",
                        "diferenca", "efeito_min_detectavel", "p_tost"]]
    poder.to_csv(BASE_DIR / "pratica-02" / "resultados" / "poder_estatistico.csv", index=False)

    pd.set_option("display.width", 220)
    print("LINHA DE TETO -- abordagens que nao olham para X:")
    print(teto.round(4).to_string())
    melhor_com_x = (avaliacao[avaliacao["braco"] == "base"]
                    .groupby("abordagem")["spearman"].mean().max())
    print(f"\n  melhor meta-modelo sob `base`: {melhor_com_x:.4f}")
    print(f"  melhor sem olhar X:           {teto['spearman'].max():.4f}")
    if teto["spearman"].max() >= melhor_com_x - 0.01:
        print("  -> um ranking fixo empata com o melhor meta-modelo: a tarefa "
              "quase nao discrimina,\n     e a margem disponivel para qualquer "
              "meta-caracteristica e minuscula.")

    primaria = comparacao[comparacao["primaria"]]
    print("\nHIPOTESE PRIMARIA (declarada antes de rodar):")
    if len(primaria):
        r = primaria.iloc[0]
        print(f"  {r['braco']} vs base | {r['abordagem']} | {r['config']} | Spearman")
        print(f"  diferenca {r['diferenca']:+.4f}  IC95% [{r['ic_baixo']:+.4f}, "
              f"{r['ic_alto']:+.4f}]  p = {r['p']:.4f}")
        print(f"  detectavel com n=30: {r['efeito_min_detectavel']:.4f}  "
              f"| TOST(+-{MARGEM_TOST}): p = {r['p_tost']:.4f}")
    else:
        print("  ausente -- rodar avaliacao_x_estendido.py com os bracos de selecao")

    print("\nTODAS as comparacoes contra a base (Spearman):")
    sp = comparacao[comparacao["metrica"] == "spearman"]
    print(sp[["abordagem", "config", "braco", "media_base", "media_braco", "diferenca",
              "ic_baixo", "ic_alto", "p", "p_bh", "efeito_min_detectavel"]]
          .round(4).to_string(index=False))

    print("\nFriedman por abordagem:")
    print(friedman.round(4).to_string(index=False))

    print("\nPODER: o meta-dataset detecta diferencas de "
          f"{sp['efeito_min_detectavel'].min():.3f} a "
          f"{sp['efeito_min_detectavel'].max():.3f} em Spearman;")
    print(f"       as diferencas observadas vao de "
          f"{sp['diferenca'].abs().min():.3f} a {sp['diferenca'].abs().max():.3f}.")
    sd_tipico = sp["sd_pareado"].median()
    n_necessario = (sd_tipico * (norm.ppf(1 - ALFA / 2) + norm.ppf(PODER_ALVO)) / 0.01) ** 2
    print(f"       para detectar 0,010 com sd tipico {sd_tipico:.3f} seriam "
          f"necessarios ~{n_necessario:.0f} datasets.")
    equivalentes = int((sp["p_tost"] < ALFA).sum())
    print(f"       equivalencia dentro de +-{MARGEM_TOST} demonstrada em "
          f"{equivalentes}/{sp['p_tost'].notna().sum()} comparacoes.")


if __name__ == "__main__":
    main()
