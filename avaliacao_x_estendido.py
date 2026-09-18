"""Leave-one-dataset-out com X base contra X estendido pelas medidas de complexidade.

Responde a pergunta que ficou pendente na Etapa 1 da Pratica 2: as medidas de
complexidade calculadas SO na regiao rara ajudam a recomendar algoritmo, ou o
ganho (se houver) vem das medidas de complexidade em si?

Por isso quatro variantes de X, e nao duas:

  base        os 13 descritores atuais (X.csv)
  +compl_all  base + as medidas de complexidade do dataset INTEIRO  <- controle
  +compl_raro base + as mesmas medidas calculadas so nos raros
  +ambos      base + as duas versoes

Sem o controle "+compl_all" nao daria para distinguir "a separacao por grupo
ajuda" de "medidas de complexidade ajudam".

A escolha das 8 medidas veio da analise de redundancia entre grupos
(complexidade-por-grupo.md, secao 4), que usa apenas o lado X. Nenhuma coluna foi
escolhida olhando P ou R, conforme a nota 4 da secao 6 do levantamento.

Saidas:
  avaliacao_x_estendido.csv     uma linha por (variante, dataset, abordagem)
  comparacao_x_estendido.csv    medias por variante e Wilcoxon contra a base
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from abordagens_agregacao import rank_medio, rank_mediano, vitorias_significativas
from abordagens_meta_modelo import abordagem1_regressor_sobre_P, abordagem2_regressor_sobre_R
from avaliacao_leave_one_out import perda_por_t
from harris import harris

BASE_DIR = Path(__file__).parent
LAMBDA_HARRIS = 0.5
SEED = 42

MEDIDAS = ["C4", "L1", "L2", "L3", "S1", "S1_sd", "S3", "S3_viz_global"]
# Abordagens que consultam X; as de agregacao (AR, MR, VS) ignoram X e por isso
# sao identicas nas quatro variantes -- servem de linha de referencia.
ABORDAGENS_COM_X = ("Abordagem_1", "Abordagem_2", "HARRIS")


def montar_variantes() -> dict[str, pd.DataFrame]:
    X = pd.read_csv(BASE_DIR / "X.csv", index_col=0)
    XC = pd.read_csv(BASE_DIR / "X_complexidade.csv", index_col=0).loc[X.index]
    cols_all = [f"{m}_all" for m in MEDIDAS]
    cols_raro = [f"{m}_raro" for m in MEDIDAS]
    return {
        "base": X,
        "+compl_all": X.join(XC[cols_all]),
        "+compl_raro": X.join(XC[cols_raro]),
        "+ambos": X.join(XC[cols_all + cols_raro]),
    }


def prever(dataset_teste, X, P, R, resultados_por_fold) -> dict[str, pd.Series]:
    X_train, X_test = X.drop(index=dataset_teste), X.loc[[dataset_teste]]
    P_train = P.drop(index=dataset_teste)
    R_train = R.drop(index=dataset_teste)
    treino_folds = resultados_por_fold[resultados_por_fold["dataset"] != dataset_teste]
    return {
        "AR": rank_medio(R_train),
        "MR": rank_mediano(R_train),
        "Vitorias_Significativas": vitorias_significativas(treino_folds),
        "Abordagem_1": abordagem1_regressor_sobre_P(X_train, P_train, X_test).loc[dataset_teste],
        "Abordagem_2": abordagem2_regressor_sobre_R(X_train, R_train, X_test).loc[dataset_teste],
        "HARRIS": harris(X_train, P_train, R_train, X_test,
                         lam=LAMBDA_HARRIS, random_state=SEED).loc[dataset_teste],
    }


def avaliar(variante, X, P, R, resultados_por_fold) -> list[dict]:
    linhas = []
    for dataset_teste in X.index:
        P_test, R_test = P.loc[dataset_teste], R.loc[dataset_teste]
        for nome, ranking in prever(dataset_teste, X, P, R, resultados_por_fold).items():
            ranking = ranking.reindex(R.columns)
            rho, _ = spearmanr(ranking, R_test)
            perdas = perda_por_t(P_test, ranking.sort_values().index.tolist())
            linhas.append({
                "variante": variante,
                "dataset": dataset_teste,
                "abordagem": nome,
                "spearman": rho,
                "auc_perda": perdas.mean(),
            })
    return linhas


def comparar_com_base(resultado: pd.DataFrame) -> pd.DataFrame:
    """Para cada abordagem e variante, media e Wilcoxon pareado contra a base."""
    linhas = []
    for abordagem, bloco in resultado.groupby("abordagem"):
        largo = bloco.pivot(index="dataset", columns="variante",
                            values=["spearman", "auc_perda"])
        for variante in ("base", "+compl_all", "+compl_raro", "+ambos"):
            linha = {"abordagem": abordagem, "variante": variante}
            for metrica in ("spearman", "auc_perda"):
                atual = largo[(metrica, variante)]
                base = largo[(metrica, "base")]
                linha[metrica] = atual.mean()
                linha[f"p_{metrica}"] = (
                    np.nan if variante == "base" or np.allclose(atual, base)
                    else wilcoxon(atual, base).pvalue
                )
            linhas.append(linha)
    return pd.DataFrame(linhas).set_index(["abordagem", "variante"])


def importancia_diagnostica(X: pd.DataFrame, R: pd.DataFrame) -> pd.Series:
    """Importancia das meta-caracteristicas numa floresta ajustada aos 30 datasets.

    Diagnostico descritivo, nao um teste: serve para ver se o meta-modelo sequer
    olha para as colunas novas. Ajustado no conjunto todo, sem leave-one-out.
    A propria Etapa 1 adverte que importancia nao garante ganho de recomendacao.
    """
    from sklearn.ensemble import RandomForestRegressor
    modelo = RandomForestRegressor(random_state=SEED, n_jobs=-1).fit(X, R)
    return pd.Series(modelo.feature_importances_,
                     index=X.columns).sort_values(ascending=False)


def main():
    P = pd.read_csv(BASE_DIR / "P.csv", index_col=0)
    R = pd.read_csv(BASE_DIR / "R.csv", index_col=0)
    resultados_por_fold = pd.read_csv(BASE_DIR / "resultados_por_fold.csv")

    linhas = []
    for variante, X in montar_variantes().items():
        print(f"... {variante} ({X.shape[1]} colunas)", flush=True)
        linhas.extend(avaliar(variante, X, P, R, resultados_por_fold))

    resultado = pd.DataFrame(linhas)
    resultado.to_csv(BASE_DIR / "avaliacao_x_estendido.csv", index=False)

    tabela = comparar_com_base(resultado)
    tabela.to_csv(BASE_DIR / "comparacao_x_estendido.csv")

    pd.set_option("display.width", 200)
    print("\n=== Media sobre os 30 rounds, e Wilcoxon pareado contra a base ===")
    print("(spearman: maior e melhor; auc_perda: menor e melhor)")
    print(tabela.loc[list(ABORDAGENS_COM_X)].round(4))
    print("\nAbordagens de agregacao (nao usam X, identicas nas variantes):")
    print(tabela.drop(index=list(ABORDAGENS_COM_X))
          .xs("base", level="variante")[["spearman", "auc_perda"]].round(4))
    imp = importancia_diagnostica(montar_variantes()["+ambos"], R)
    imp.rename("importancia").to_csv(BASE_DIR / "importancia_meta_features.csv")
    bloco = imp.groupby(lambda c: "novo_raro" if c.endswith("_raro")
                        else "novo_all" if c.endswith("_all") else "base").sum()
    print("\n=== Diagnostico: importancia na floresta (variante +ambos) ===")
    print(imp.head(8).round(4).to_string())
    print("\nSoma da importancia por bloco de colunas:")
    print(bloco.round(3).to_string())

    print("\nSalvo: avaliacao_x_estendido.csv, comparacao_x_estendido.csv,"
          " importancia_meta_features.csv")


if __name__ == "__main__":
    main()
