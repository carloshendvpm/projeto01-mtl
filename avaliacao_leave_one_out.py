from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from abordagens_agregacao import rank_medio, rank_mediano, vitorias_significativas
from abordagens_meta_modelo import abordagem1_regressor_sobre_P, abordagem2_regressor_sobre_R
from harris import harris

BASE_DIR = Path(__file__).parent
LAMBDAS_HARRIS = [0.0, 0.25, 0.5, 0.75, 1.0]
LAMBDA_OFICIAL = 0.5


def perda_por_t(P_dataset: pd.Series, ordem: list) -> np.ndarray:
    melhor_possivel = P_dataset.min()
    pior_possivel = P_dataset.max()
    faixa = pior_possivel - melhor_possivel if pior_possivel > melhor_possivel else 1.0
    valores_ordenados = P_dataset[ordem].to_numpy()
    melhor_ate_t = np.minimum.accumulate(valores_ordenados)
    return (melhor_ate_t - melhor_possivel) / faixa


def avaliar_rodada(dataset_teste, X, P, R, resultados_por_fold):
    X_train, X_test = X.drop(index=dataset_teste), X.loc[[dataset_teste]]
    P_train = P.drop(index=dataset_teste)
    R_train = R.drop(index=dataset_teste)
    P_test = P.loc[dataset_teste]
    R_test = R.loc[dataset_teste]
    resultados_train = resultados_por_fold[resultados_por_fold["dataset"] != dataset_teste]

    previsoes = {
        "AR": rank_medio(R_train),
        "MR": rank_mediano(R_train),
        "Vitorias_Significativas": vitorias_significativas(resultados_train),
        "Abordagem_1": abordagem1_regressor_sobre_P(X_train, P_train, X_test).loc[dataset_teste],
        "Abordagem_2": abordagem2_regressor_sobre_R(X_train, R_train, X_test).loc[dataset_teste],
    }
    for lam in LAMBDAS_HARRIS:
        pred = harris(X_train, P_train, R_train, X_test, lam=lam, random_state=42)
        nome = "HARRIS" if lam == LAMBDA_OFICIAL else f"HARRIS_lambda={lam}"
        previsoes[nome] = pred.loc[dataset_teste]

    linhas = []
    for nome, ranking_previsto in previsoes.items():
        ranking_previsto = ranking_previsto.reindex(R.columns)
        rho, _ = spearmanr(ranking_previsto, R_test)
        ordem = ranking_previsto.sort_values().index.tolist()
        perdas = perda_por_t(P_test, ordem)
        linha = {"dataset": dataset_teste, "abordagem": nome, "spearman": rho, "auc_perda": perdas.mean()}
        linha.update({f"perda_t{i + 1}": v for i, v in enumerate(perdas)})
        linhas.append(linha)
    return linhas


def main():
    X = pd.read_csv(BASE_DIR / "X.csv", index_col=0)
    P = pd.read_csv(BASE_DIR / "P.csv", index_col=0)
    R = pd.read_csv(BASE_DIR / "R.csv", index_col=0)
    resultados_por_fold = pd.read_csv(BASE_DIR / "resultados_por_fold.csv")

    todas_linhas = []
    for i, dataset_teste in enumerate(X.index, start=1):
        print(f"[{i}/{len(X.index)}] {dataset_teste}")
        todas_linhas.extend(avaliar_rodada(dataset_teste, X, P, R, resultados_por_fold))

    resultado = pd.DataFrame(todas_linhas)
    resultado.to_csv(BASE_DIR / "avaliacao_por_dataset.csv", index=False)

    resumo = resultado.groupby("abordagem")[["spearman", "auc_perda"]].mean().sort_values("spearman", ascending=False)
    print("\nResumo (média sobre os 30 rounds leave-one-out):")
    print(resumo)


if __name__ == "__main__":
    main()
