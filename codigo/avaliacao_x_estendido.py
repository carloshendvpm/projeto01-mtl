"""Analise 3 do enunciado: efeito das colunas novas no sistema de recomendacao.

Mesmo meta-dataset, mesma matriz P, mesmo protocolo leave-one-dataset-out e as
mesmas metricas do Projeto 1 -- correlacao de Spearman e area da curva de
perda. So X muda, que e o que permite atribuir qualquer diferenca as novas
meta-caracteristicas.

Seis bracos, cada comparacao isolando uma coisa:

    base               13 colunas, o controle do Projeto 1
    +compl_all         controle indispensavel: complexidade ajuda, ou e a
                       separacao por grupo que ajuda?
    +compl_raro        a regiao rara ajuda?
    +ambos             a tripla bruta completa
    +compl_all_sel13   controle da selecao, com pool sem colunas de grupo
    +ambos_sel13       a selecao resolve o problema dimensional?

E a configuracao da floresta entra como FATOR cruzado, nao como escolha: tudo
roda sob `original` (defaults do Projeto 1) e sob `regularizada`. A hipotese e
a interacao configuracao x braco -- a regularizacao ajudar MAIS o braco
estendido. A configuracao e a mesma em todos os bracos; se variasse entre eles
seria confundidor, nao tratamento.

AR, MR e Vitorias Significativas ignoram X e sao constantes em todos os bracos.
Entram uma vez, como LINHA DE TETO: um ranking fixo, igual para os 30 datasets,
e o que qualquer meta-modelo precisa bater para justificar olhar para X.

Ressalva registrada sobre HARRIS: `HARRISForest` usa `max_features="sqrt"`,
logo ao passar de 13 para 49 colunas o seu mtry muda de 3 para 7. O modelo muda
de comportamento so por receber mais colunas, o que nao acontece com a floresta
do sklearn sob configuracao fixa.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "lib"))

from abordagens_agregacao import rank_medio, rank_mediano, vitorias_significativas  # noqa: E402
from abordagens_meta_modelo import (  # noqa: E402
    CONFIGS,
    abordagem1_regressor_sobre_P,
    abordagem2_regressor_sobre_R,
)
from avaliacao_leave_one_out import perda_por_t  # noqa: E402
from harris import harris  # noqa: E402
from montar_x import carregar, montar_variantes, pool_de_selecao  # noqa: E402
from selecao_meta_features import K_PADRAO, estabilidade, selecionar_filtro  # noqa: E402

LAMBDA_HARRIS = 0.5
SEED = 42
ABORDAGENS_SEM_X = ("AR", "MR", "Vitorias_Significativas")


def _metricas(ranking, P_test, R_test, colunas):
    ranking = ranking.reindex(colunas)
    rho, _ = spearmanr(ranking, R_test)
    perdas = perda_por_t(P_test, ranking.sort_values().index.tolist())
    return {"spearman": rho, "auc_perda": perdas.mean(),
            **{f"perda_t{i + 1}": v for i, v in enumerate(perdas)}}


def avaliar_rodada(dataset_teste, bracos, bracos_selecao, P, R, folds, args):
    P_train, R_train = P.drop(index=dataset_teste), R.drop(index=dataset_teste)
    P_test, R_test = P.loc[dataset_teste], R.loc[dataset_teste]
    folds_train = folds[folds["dataset"] != dataset_teste]

    linhas, selecionadas = [], []

    # Linha de teto: nao consultam X, logo sao identicas em todos os bracos.
    teto = {
        "AR": rank_medio(R_train),
        "MR": rank_mediano(R_train),
        "Vitorias_Significativas": vitorias_significativas(folds_train),
    }
    for nome, ranking in teto.items():
        linhas.append({"dataset": dataset_teste, "braco": "teto", "config": "-",
                       "abordagem": nome, **_metricas(ranking, P_test, R_test, R.columns)})

    todos = {**bracos, **bracos_selecao}
    for braco, X_braco in todos.items():
        X_train = X_braco.drop(index=dataset_teste)
        X_test = X_braco.loc[[dataset_teste]]

        if braco in bracos_selecao:
            colunas = selecionar_filtro(X_train, R_train, k=args.k)
            selecionadas.extend(
                {"dataset_teste": dataset_teste, "braco": braco, "coluna": c} for c in colunas
            )
            X_train, X_test = X_train[colunas], X_test[colunas]

        for config, rf in CONFIGS.items():
            previsoes = {
                "Abordagem_1": abordagem1_regressor_sobre_P(X_train, P_train, X_test, **rf),
                "Abordagem_2": abordagem2_regressor_sobre_R(X_train, R_train, X_test, **rf),
            }
            for nome, pred in previsoes.items():
                linhas.append({
                    "dataset": dataset_teste, "braco": braco, "config": config,
                    "abordagem": nome,
                    **_metricas(pred.loc[dataset_teste], P_test, R_test, R.columns),
                })

        if not args.sem_harris:
            pred = harris(X_train, P_train, R_train, X_test,
                          lam=LAMBDA_HARRIS, random_state=SEED)
            linhas.append({
                "dataset": dataset_teste, "braco": braco, "config": "propria",
                "abordagem": "HARRIS",
                **_metricas(pred.loc[dataset_teste], P_test, R_test, R.columns),
            })

    return linhas, selecionadas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sem-harris", action="store_true",
                        help="pula HARRIS, que e Python puro com splits exaustivos")
    parser.add_argument("--k", type=int, default=K_PADRAO,
                        help="colunas selecionadas dentro de cada fold")
    parser.add_argument("--limite", type=int, default=0,
                        help="so os N primeiros datasets, para cronometrar")
    parser.add_argument("--sufixo", default="")
    args = parser.parse_args()

    X, Xc = carregar()
    P = pd.read_csv(BASE_DIR / "meta-dataset" / "P.csv", index_col=0)
    R = pd.read_csv(BASE_DIR / "meta-dataset" / "R.csv", index_col=0)
    folds = pd.read_csv(BASE_DIR / "meta-dataset" / "resultados_por_fold.csv")

    bracos = montar_variantes(X, Xc)
    bracos_selecao = pool_de_selecao(X, Xc)
    print("bracos:", ", ".join(f"{n} ({d.shape[1]} col)"
                               for n, d in {**bracos, **bracos_selecao}.items()))

    datasets = list(X.index)[: args.limite] if args.limite else list(X.index)
    linhas, selecionadas = [], []
    for i, dataset_teste in enumerate(datasets, start=1):
        inicio = time.perf_counter()
        novas, sel = avaliar_rodada(dataset_teste, bracos, bracos_selecao, P, R, folds, args)
        linhas.extend(novas)
        selecionadas.extend(sel)
        print(f"[{i}/{len(datasets)}] {dataset_teste}: "
              f"{time.perf_counter() - inicio:.1f}s", flush=True)

    s = args.sufixo
    resultado = pd.DataFrame(linhas)
    resultado.to_csv(BASE_DIR / "pratica-02" / "resultados" / f"avaliacao_x_estendido{s}.csv", index=False)
    sel_df = pd.DataFrame(selecionadas)
    sel_df.to_csv(BASE_DIR / "pratica-02" / "resultados" / f"selecao_por_fold{s}.csv", index=False)

    resumo = (resultado.groupby(["abordagem", "braco", "config"])[["spearman", "auc_perda"]]
              .mean().sort_values("spearman", ascending=False))
    print("\nmedia sobre os rounds leave-one-dataset-out:")
    print(resumo.round(4).to_string())

    if len(sel_df):
        print("\ncolunas mais escolhidas dentro dos folds:")
        for braco in sel_df["braco"].unique():
            est = estabilidade(sel_df[sel_df["braco"] == braco])
            print(f"\n  {braco}:")
            print(est.head(8).round(2).to_string())


if __name__ == "__main__":
    main()
