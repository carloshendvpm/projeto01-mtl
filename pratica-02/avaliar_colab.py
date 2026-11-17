"""Avaliação LODO das representações do notebook, sem executar ao importar."""
from pathlib import Path
from time import perf_counter
import json
import platform

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr

from abordagens_agregacao import rank_medio, rank_mediano, vitorias_significativas
from abordagens_meta_modelo import abordagem1_regressor_sobre_P, abordagem2_regressor_sobre_R
from harris import harris


def preparar_fold(treino, teste, ranks_treino, selecionar=False, k=13):
    """Medianas, constantes e escores são obtidos exclusivamente no treino."""
    medianas = treino.median()
    utilizaveis = medianas.index[medianas.notna()]
    if not len(utilizaveis):
        raise ValueError("Treino sem atributos utilizáveis.")
    tr = treino[utilizaveis].fillna(medianas[utilizaveis])
    te = teste[utilizaveis].fillna(medianas[utilizaveis])
    escores = pd.Series(np.nan, index=utilizaveis, dtype=float)
    if selecionar:
        variaveis = tr.columns[tr.nunique() > 1]
        if not len(variaveis):
            raise ValueError("Seleção sem atributos variáveis no treino.")
        for coluna in variaveis:
            valores = []
            for algoritmo in ranks_treino.columns:
                alvo = ranks_treino[algoritmo]
                if alvo.nunique() > 1:
                    valores.append(abs(float(spearmanr(tr[coluna], alvo).statistic)))
            escores[coluna] = np.mean(valores) if valores else 0.0
        melhores = escores.loc[variaveis].sort_values(
            ascending=False, kind="stable"
        ).head(k).index
        # Selecionar não deve reordenar as colunas: isso altera a associação
        # entre índices de atributos e sorteios internos dos modelos de árvore.
        escolhidas = utilizaveis[utilizaveis.isin(melhores)]
    else:
        escolhidas = utilizaveis
    auditoria = [
        {"meta_feature": c, "mediana_treino": float(medianas[c]),
         "escore_selecao": float(escores[c]), "selecionada": c in escolhidas}
        for c in utilizaveis
    ]
    return tr[escolhidas], te[escolhidas], auditoria


def medir_ranking(previsto, desempenho, real):
    previsto = previsto.reindex(real.index)
    if not np.isfinite(previsto.to_numpy()).all():
        raise ValueError("Previsão de rank não finita.")
    constante = previsto.nunique() < 2 or real.nunique() < 2
    rho = np.nan if constante else float(spearmanr(previsto, real).statistic)
    # Ordem fixa de algoritmos desempata as recomendações de modo reprodutível.
    ordem = previsto.sort_values(kind="stable").index
    melhor = float(desempenho.min())
    amplitude = float(desempenho.max() - melhor)
    perdas = (np.minimum.accumulate(desempenho.loc[ordem].to_numpy()) - melhor)
    perdas = perdas / (amplitude if amplitude > 0 else 1.0)
    # Mesma definição discreta de área usada na Atividade 1 (não trapézios).
    saida = {"spearman": rho, "spearman_definido": not constante,
             "auc_perda": float(perdas.mean()),
             "ordem_recomendada": "|".join(ordem)}
    saida.update({f"perda_t{i+1}": float(v) for i, v in enumerate(perdas)})
    saida.update({f"rank_previsto_{a}": float(previsto[a]) for a in real.index})
    return saida


def avaliar_representacoes(representacoes, P, R, resultados_folds,
                          pasta="resultados/avaliacao_lodo", seed=42,
                          tratamentos=(False, True)):
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    tratamentos = tuple(tratamentos)
    if not tratamentos or any(type(t) is not bool for t in tratamentos):
        raise ValueError("tratamentos deve conter False e/ou True.")
    algoritmos = list(P.columns)
    assert P.index.is_unique and R.index.is_unique
    assert set(P.index) == set(R.index) and set(P.columns) == set(R.columns)
    R = R.loc[P.index, algoritmos]
    assert np.isfinite(P.to_numpy()).all() and np.isfinite(R.to_numpy()).all()
    np.testing.assert_allclose(R.to_numpy(), P.rank(axis=1, method="average").to_numpy())
    assert set(resultados_folds["dataset"]) == set(P.index)
    for matriz in representacoes.values():
        assert matriz.index.is_unique and matriz.columns.is_unique
        assert set(matriz.index) == set(P.index)
        assert not np.isinf(matriz.to_numpy(dtype=float)).any()

    metadados = {
        "protocolo": "leave-one-dataset-out", "seed": seed,
        "versao_avaliador": 2,
        "tratamentos": list(tratamentos),
        "ordem_atributos": "ordem original preservada após seleção",
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "dimensoes": {n: list(x.shape) for n, x in representacoes.items()},
        "selecao": "até 13; média de |Spearman| com os seis ranks no treino após imputação",
        "imputacao": "mediana do treino",
        "harris_lambda": 0.5,
        "rf": {"n_estimators": 100, "max_features": 1.0,
               "min_samples_leaf": 1, "max_depth": None},
        "area": "média das perdas normalizadas para t=1..6",
        "desempate_recomendacao": algoritmos,
    }
    (pasta / "protocolo.json").write_text(
        json.dumps(metadados, ensure_ascii=False, indent=2), encoding="utf-8")
    linhas, auditorias = [], []
    inicio_total = perf_counter()
    for rodada, dataset in enumerate(P.index, 1):
        inicio = perf_counter()
        print(f"[{rodada}/{len(P)}] {dataset}: iniciando", flush=True)
        p_tr = P.drop(index=dataset)
        r_tr = R.drop(index=dataset)
        folds_tr = resultados_folds.loc[resultados_folds["dataset"] != dataset]
        referencias = {
            "AR": rank_medio(r_tr), "MR": rank_mediano(r_tr),
            "Vitorias_Significativas": vitorias_significativas(folds_tr),
        }
        for abordagem, previsto in referencias.items():
            linhas.append({"dataset": dataset, "representacao": "independente_de_X",
                           "tratamento": "referencia", "abordagem": abordagem,
                           "n_atributos": 0,
                           **medir_ranking(previsto, P.loc[dataset], R.loc[dataset])})
        for nome, matriz in representacoes.items():
            matriz = matriz.loc[P.index]
            for selecionar in tratamentos:
                tratamento = "selecao_13" if selecionar else "sem_selecao"
                tr, te, aud = preparar_fold(
                    matriz.drop(index=dataset), matriz.loc[[dataset]], r_tr,
                    selecionar=selecionar)
                auditorias.extend({"dataset_teste": dataset, "representacao": nome,
                                    "tratamento": tratamento, **a} for a in aud)
                rf = {"n_estimators": 100, "max_features": 1.0,
                      "min_samples_leaf": 1, "max_depth": None}
                previsoes = {
                    "Abordagem_1": abordagem1_regressor_sobre_P(tr, p_tr, te, **rf),
                    "Abordagem_2": abordagem2_regressor_sobre_R(tr, r_tr, te, **rf),
                    "HARRIS": harris(tr, p_tr, r_tr, te, lam=0.5, random_state=seed),
                }
                for abordagem, previsto in previsoes.items():
                    linhas.append({"dataset": dataset, "representacao": nome,
                                   "tratamento": tratamento, "abordagem": abordagem,
                                   "n_atributos": tr.shape[1],
                                   **medir_ranking(previsto.loc[dataset],
                                                  P.loc[dataset], R.loc[dataset])})
        # Arquivos parciais preservam rodadas completas; nova chamada reinicia o experimento.
        pd.DataFrame(linhas).to_csv(pasta / "resultados_por_dataset.csv", index=False)
        pd.DataFrame(auditorias).to_csv(pasta / "auditoria_preprocessamento.csv", index=False)
        print(f"  concluído em {perf_counter()-inicio:.1f} s", flush=True)

    resultado = pd.DataFrame(linhas)
    chaves = ["representacao", "tratamento", "abordagem"]
    resumo = resultado.groupby(chaves, sort=False).agg(
        n_datasets=("dataset", "size"), spearman_medio=("spearman", "mean"),
        n_spearman_validos=("spearman", "count"),
        auc_perda_media=("auc_perda", "mean"),
        n_atributos_medio=("n_atributos", "mean"),
    ).reset_index()
    resumo.to_csv(pasta / "resumo.csv", index=False)
    colunas_perda = [f"perda_t{i+1}" for i in range(len(algoritmos))]
    resultado.groupby(chaves)[colunas_perda].mean().to_csv(pasta / "curvas_perda.csv")
    segundos = perf_counter()-inicio_total
    (pasta / "tempo_avaliacao_s.txt").write_text(f"{segundos:.6f}\n")
    print(f"Concluído: {len(resultado)} avaliações em {segundos/60:.2f} min.", flush=True)
    return resultado, resumo
