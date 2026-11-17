"""Meta-características do dataset alvo calculadas só no treino de cada fold externo.

Na primeira versão da Prática 4, a MtL e o Warm-start descreviam o dataset alvo
com Xbase e Xestendido calculados sobre o dataset inteiro. Vários descritores
dependem do alvo (assimetria e curtose de y, proporção de raros, complexidade
por grupo), então as observações do fold de teste influenciavam a recomendação.

Este script recalcula, para cada dataset e cada um dos dez folds externos, os
13 descritores de Xbase e os 51 de Xestendido usando apenas X[treino], y[treino],
com o mesmo código das Práticas 1 e 2 (`gerar_X.meta_features_de` e
`gerar_complexidade.processar_arrays`, sem as distribuições nulas, que eram
diagnóstico e não entram em X).

Os descritores dos outros 29 datasets continuam vindo das matrizes das práticas
anteriores: são meta-conhecimento offline, de datasets diferentes do alvo.

Saída: resultados/meta_por_fold.jsonl (uma linha por dataset × fold, com tempos)
Uso: .venv/bin/python meta_por_fold.py [--jobs 7] [--datasets ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import comum as c

sys.path.insert(0, str(c.BASE / "codigo"))
from gerar_X import meta_features_de  # noqa: E402
from gerar_complexidade import Cronometro, processar_arrays  # noqa: E402

warnings.filterwarnings("ignore")

MEDIDAS = ["C1", "C2", "C3", "C4", "L1", "L2", "L3", "S1", "S1_sd", "S2", "S3", "S4"]
COLUNAS_COMPLEXIDADE = [f"{m}_{g}" for g in ["all", "normal", "raro"] for m in MEDIDAS]
ARQUIVO = c.SAIDA / "meta_por_fold.jsonl"


def xbase(X: np.ndarray, y: np.ndarray) -> dict:
    return meta_features_de(pd.DataFrame(X), pd.Series(y))


def complexidade(nome: str, X: np.ndarray, y: np.ndarray) -> dict:
    linha = processar_arrays(nome, X, y, Cronometro(), sem_nulo=True)["X"]
    return {k: linha[k] for k in COLUNAS_COMPLEXIDADE}


def xestendido(base: dict, compl: dict) -> dict:
    """Mesma montagem da Prática 3: Xbase + 36 medidas por grupo + T2 por grupo."""
    linha = {**base, **compl}
    linha["T2_normal"] = base["instances_per_feature"] * (1 - base["pct_relevante"])
    linha["T2_raro"] = base["instances_per_feature"] * base["pct_relevante"]
    return linha


def rodar(nome: str, fold: int) -> dict:
    X, y = c.carregar(nome)
    tr, _ = c.folds(X)[fold]
    t0 = time.perf_counter()
    base = xbase(X[tr], y[tr])
    t1 = time.perf_counter()
    compl = complexidade(nome, X[tr], y[tr])
    t2 = time.perf_counter()
    linha = dict(dataset=nome, fold=fold, n_treino=len(tr), segundos_xbase=t1 - t0,
                 segundos_complexidade=t2 - t1, **xestendido(base, compl))
    with ARQUIVO.open("a") as f:
        f.write(json.dumps(linha, default=float) + "\n")
    return linha


def carregar_meta_por_fold() -> pd.DataFrame:
    linhas = [json.loads(l) for l in ARQUIVO.read_text().splitlines()]
    return pd.DataFrame(linhas).drop_duplicates(["dataset", "fold"], keep="last").set_index(["dataset", "fold"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--datasets", nargs="*")
    args = ap.parse_args()
    feitos = set()
    if ARQUIVO.exists():
        feitos = set(carregar_meta_por_fold().index)
    nomes = args.datasets or c.datasets()
    # os maiores primeiro, para equilibrar a fila
    tamanho = {n: len(pd.read_csv(c.DADOS / f"{n}.csv", usecols=[0])) for n in nomes}
    tarefas = [(n, f) for n in sorted(nomes, key=tamanho.get, reverse=True)
               for f in range(c.FOLDS) if (n, f) not in feitos]
    print(len(tarefas), "tarefas", flush=True)
    Parallel(n_jobs=args.jobs, verbose=5)(delayed(rodar)(n, f) for n, f in tarefas)


if __name__ == "__main__":
    main()
