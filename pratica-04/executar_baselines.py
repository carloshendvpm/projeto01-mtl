"""Reexecuta, nas mesmas partições do AutoML, os seis regressores padrão e a DESReg.

Saídas (uma linha por dataset × fold × método):
  resultados/base_por_fold.csv     seis regressores da matriz P, configuração padrão
  resultados/desreg_por_fold.csv   seleção dinâmica (DESReg) sobre o pool dos seis

Uso: .venv/bin/python executar_baselines.py [--jobs 6] [--datasets a1 a2 ...]
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import comum as c
from desReg.des.DESRegression import DESRegression
from desreg_cache import DESRegCache, parametros_phi

warnings.filterwarnings("ignore")

K = 5  # tamanho da região de competência: o padrão da DESReg, sem ajuste


def desreg() -> DESRegression:
    """Pool heterogêneo: bagging (10) de cada um dos seis regressores padrão = 60 modelos.

    Duas medidas de competência: `all_errors` (padrão da DESReg) e `phi_sse`,
    erro quadrático local ponderado pela distância e por 1 + phi(y), com phi
    estimada só no alvo de treino do fold.
    """
    pool = list(c.regressores(n_jobs=1).values())
    return DESRegression(regressors_list=pool, n_estimators_bag=10, random_state=c.SEED,
                         DSEL_perc=0.5, XTRAIN_full=True, n_jobs=1, k=K)


def rodar(nome: str) -> tuple[list[dict], list[dict]]:
    X, y = c.carregar(nome)
    base, des = [], []
    for fold, (tr, te) in enumerate(c.folds(X)):
        for alg, modelo in c.regressores(n_jobs=1).items():
            t0 = time.perf_counter()
            modelo.fit(X[tr], y[tr])
            pred = modelo.predict(X[te])
            base.append(dict(dataset=nome, fold=fold, metodo=alg, sera=c.sera(y[te], pred),
                             segundos=time.perf_counter() - t0))
        # O pool (bagging dos seis) e o DSEL são os da DESRegression; a seleção é
        # feita pelo cache vetorizado, equivalente ao predict original
        # (testes/test_desreg_cache.py), que é inviável nos datasets maiores.
        t0 = time.perf_counter()
        modelo = desreg().fit(X[tr], y[tr])
        cache = DESRegCache(modelo, X[te], k_max=K, params_phi=parametros_phi(y[tr]))
        t_comum = time.perf_counter() - t0
        for variante, competencia in [("DESReg", "all_errors"), ("DESReg_phi", "phi_sse")]:
            t1 = time.perf_counter()
            pred = cache.prever("DES", K, competencia, "mean", "mean")
            des.append(dict(dataset=nome, fold=fold, metodo=variante, sera=c.sera(y[te], pred),
                            segundos=t_comum + time.perf_counter() - t1))
    print(nome, "concluído", flush=True)
    return base, des


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--datasets", nargs="*")
    args = ap.parse_args()
    nomes = args.datasets or c.datasets()
    saida = Parallel(n_jobs=args.jobs)(delayed(rodar)(n) for n in nomes)
    sufixo = "" if not args.datasets else "_parcial"
    pd.DataFrame([r for b, _ in saida for r in b]).to_csv(c.SAIDA / f"base_por_fold{sufixo}.csv", index=False)
    pd.DataFrame([r for _, d in saida for r in d]).to_csv(c.SAIDA / f"desreg_por_fold{sufixo}.csv", index=False)


if __name__ == "__main__":
    main()
