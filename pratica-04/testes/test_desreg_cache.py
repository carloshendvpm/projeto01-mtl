"""O cache precisa prever exatamente o que a DESRegression original prevê.

É o portão da Prática 3: a matriz de perdas inteira sai do cache, então se ele
divergir da biblioteca, nada do que vem depois mede a DESReg.
"""

import itertools

import numpy as np
import pandas as pd
import pytest
from desReg.des.DESRegression import DESRegression
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from comum import DADOS as DATA_DIR
from desreg_cache import (AGREGACAO, COMPETENCIAS, SELECAO, DESRegCache,
                          medida_desreg, parametros_phi)

N_TESTE = 12  # a DESReg original é lenta: poucas instâncias bastam para comparar


def preparar(nome, pool, seed=0):
    df = pd.read_csv(DATA_DIR / f"{nome}.csv")
    y, X = df.iloc[:, 0].to_numpy(float), df.iloc[:, 1:].to_numpy(float)
    rng = np.random.default_rng(seed)
    ordem = rng.permutation(len(y))
    teste, treino = ordem[:N_TESTE], ordem[N_TESTE:]
    escala = StandardScaler().fit(X[treino])
    X_tr, X_te = escala.transform(X[treino]), escala.transform(X[teste])

    regs = None if pool == "dt" else [DecisionTreeRegressor(), KNeighborsRegressor(), Ridge()]
    des = DESRegression(regressors_list=regs, n_estimators_bag=3, random_state=seed,
                        n_jobs=1).fit(X_tr, y[treino])
    params_phi = parametros_phi(y[treino])
    return des, X_te, params_phi


@pytest.mark.parametrize("nome,pool", [("a1", "dt"), ("triazines", "het")])
def test_cache_igual_desreg(nome, pool):
    des, X_te, params_phi = preparar(nome, pool)
    cache = DESRegCache(des, X_te, k_max=7, params_phi=params_phi)

    for k, comp in itertools.product([3, 7], COMPETENCIAS):
        base = dict(k=k, competence_level=medida_desreg(comp, params_phi))

        esperado = np.ravel(des.predict(X_te, params={**base, "ensemble_type": "DRS"}))
        obtido = cache.prever("DRS", k, comp)
        np.testing.assert_allclose(obtido, esperado, rtol=1e-8, atol=1e-8,
                                   err_msg=f"DRS k={k} {comp}")

        for sel, agg in itertools.product(SELECAO, AGREGACAO):
            esperado = np.ravel(des.predict(X_te, params={
                **base, "ensemble_type": "DES",
                "regressor_selection": SELECAO[sel], "aggregation_method": AGREGACAO[agg]}))
            obtido = cache.prever("DES", k, comp, sel, agg)
            # a ponderação 1/competência transforma ruído de último bit numa
            # competência ~0 em peso ~1e10; a própria DESReg é instável aí
            tol = 1e-5 if agg == "weighted" else 1e-8
            np.testing.assert_allclose(obtido, esperado, rtol=tol, atol=1e-8,
                                       err_msg=f"DES k={k} {comp} {sel} {agg}")
