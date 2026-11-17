"""Isolamento do fold externo na MtL e no Warm-start.

A ideia de cada teste: trocar os valores de teste (X e y) do fold por lixo e
conferir que nada do que decide o modelo ou os hiperparâmetros muda.
"""
import numpy as np
import pandas as pd
import pytest

import comum as c
import executar_warmstart as ws
import meta_por_fold as mpf
import mtl

NOME = "boston"


def _corromper_teste(X, y, te, seed=0):
    rng = np.random.default_rng(seed)
    X2, y2 = X.copy(), y.copy()
    X2[te] = rng.normal(size=X2[te].shape) * 1e3
    y2[te] = rng.normal(size=te.size) * 1e3 + y.max() * 10
    return X2, y2


def _descritor(X, y, tr, nome):
    base = mpf.xbase(X[tr], y[tr])
    return pd.Series(mpf.xestendido(base, mpf.complexidade(nome, X[tr], y[tr])))


@pytest.fixture(scope="module")
def dados():
    X, y = c.carregar(NOME)
    tr, te = c.folds(X)[0]
    return X, y, tr, te


@pytest.fixture(scope="module")
def descritores(dados):
    X, y, tr, te = dados
    X2, y2 = _corromper_teste(X, y, te)
    return _descritor(X, y, tr, NOME), _descritor(X2, y2, tr, NOME)


def test_xbase_reproduz_pratica_1():
    """A refatoração de gerar_X não mudou os descritores calculados no dataset inteiro."""
    X0 = pd.read_csv(c.META / "X.csv", index_col=0)
    X, y = c.carregar(NOME)
    calc = pd.Series(mpf.xbase(X, y))[X0.columns].astype(float)
    np.testing.assert_allclose(calc, X0.loc[NOME].astype(float), rtol=1e-9)


def test_descritores_ignoram_o_teste(descritores):
    original, corrompido = descritores
    pd.testing.assert_series_equal(original, corrompido)


def test_descritores_do_fold_diferem_do_dataset_inteiro(descritores):
    """Sanidade: o descritor do fold não é uma cópia do calculado com o teste."""
    X0 = pd.read_csv(c.META / "X.csv", index_col=0)
    assert descritores[0]["n_instances"] < X0.loc[NOME, "n_instances"]
    assert descritores[0]["y_skewness"] != pytest.approx(X0.loc[NOME, "y_skewness"], rel=1e-9)


def test_mtl_nao_usa_o_alvo_nem_o_teste(descritores):
    Xbase = pd.read_csv(c.META / "X.csv", index_col=0)
    R = pd.read_csv(c.META / "R.csv", index_col=0)
    original, corrompido = descritores
    rec = mtl.recomendar(Xbase, R, NOME, original)
    assert rec == mtl.recomendar(Xbase, R, NOME, corrompido)
    # a linha do alvo em Xbase e R (calculada com o dataset inteiro) não pode influenciar
    Xb2, R2 = Xbase.copy(), R.copy()
    Xb2.loc[NOME] = 1e9
    R2.loc[NOME] = R2.loc[NOME].iloc[::-1].to_numpy()
    assert rec == mtl.recomendar(Xb2, R2, NOME, original)


def test_vizinhos_do_warmstart_ignoram_o_teste_e_o_alvo(descritores):
    meta = ws.xestendido()
    original, corrompido = descritores
    v1 = ws.vizinhos(meta.drop(index=NOME), original)
    v2 = ws.vizinhos(meta.drop(index=NOME), corrompido)
    pd.testing.assert_series_equal(v1, v2)
    assert NOME not in v1.index


def test_busca_interna_nao_ve_o_teste(dados):
    """A busca recebe só o treino; corromper o teste não muda a configuração escolhida."""
    X, y, tr, te = dados
    X2, y2 = _corromper_teste(X, y, te)
    ini = ["c000", "c011", "c022", "c033", "c044"]
    a = ws.buscar(X[tr], y[tr], ini, seed=0)
    b = ws.buscar(X2[tr], y2[tr], ini, seed=0)
    assert a[0] == b[0] and a[2] == b[2]


def test_warmstart_reaproveitado():
    """Folds reaproveitados da versão 1: refazer a busca dá a mesma SERA."""
    novo = pd.read_csv(c.SAIDA / "warmstart_por_fold.csv")
    reap = novo[novo.reaproveitado].query("dataset in ['a1', 'boston', 'cpu_small']").head(2)
    if reap.empty:
        pytest.skip("nenhum fold reaproveitado nesses datasets")
    for _, r in reap.iterrows():
        X, y = c.carregar(r.dataset)
        tr, te = c.folds(X)[r.fold]
        melhor, _, ordem = ws.buscar(X[tr], y[tr], r.avaliadas.split(",")[:ws.N_INIT], seed=0)
        assert ",".join(ordem) == r.avaliadas and melhor == r.config
        m = ws.RandomForestRegressor(**ws.CONFIGS[melhor], n_estimators=100, random_state=c.SEED, n_jobs=1)
        m.fit(X[tr], y[tr])
        assert c.sera(y[te], m.predict(X[te])) == pytest.approx(r.sera, rel=1e-12)
