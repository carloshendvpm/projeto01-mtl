"""Um caso por motivo do vocabulario fechado de indefinicoes.

Na primeira tentativa todo o mecanismo de NaN era latente: nenhum disparou nos
30 datasets, logo era codigo nunca exercitado. Aqui cada motivo tem o seu caso,
e o teste final garante que um erro de programacao NAO vira NaN silencioso.
"""

import numpy as np
import pytest

from complexidade import (
    MOTIVOS,
    ContextoGrupo,
    MedidaIndefinida,
    calcular_grupo,
    calcular_vizglobal,
    minmax,
    residuos_treino,
)


def ctx(Z, y):
    Z = np.asarray(Z, dtype=float)
    return ContextoGrupo(Z, np.asarray(y, dtype=float).ravel())


def test_grupo_vazio():
    r = calcular_grupo(ctx(np.empty((0, 3)), np.empty(0)))
    assert r.motivos["T2"] == "grupo_vazio"
    assert {r.motivos[m] for m in ("C1", "C2", "S1", "S3")} == {"grupo_vazio"}


def test_grupo_pequeno():
    rng = np.random.default_rng(0)
    r = calcular_grupo(ctx(rng.random((5, 2)), rng.random(5)))
    assert r.motivos["S1"] == "grupo_pequeno"
    assert r.motivos["S3"] == "grupo_pequeno"
    assert r.motivos["C1"] == "grupo_pequeno"
    assert np.isnan(r.valores["S1"])


def test_p_maior_que_n():
    """Mais atributos que exemplos: a CV nao cabe e L1 sai indefinida, mas a
    variante in-sample continua definida -- e daria residuo ~0, que era
    exatamente a armadilha da primeira tentativa."""
    rng = np.random.default_rng(1)
    Z, y = rng.random((15, 30)), rng.random(15)
    r = calcular_grupo(ctx(minmax(Z), minmax(y.reshape(-1, 1)).ravel()))
    assert r.motivos["L1"] == "p_maior_que_n"
    assert r.valores["L1_treino"] < 1e-9


def test_alvo_constante():
    rng = np.random.default_rng(2)
    Z = minmax(rng.random((20, 3)))
    with pytest.raises(MedidaIndefinida) as e:
        residuos_treino(ctx(Z, np.full(20, 0.5)))
    assert e.value.motivo == "alvo_constante"


def test_sem_atributo_valido():
    """Todas as colunas constantes: rho indefinido em todas, C1/C2 sem base."""
    Z = np.ones((20, 3))
    y = minmax(np.linspace(0, 1, 20).reshape(-1, 1)).ravel()
    r = calcular_grupo(ctx(Z, y))
    assert r.motivos["C1"] == "sem_atributo_valido"
    assert r.motivos["C2"] == "sem_atributo_valido"


def test_aresta_unica_no_vizglobal():
    """Mascara que isola pontos nao adjacentes na MST do dataset inteiro: nao
    sobra aresta interna suficiente para um desvio."""
    Z = minmax(np.linspace(0, 1, 40).reshape(-1, 1))
    y = minmax(np.linspace(0, 1, 40).reshape(-1, 1)).ravel()
    mascara = np.zeros(40, dtype=bool)
    mascara[[0, 20]] = True
    r = calcular_vizglobal(ctx(Z, y), mascara)
    assert r.motivos["S1_vizglobal"] == "aresta_unica"


def test_identico_ao_primario():
    """Com mascara toda verdadeira a variante global e algebricamente igual ao
    primario. Na primeira tentativa `S3_viz_global_all` e `S3_all` eram a mesma
    coluna e as duas entraram em X."""
    rng = np.random.default_rng(3)
    Z, y = minmax(rng.random((30, 2))), minmax(rng.random(30).reshape(-1, 1)).ravel()
    r = calcular_vizglobal(ctx(Z, y), np.ones(30, dtype=bool))
    for chave, motivo in r.motivos.items():
        assert motivo == "identico_ao_primario", chave
        assert np.isnan(r.valores[chave])


def test_todo_motivo_registrado_esta_no_vocabulario():
    with pytest.raises(ValueError):
        MedidaIndefinida("motivo_inventado")
    assert "p_maior_que_n" in MOTIVOS


def test_erro_de_programacao_nao_vira_nan():
    """O contrato que substitui o `except Exception: np.nan`: qualquer coisa
    que nao seja MedidaIndefinida propaga e aborta a execucao."""
    with pytest.raises(ValueError):
        calcular_grupo(ctx(np.ones((10, 2)), np.ones(5)))
