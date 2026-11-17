"""Confere as medidas contra o exemplo numerico da Secao 5 do levantamento.

Dois datasets 1-D de 10 pontos com os mesmos resumos marginais e dificuldades
locais diferentes: em A os dois casos relevantes sao adjacentes, em B ha um em
cada extremo. Os valores de C2, S1, S1_sd e S3 estao no documento; S2 e T2 sao
derivados aqui a mao, no docstring de cada teste.

Todas as expectativas sao sobre as medidas BRUTAS, as unicas comparaveis com o
documento e com Lorena et al. As `_sq` sao verificadas por invariancia de
ordem em test_invariancias.py.
"""

import numpy as np
import pytest

from complexidade import ContextoGrupo, calcular_grupo, minmax

Z = np.array([0, 1, 4, 5, 9, 10, 15, 16, 22, 23], dtype=float).reshape(-1, 1)
Y_A = np.array([0, 0, 0, 0, 0, 0, 0, 0, 10, 10], dtype=float)
Y_B = np.array([0, 10, 0, 0, 0, 0, 0, 0, 0, 10], dtype=float)

# phi manual do exemplo: phi(0) = 0, phi(10) = 1, logo R = {i : y_i = 10}
ESPERADO = {
    "A": {"C1": 0.696311, "C2": 0.696311, "S1": 1 / 9, "S1_sd": 1 / 3, "S3": 0.0,
          "S2": 1 / 9, "T2": 10.0},
    "B": {"C1": 0.087039, "C2": 0.087039, "S1": 3 / 9, "S1_sd": 0.5, "S3": 0.4,
          "S2": 65 / (9 * 23), "T2": 10.0},
}


def contexto(y):
    return ContextoGrupo(minmax(Z), minmax(y.reshape(-1, 1)).ravel())


@pytest.fixture(scope="module")
def resultados():
    return {nome: calcular_grupo(contexto(y)) for nome, y in (("A", Y_A), ("B", Y_B))}


@pytest.mark.parametrize("dataset", ["A", "B"])
def test_valores_do_documento(resultados, dataset):
    """C1/C2 (correlacao), S1/S1_sd (MST), S3 (1-NN), S2 e T2.

    S2 de A: a ordenacao por y e a identidade (empates estaveis), logo as
    distancias consecutivas em Z somam (23-0)/23 = 1 sobre 9 pares -> 1/9.
    S2 de B: a ordem fica [0,4,5,9,10,15,16,22,1,23]/23 e as diferencas
    absolutas somam 65/23 sobre 9 pares -> 65/(9*23).
    T2 = n/d = 10/1 nos dois (exemplos por dimensao, eq. 13 de Lorena).
    """
    obtido = resultados[dataset].valores
    for medida, esperado in ESPERADO[dataset].items():
        assert obtido[medida] == pytest.approx(esperado, abs=1e-6), medida


def test_os_raros_de_B_sao_mais_dificeis(resultados):
    """A hipotese que o exemplo existe para demonstrar: mesmos resumos
    marginais, dificuldade local diferente. B tem um relevante em cada
    extremo, logo MST e 1-NN pioram."""
    a, b = resultados["A"].valores, resultados["B"].valores
    assert b["S1"] > a["S1"]
    assert b["S3"] > a["S3"]
    assert b["C2"] < a["C2"]


def test_L_canonico_indefinido_com_p_proximo_de_n(resultados):
    """Com n=10 e p=1 a CV usa leave-one-out: 9 exemplos de treino contra o
    minimo exigido de p + 10 = 11. A medida canonica sai indefinida, com
    motivo, e a variante in-sample continua definida -- e este e justamente o
    caso em que a definicao literal enganaria."""
    for dataset in ("A", "B"):
        r = resultados[dataset]
        assert np.isnan(r.valores["L1"])
        assert r.motivos["L1"] == "p_maior_que_n"
        assert np.isfinite(r.valores["L1_treino"])


def test_medidas_dependentes_de_rng_sao_finitas(resultados):
    """L3 e S4 interpolam pontos com o gerador semeado; aqui basta que
    existam e que B nao seja mais facil que A."""
    a, b = resultados["A"].valores, resultados["B"].valores
    for medida in ("L3", "S4"):
        assert np.isfinite(a[medida]) and np.isfinite(b[medida])
    assert b["S4"] >= a["S4"]


def test_squash_preserva_ordem_entre_os_dois_datasets(resultados):
    """A escala `_sq` existe para comparar medidas entre si, e nao pode
    inverter a comparacao entre datasets."""
    a, b = resultados["A"].valores, resultados["B"].valores
    for medida in ("S1", "S3", "S2"):
        bruto = np.sign(b[medida] - a[medida])
        squash = np.sign(b[f"{medida}_sq"] - a[f"{medida}_sq"])
        assert bruto == squash, medida
