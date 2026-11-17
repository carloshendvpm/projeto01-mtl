"""Calibracao da distribuicao nula -- portao obrigatorio do plano.

O controle de tamanho so tem valor se acertar os dois lados: nao acusar
diferenca onde o grupo e apenas pequeno, e acusar onde a regiao e de fato mais
dificil. Se falhar aqui, nenhum z calculado sobre os 30 datasets significa
coisa alguma.
"""

import numpy as np
import pytest

from complexidade import ContextoGrupo, calcular_grupo, minmax
from nulo_tamanho import distribuicao_nula, resumir

B_TESTE = 60
MEDIDAS = ["S1", "S3", "S2"]
N = 300
FRACAO_RARA = 0.1


def normalizar(Z, y):
    return minmax(Z), minmax(np.asarray(y, dtype=float).reshape(-1, 1)).ravel()


def dataset_homogeneo(seed):
    """y suave em Z por todo o espaco: nenhuma regiao e mais dificil."""
    rng = np.random.default_rng(seed)
    Z = rng.random((N, 3))
    y = Z @ np.array([1.0, 0.5, -0.25]) + 0.02 * rng.standard_normal(N)
    return normalizar(Z, y)


def z_do_grupo(Z, y, raro, tipo, medida, seed=0):
    observado = calcular_grupo(ContextoGrupo(Z[raro], y[raro]), MEDIDAS).valores[medida]
    nulos = distribuicao_nula(Z, y, raro, tipo, MEDIDAS, B=B_TESTE, seed=seed)
    return resumir(observado, nulos.get(medida))["z"]


@pytest.mark.parametrize("medida", MEDIDAS)
def test_grupo_sorteado_em_dataset_homogeneo_da_z_perto_de_zero(medida):
    """Raros sorteados ao acaso, nao por dificuldade: o observado E um sorteio
    da propria nula, logo |z| < 2 quase sempre. Se aqui desse z alto, o
    controle estaria acusando tamanho como se fosse dificuldade -- exatamente o
    artefato que ele existe para detectar."""
    fora = 0
    repeticoes = 20
    for seed in range(repeticoes):
        Z, y = dataset_homogeneo(seed)
        rng = np.random.default_rng(1000 + seed)
        raro = np.zeros(N, dtype=bool)
        raro[rng.choice(N, size=int(N * FRACAO_RARA), replace=False)] = True
        z = z_do_grupo(Z, y, raro, "qualquer", medida, seed=seed)
        fora += abs(z) >= 2
    assert fora <= 0.25 * repeticoes, f"{medida}: {fora}/{repeticoes} fora de |z|<2"


def test_regiao_genuinamente_dificil_da_z_alto():
    """Mesma geometria e mesmo tamanho de grupo, mas dentro da regiao marcada
    o alvo e ruido puro em vez de suave. A dificuldade e real e o controle tem
    de ve-la."""
    rng = np.random.default_rng(7)
    Z = rng.random((N, 3))
    y = Z @ np.array([1.0, 0.5, -0.25])
    dificil = Z[:, 0] > np.quantile(Z[:, 0], 1 - FRACAO_RARA)
    y[dificil] = rng.random(int(dificil.sum())) * 3.0
    Zn, yn = normalizar(Z, y)
    for medida in ("S1", "S3"):
        z = z_do_grupo(Zn, yn, dificil, "qualquer", medida)
        assert z > 2, f"{medida}: z={z:.2f}"


def test_os_tres_nulos_produzem_amostras_do_tamanho_do_grupo():
    Z, y = dataset_homogeneo(0)
    rng = np.random.default_rng(0)
    raro = np.zeros(N, dtype=bool)
    raro[rng.choice(N, size=30, replace=False)] = True
    for tipo in ("qualquer", "normal", "janela_y"):
        nulos = distribuicao_nula(Z, y, raro, tipo, ["S1"], B=10)
        assert len(nulos["S1"]) == 10, tipo
        assert np.all(np.isfinite(nulos["S1"])), tipo


def test_p_perm_e_bicaudal_e_nunca_zero():
    """Com B replicas o menor p possivel e 2/(B+1); um p exatamente zero
    indicaria contagem errada."""
    nulos = np.arange(100, dtype=float)
    assert resumir(-50.0, nulos)["p_perm"] == pytest.approx(2 / 101)
    assert resumir(150.0, nulos)["p_perm"] == pytest.approx(2 / 101)
    assert resumir(50.0, nulos)["p_perm"] > 0.5


def test_resumir_devolve_nan_sem_explodir():
    assert np.isnan(resumir(np.nan, np.arange(10.0))["z"])
    assert np.isnan(resumir(1.0, None)["z"])
    assert np.isnan(resumir(1.0, np.array([np.nan, np.nan]))["z"])
