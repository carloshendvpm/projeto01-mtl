"""Casos construidos para C3 e C4.

Sao as duas medidas com algoritmo iterativo -- remocao gulosa ate correlacionar
e cobertura coletiva por atributo -- e as duas que a primeira tentativa deixou
sem nenhum teste. Aqui cada caso tem um valor ou uma ordenacao derivavel a mao.
"""

import numpy as np
import pytest

from complexidade import C3, C4, ContextoGrupo, minmax

N = 40


def ctx(Z, y):
    return ContextoGrupo(minmax(np.asarray(Z, dtype=float)), minmax(np.asarray(y, dtype=float).reshape(-1, 1)).ravel())


def monotono(n=N):
    z = np.arange(n, dtype=float).reshape(-1, 1)
    return z, z.ravel().copy()


def test_c3_zero_quando_y_e_monotono_no_atributo():
    """|rho| = 1 sem remover nada, logo C3 = 0."""
    Z, y = monotono()
    assert C3(ctx(Z, y)) == pytest.approx(0.0)


@pytest.mark.parametrize("k", [1, 2, 4])
def test_c3_cresce_com_o_numero_de_pontos_discordantes(k):
    """Trocar k pares nos extremos exige remover exemplos para voltar a |rho|
    >= 0.9, e quanto mais trocas, mais se remove."""
    Z, y = monotono()
    y_trocado = y.copy()
    for i in range(k):
        y_trocado[i], y_trocado[-1 - i] = y_trocado[-1 - i], y_trocado[i]
    c3 = C3(ctx(Z, y_trocado))
    assert c3 > 0
    assert c3 >= k / N - 1e-9


def test_c3_maior_sob_independencia():
    """Sem associacao nenhuma, remover poucos exemplos nao produz |rho| >= 0.9."""
    rng = np.random.default_rng(0)
    Z = rng.random((N, 1))
    y = rng.random(N)
    Zm, ym = monotono()
    assert C3(ctx(Z, y)) > C3(ctx(Zm, ym))


def test_c4_quase_zero_quando_um_atributo_explica_tudo():
    """y = z1 exato: o ajuste linear simples zera o residuo de todos, e nao
    sobra exemplo por explicar."""
    Z, y = monotono()
    assert C4(ctx(Z, y)) < 0.1


def test_c4_alto_sob_ruido_independente():
    rng = np.random.default_rng(1)
    Z = rng.random((N, 2))
    y = rng.random(N)
    assert C4(ctx(Z, y)) > 0.5


def test_c4_e_coletiva_nao_marginal():
    """Metade de y depende de z1 e metade de z2, com cada atributo constante
    onde nao manda. Usando os dois, C4 tem de ficar menor do que usando so o
    primeiro -- e a diferenca entre eficiencia coletiva e marginal."""
    n = N
    metade = n // 2
    z1 = np.concatenate([np.linspace(0, 1, metade), np.zeros(n - metade)])
    z2 = np.concatenate([np.zeros(metade), np.linspace(0, 1, n - metade)])
    y = np.concatenate([np.linspace(0, 1, metade), np.linspace(0, 1, n - metade)])
    juntos = C4(ctx(np.column_stack([z1, z2]), y))
    so_z1 = C4(ctx(z1.reshape(-1, 1), y))
    assert juntos < so_z1
