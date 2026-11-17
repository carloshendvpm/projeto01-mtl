"""Invariancias que as medidas devem respeitar.

Sao propriedades, nao valores: pegam erros de indexacao, de ordenacao e de
normalizacao que nenhum valor esperado isolado pegaria.
"""

import numpy as np
import pytest

from complexidade import MEDIDAS_META, ContextoGrupo, calcular_grupo, minmax

N, P = 60, 4


@pytest.fixture(scope="module")
def dados():
    """Sem empates em y: empates tornam S2/L3/S4 dependentes da ordem das
    linhas, o que e legitimo mas nao e invariancia."""
    rng = np.random.default_rng(7)
    Z = rng.random((N, P))
    y = Z @ np.array([1.0, -0.5, 0.25, 0.0]) + 0.1 * rng.random(N)
    return Z, y


def valores(Z, y):
    return calcular_grupo(ContextoGrupo(minmax(Z), minmax(y.reshape(-1, 1)).ravel())).valores


def comparar(a, b, exceto=()):
    for chave, valor in a.items():
        if chave in exceto or not np.isfinite(valor):
            continue
        assert b[chave] == pytest.approx(valor, rel=1e-9, abs=1e-12), chave


def test_permutacao_de_colunas(dados):
    """Distancias, MST e correlacoes agregadas nao dependem da ordem dos
    atributos. C3 e C4 podem depender: desempatam pelo indice do atributo."""
    Z, y = dados
    rng = np.random.default_rng(0)
    ordem = rng.permutation(P)
    comparar(valores(Z, y), valores(Z[:, ordem], y), exceto=("C3", "C4"))


# L1 e L2 canonicos usam validacao cruzada com embaralhamento semeado, logo a
# particao em folds depende da ordem das linhas. E dependencia real, nao bug:
# no pipeline a ordem vem do CSV e e fixa, portanto o valor e reprodutivel.
SENSIVEIS_A_ORDEM = ("L1", "L1_sq", "L2", "L2_sq")


def test_permutacao_de_linhas(dados):
    """Nenhuma medida depende da ordem dos exemplos, exceto as de CV."""
    Z, y = dados
    rng = np.random.default_rng(1)
    ordem = rng.permutation(N)
    comparar(valores(Z, y), valores(Z[ordem], y[ordem]), exceto=SENSIVEIS_A_ORDEM)


def test_L_por_cv_depende_da_ordem_mas_pouco(dados):
    """Documenta a excecao acima e limita o seu tamanho: a particao muda, o
    valor oscila, mas nao a ponto de mudar a leitura da medida."""
    Z, y = dados
    rng = np.random.default_rng(2)
    ordem = rng.permutation(N)
    a, b = valores(Z, y), valores(Z[ordem], y[ordem])
    for chave in ("L1", "L2"):
        assert a[chave] != b[chave]
        assert abs(a[chave] - b[chave]) < 0.2 * abs(a[chave])


def test_L_treino_nao_depende_da_ordem(dados):
    """A variante in-sample nao tem particao, logo e invariante -- util para
    separar uma diferenca de particao de um bug de indexacao."""
    Z, y = dados
    rng = np.random.default_rng(3)
    ordem = rng.permutation(N)
    a, b = valores(Z, y), valores(Z[ordem], y[ordem])
    assert b["L1_treino"] == pytest.approx(a["L1_treino"], rel=1e-9)


def test_reescala_afim_antes_do_minmax(dados):
    """Multiplicar e deslocar cada coluna nao muda nada depois do min-max --
    e por isso que a normalizacao pode acontecer uma vez, no dataset inteiro."""
    Z, y = dados
    Z2 = Z * np.array([3.0, 0.5, 100.0, 2.0]) + np.array([-1.0, 7.0, 0.0, 5.0])
    comparar(valores(Z, y), valores(Z2, y * 1000 - 42))


def test_minmax_idempotente(dados):
    Z, _ = dados
    assert minmax(minmax(Z)) == pytest.approx(minmax(Z))


def test_minmax_zera_coluna_constante():
    Z = np.column_stack([np.linspace(0, 1, 10), np.full(10, 3.0)])
    assert np.all(minmax(Z)[:, 1] == 0.0)


def test_squash_existe_para_toda_medida_agregativa(dados):
    """Contrato do registro: agregativa tem `_sq`, escalar nao tem."""
    Z, y = dados
    v = valores(Z, y)
    for nome, medida in MEDIDAS_META.items():
        if medida.agregativa:
            assert f"{nome}_sq" in v, nome
        else:
            assert f"{nome}_sq" not in v, nome


def test_squash_fica_em_zero_um(dados):
    """A escala `_sq` existe para por medidas de ordens de grandeza
    diferentes na mesma faixa; se escapar de [0,1) nao serve para isso."""
    Z, y = dados
    for chave, valor in valores(Z, y).items():
        if chave.endswith("_sq") and np.isfinite(valor):
            assert 0.0 <= valor < 1.0, chave
