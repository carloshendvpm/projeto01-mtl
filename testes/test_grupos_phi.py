"""Regressao contra o pipeline ja defendido do Projeto 1.

Se `pct_raro` divergir de `pct_relevante` em `X.csv`, a definicao de grupo
usada aqui nao e a mesma que gerou as meta-caracteristicas da base, e toda a
comparacao entre grupos passa a medir outra coisa.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from grupos_phi import (
    MAX_INSTANCIAS,
    carregar_dataset,
    diagnosticar,
    mascara_raro,
    mascaras_dos_grupos,
    replicas_subamostra,
)

RAIZ = Path(__file__).resolve().parent.parent
DATA = RAIZ / "data"


@pytest.fixture(scope="module")
def X_base():
    return pd.read_csv(RAIZ / "meta-dataset" / "X.csv", index_col=0)


@pytest.fixture(scope="module")
def grupos():
    saida = {}
    for csv in sorted(DATA.glob("*.csv")):
        Z, y = carregar_dataset(csv)
        raro = mascara_raro(y)
        saida[csv.stem] = (Z, y, raro)
    return saida


def test_pct_raro_reproduz_pct_relevante(grupos, X_base):
    for nome, (_, _, raro) in grupos.items():
        esperado = X_base.loc[nome, "pct_relevante"]
        assert raro.mean() == pytest.approx(esperado, abs=1e-12), nome


def test_decomposicao_por_cauda_reproduz_a_base(grupos, X_base):
    """`pct_relevante_alta` e `_baixa` ja existem em X e sao a base do
    diagnostico de bimodalidade; conferir que a particao e a mesma."""
    for nome, (Z, y, raro) in grupos.items():
        d = diagnosticar(Z, y, raro)
        assert d.raro_alta / len(y) == pytest.approx(
            X_base.loc[nome, "pct_relevante_alta"], abs=1e-12
        ), nome
        assert d.raro_baixa / len(y) == pytest.approx(
            X_base.loc[nome, "pct_relevante_baixa"], abs=1e-12
        ), nome


def test_todo_dataset_tem_raros(grupos):
    """Um dataset sem raros deixaria o grupo vazio e as medidas indefinidas.
    Se algum dia isso mudar, o pipeline tem de saber antes de rodar."""
    for nome, (_, _, raro) in grupos.items():
        assert raro.any(), nome
        assert not raro.all(), nome


def test_mascaras_particionam(grupos):
    for nome, (_, _, raro) in grupos.items():
        m = mascaras_dos_grupos(raro)
        assert np.array_equal(m["normal"] | m["raro"], m["all"]), nome
        assert not np.any(m["normal"] & m["raro"]), nome


def test_replicas_preservam_a_proporcao_de_raros(grupos):
    """Estratificacao pela propria condicao de raro: a subamostra tem de
    manter a proporcao, senao o grupo raro do dataset grande vira outro
    objeto."""
    grandes = [n for n, (_, y, _) in grupos.items() if len(y) > MAX_INSTANCIAS]
    assert grandes, "nenhum dataset acima do teto -- conferir MAX_INSTANCIAS"
    for nome in grandes:
        _, _, raro = grupos[nome]
        replicas = replicas_subamostra(raro)
        assert len(replicas) == 5
        for manter in replicas:
            assert abs(raro[manter].mean() - raro.mean()) < 0.02, nome
        assert not np.array_equal(replicas[0], replicas[1]), nome


def test_datasets_pequenos_nao_sao_subamostrados(grupos):
    for nome, (_, y, raro) in grupos.items():
        if len(y) <= MAX_INSTANCIAS:
            assert replicas_subamostra(raro) == [pytest.approx(np.ones(len(y), dtype=bool))]
