"""Fonte unica de verdade dos blocos e dos bracos da matriz X.

Todas as analises e o experimento importam daqui. Sem isto, a Analise 1, a
Analise 2 e a avaliacao divergem sobre "quais sao as colunas novas" -- e a
primeira tentativa chegou a ter duas colunas algebricamente identicas dentro do
mesmo braco sem que nada acusasse.

Convencao das colunas novas: `{medida}_{grupo}`, com grupo em all/normal/raro.
Para as medidas agregativas o valor e o da escala `_sq` (squashing v/(1+v)
aplicado elemento a elemento antes de agregar), porque sem isso L2 (~0,01) e S2
(~1,2) entrariam na mesma matriz com duas ordens de grandeza de diferenca. As
medidas de correlacao ja estao em [0,1] e entram brutas. T2 fica de fora: no
grupo raro degenera em contador de raros e duplica `pct_relevante`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "lib"))

from complexidade import MEDIDAS_EM_X  # noqa: E402
from grupos_phi import GRUPOS  # noqa: E402

LIMIAR_COLINEARIDADE = 0.99


def colunas_do_grupo(grupo: str) -> list[str]:
    return [f"{medida}_{grupo}" for medida in MEDIDAS_EM_X]


BLOCO_ALL = colunas_do_grupo("all")
BLOCO_NORMAL = colunas_do_grupo("normal")
BLOCO_RARO = colunas_do_grupo("raro")
BLOCO_NOVO = BLOCO_ALL + BLOCO_NORMAL + BLOCO_RARO


def carregar(base_dir: Path = BASE_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """X base do Projeto 1 e as colunas de complexidade, na mesma ordem."""
    X = pd.read_csv(base_dir / "meta-dataset" / "X.csv", index_col=0)
    Xc = pd.read_csv(base_dir / "meta-dataset" / "X_complexidade.csv", index_col=0).loc[X.index]
    return X, Xc


def origem_da_coluna(coluna: str) -> str:
    """base / all / normal / raro -- usado para colorir o dendrograma e para
    agregar importancias por bloco."""
    for grupo in GRUPOS:
        if coluna.endswith(f"_{grupo}"):
            return grupo
    return "base"


def montar_variantes(X: pd.DataFrame, Xc: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Os bracos que nao dependem de selecao dentro do fold.

    `+compl_all` e o controle indispensavel: sem ele nao da para distinguir
    "a separacao por grupo ajuda" de "complexidade ajuda".
    """
    variantes = {
        "base": X,
        "+compl_all": X.join(Xc[BLOCO_ALL]),
        "+compl_raro": X.join(Xc[BLOCO_RARO]),
        "+ambos": X.join(Xc[BLOCO_NOVO]),
    }
    for nome, df in variantes.items():
        checar_colinearidade(df, nome)
    return variantes


def pool_de_selecao(X: pd.DataFrame, Xc: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Bracos em que a selecao escolhe k colunas dentro de cada fold.

    `+compl_all_sel13` e o controle de `+ambos_sel13`: mesma selecao, mesmo k,
    pool sem as colunas de grupo. Sem ele, uma diferenca contra `base`
    confundiria "colunas novas" com "selecao".
    """
    return {
        "+compl_all_sel13": X.join(Xc[BLOCO_ALL]),
        "+ambos_sel13": X.join(Xc[BLOCO_NOVO]),
    }


def checar_colinearidade(df: pd.DataFrame, nome: str = "X") -> None:
    """Falha se duas colunas forem iguais ou quase.

    Guarda contra o defeito concreto da primeira tentativa, em que
    `S3_viz_global_all` era identicamente igual a `S3_all` e as duas entravam
    no mesmo braco. Para floresta isso e inocuo; para qualquer modelo linear,
    nao e.
    """
    numerico = df.select_dtypes(include=[np.number])
    variaveis = numerico.loc[:, numerico.std(numeric_only=True) > 0]
    corr = variaveis.corr(method="spearman").abs()
    problemas = [
        (a, b, corr.loc[a, b])
        for i, a in enumerate(corr.columns)
        for b in corr.columns[i + 1 :]
        if corr.loc[a, b] >= LIMIAR_COLINEARIDADE
    ]
    constantes = [c for c in numerico.columns if numerico[c].std(numeric_only=True) == 0]
    if problemas or constantes:
        detalhe = "; ".join(f"{a} ~ {b} (rho={r:.4f})" for a, b, r in problemas)
        raise ValueError(
            f"colinearidade em '{nome}': {detalhe or '-'} | constantes: {constantes or '-'}"
        )


def main():
    X, Xc = carregar()
    print(f"X base: {X.shape[1]} colunas | bloco novo: {len(BLOCO_NOVO)} colunas")
    for nome, df in montar_variantes(X, Xc).items():
        print(f"  {nome:14s} {df.shape[0]} x {df.shape[1]}")
    faltando = Xc[BLOCO_NOVO].isna().sum()
    if faltando.any():
        print("\ncolunas novas com NaN:")
        print(faltando[faltando > 0].to_string())


if __name__ == "__main__":
    main()
