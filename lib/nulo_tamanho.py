"""Distribuicoes nulas por tamanho de grupo.

O problema que este modulo existe para resolver. Ao calcular uma medida "so
nos raros", o grupo tem tipicamente 10% dos exemplos. Para toda medida que
dependa de vizinhanca ou de ajuste de modelo, isso muda o resultado por
geometria, e nao por dificuldade: uma MST sobre 10% dos pontos tem arestas
mais longas, logo |delta y| maior; um 1-NN com 10x menos candidatos tem
vizinhos mais distantes. Na primeira tentativa, S1, S1_sd e S3 foram
justamente as medidas que mais separaram raros de normais -- e os normais eram
~10x maiores que os raros, de modo que o teste pareado comparava tambem dois
tamanhos, nao so dois grupos.

A correcao e comparar o grupo raro contra o que se esperaria de QUALQUER grupo
daquele tamanho. Para cada dataset e medida sorteiam-se B subconjuntos de
tamanho exatamente |R|, processados pelo MESMO caminho de codigo do grupo raro,
e o valor observado e lido dentro dessa distribuicao.

Tres nulos, cada um isolando um confundidor diferente:

    qualquer  - |R| pontos entre todos os n. Isola so o tamanho.
    normal    - |R| pontos entre os normais. E o controle direto do teste
                raro x normal: "o grupo raro difere de uma regiao normal DO
                MESMO TAMANHO?", que e a pergunta que o Wilcoxon da primeira
                tentativa achava estar respondendo.
    janela_y  - janela contigua de |R| pontos na ordenacao de y. Isola tamanho
                mais localizacao em y: os raros nao sao so poucos, sao um bloco
                extremo do alvo.

Se o efeito sobrevive a `qualquer` e a `normal` mas morre em `janela_y`, a
conclusao correta nao e "os raros sao mais dificeis", e sim "o que importa e a
regiao ser localizada em y, nao ser rara".

Os resultados sao DIAGNOSTICO: entram na comparacao entre grupos e no
relatorio, e nao viram colunas de X.
"""

from __future__ import annotations

import numpy as np

from complexidade import ContextoGrupo, calcular_grupo

TIPOS = ("qualquer", "normal", "janela_y")
B_PADRAO = 200
SEED = 42


def _amostrar(tipo: str, n: int, tamanho: int, raro: np.ndarray, ordem_y, rng):
    """Uma mascara booleana de exatamente `tamanho` pontos, do tipo pedido."""
    if tipo == "qualquer":
        idx = rng.choice(n, size=tamanho, replace=False)
    elif tipo == "normal":
        candidatos = np.flatnonzero(~raro)
        if len(candidatos) < tamanho:
            return None
        idx = rng.choice(candidatos, size=tamanho, replace=False)
    elif tipo == "janela_y":
        inicio = int(rng.integers(0, n - tamanho + 1))
        idx = ordem_y[inicio : inicio + tamanho]
    else:
        raise ValueError(f"tipo de nulo desconhecido: {tipo!r}")
    mascara = np.zeros(n, dtype=bool)
    mascara[idx] = True
    return mascara


def distribuicao_nula(
    Z: np.ndarray,
    y: np.ndarray,
    raro: np.ndarray,
    tipo: str,
    medidas=None,
    B: int = B_PADRAO,
    seed: int = SEED,
) -> dict[str, np.ndarray]:
    """B valores de cada medida em subconjuntos de tamanho |R|.

    Z e y ja vem normalizados pelo dataset inteiro, como o grupo raro -- e
    essencial que o caminho seja o mesmo, senao a nula nao e comparavel com o
    observado.
    """
    n = len(y)
    tamanho = int(raro.sum())
    if tamanho == 0 or tamanho == n:
        return {}

    ordem_y = np.argsort(y, kind="stable")
    rng = np.random.default_rng(np.random.SeedSequence([seed, TIPOS.index(tipo)]))

    acumulado: dict[str, list[float]] = {}
    for _ in range(B):
        mascara = _amostrar(tipo, n, tamanho, raro, ordem_y, rng)
        if mascara is None:
            return {}
        valores = calcular_grupo(ContextoGrupo(Z[mascara], y[mascara]), medidas).valores
        for chave, valor in valores.items():
            acumulado.setdefault(chave, []).append(valor)
    return {chave: np.asarray(v, dtype=float) for chave, v in acumulado.items()}


def resumir(observado: float, nulos: np.ndarray) -> dict:
    """Posicao do valor observado dentro da distribuicao nula.

    Tres leituras, por ordem de robustez: `p_perm` nao supoe nada sobre a
    forma da nula; `pctl` tambem nao, e e o que se le no histograma; `z` supoe
    simetria aproximada e existe para permitir o teste pareado entre datasets,
    que precisa de uma escala comum.
    """
    validos = nulos[np.isfinite(nulos)] if nulos is not None else np.array([])
    if not np.isfinite(observado) or len(validos) < 2:
        return {
            "media_nula": np.nan, "sd_nula": np.nan, "z": np.nan,
            "pctl": np.nan, "p_perm": np.nan, "B_validos": len(validos),
        }

    media = float(validos.mean())
    sd = float(validos.std(ddof=1))
    # Nula degenerada: todo subconjunto de tamanho |R| da o mesmo valor. E o
    # caso de T2, que so depende do tamanho do grupo -- e a demonstracao de que
    # ela nao mede dificuldade. Sem esta tolerancia, um sd da ordem de 1e-17
    # produziria um z sem sentido em vez de um NaN honesto.
    if sd <= 1e-12 * max(1.0, abs(media)):
        sd = 0.0
    B = len(validos)
    acima = int(np.count_nonzero(validos >= observado))
    abaixo = int(np.count_nonzero(validos <= observado))
    p_unilateral = min((1 + acima) / (B + 1), (1 + abaixo) / (B + 1))

    return {
        "media_nula": media,
        "sd_nula": sd,
        "z": float((observado - media) / sd) if sd > 0 else np.nan,
        "pctl": float(abaixo / B),
        "p_perm": float(min(1.0, 2 * p_unilateral)),
        "B_validos": B,
    }


def avaliar_grupo_raro(
    Z: np.ndarray,
    y: np.ndarray,
    raro: np.ndarray,
    valores_observados: dict[str, float],
    medidas=None,
    tipos=TIPOS,
    B: int = B_PADRAO,
    seed: int = SEED,
) -> list[dict]:
    """Linhas longas (medida x tipo de nulo) para o grupo raro de um dataset."""
    linhas = []
    for tipo in tipos:
        nulos = distribuicao_nula(Z, y, raro, tipo, medidas, B, seed)
        for chave, observado in valores_observados.items():
            resumo = resumir(observado, nulos.get(chave))
            linhas.append(
                {"medida": chave, "nulo": tipo, "observado": observado, **resumo}
            )
    return linhas
