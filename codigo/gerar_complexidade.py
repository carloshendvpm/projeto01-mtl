"""Extrai as medidas de complexidade por grupo nos datasets de `data/`.

Para cada dataset, cada medida sai tres vezes -- no conjunto todo, so nos
normais e so nos raros -- conforme a sugestao da professora. A ordem das
operacoes e a de `lib/grupos_phi.py`: phi no y bruto, subamostragem, min-max
uma vez no conjunto inteiro, e so entao a divisao em grupos.

Alem dos valores, grava o que os torna interpretaveis: o motivo de cada
indefinicao, a variancia entre replicas de subamostragem, os diagnosticos de
bimodalidade do grupo raro, o custo de extracao e -- o principal -- a posicao
de cada medida do grupo raro dentro da distribuicao nula de subconjuntos do
mesmo tamanho.

Uso:
    python3 gerar_complexidade.py                     # tudo
    python3 gerar_complexidade.py --dry-run           # 3 datasets representativos
    python3 gerar_complexidade.py --sem-nulo --B 50   # mais rapido
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
import tracemalloc
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "lib"))

from complexidade import (  # noqa: E402
    MEDIDAS_EM_X,
    MEDIDAS_META,
    ContextoGrupo,
    calcular_grupo,
    calcular_vizglobal,
    minmax,
)
from grupos_phi import (  # noqa: E402
    GRUPOS,
    carregar_dataset,
    diagnosticar,
    mascara_raro,
    mascaras_dos_grupos,
    replicas_subamostra,
)
from nulo_tamanho import B_PADRAO, avaliar_grupo_raro  # noqa: E402

DATA_DIR = BASE_DIR / "data"
DRY_RUN = ("a1", "pdgfr", "california")  # pequeno, p >> n_raro, grande


class Cronometro:
    """Tempo por fase, para o registro de custo que o enunciado exige.

    A medicao de memoria e opcional e desligada por padrao: `tracemalloc`
    rastreia cada alocacao e chega a dominar o tempo do proprio processo que
    deveria medir -- no laco das distribuicoes nulas, que faz milhares de
    alocacoes pequenas, o custo explode. O numero que de fato justifica o teto
    de MAX_INSTANCIAS e analitico (a matriz densa de distancias e n^2 * 8
    bytes) e sai em `memoria_estimada_mb`.
    """

    def __init__(self, medir_memoria: bool = False):
        self.medir_memoria = medir_memoria
        self.linhas: list[dict] = []
        self._contexto: dict = {}

    def contexto(self, **kwargs):
        self._contexto = kwargs

    @contextmanager
    def fase(self, nome: str, **extra):
        if self.medir_memoria:
            tracemalloc.start()
        inicio = time.perf_counter()
        try:
            yield
        finally:
            segundos = time.perf_counter() - inicio
            if self.medir_memoria:
                _, pico = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                pico_mb = pico / 2**20
            else:
                pico_mb = np.nan
            self.linhas.append(
                {**self._contexto, **extra, "fase": nome,
                 "segundos": segundos, "pico_mb": pico_mb}
            )


def _valor_para_X(valores: dict, medida: str) -> float:
    """Coluna canonica de X: `_sq` para medidas agregativas, bruta para as
    escalares (C1..C4 ja estao em [0,1])."""
    chave = f"{medida}_sq" if MEDIDAS_META[medida].agregativa else medida
    return valores.get(chave, np.nan)


def processar(nome: str, csv_path: Path, cron: Cronometro, args) -> dict:
    Z, y = carregar_dataset(csv_path)
    # Memoria da matriz densa de distancias, que e o custo que forca o teto:
    # 1500^2 * 8 B = 18 MB, mas california (20640) daria 3,4 GB.
    cron.contexto(dataset=nome, n=len(y), p=Z.shape[1],
                  memoria_estimada_mb=len(y) ** 2 * 8 / 2**20)

    with cron.fase("phi"):
        raro_completo = mascara_raro(y)
    with cron.fase("diagnostico"):
        diag = diagnosticar(Z, y, raro_completo)

    replicas = replicas_subamostra(raro_completo)
    subamostrado = len(replicas) > 1

    longo, motivos, nulo = [], [], []
    for i_rep, manter in enumerate(replicas):
        Zs, ys, raros = Z[manter], y[manter], raro_completo[manter]
        with cron.fase("minmax", replica=i_rep):
            Zn = minmax(Zs)
            yn = minmax(ys.reshape(-1, 1)).ravel()
        ctx_completo = ContextoGrupo(Zn, yn)

        for grupo, mascara in mascaras_dos_grupos(raros).items():
            with cron.fase(f"medidas_{grupo}", replica=i_rep):
                res = calcular_grupo(ContextoGrupo(Zn[mascara], yn[mascara]))
            with cron.fase(f"vizglobal_{grupo}", replica=i_rep):
                viz = calcular_vizglobal(ctx_completo, mascara)

            n_grupo = int(mascara.sum())
            comum = {
                "dataset": nome, "replica": i_rep, "grupo": grupo,
                "n_grupo": n_grupo,
                "n_atributos_validos": ContextoGrupo(Zn[mascara], yn[mascara]).n_atributos_validos
                if n_grupo else 0,
                "confiabilidade_baixa": n_grupo < 20,
            }
            longo.append({**comum, **res.valores, **viz.valores})
            for chave, motivo in {**res.motivos, **viz.motivos}.items():
                motivos.append({**comum, "coluna": chave, "motivo": motivo})

        # A nula e cara e diagnostica: roda na primeira replica, o que basta
        # para situar o valor observado. A variancia entre replicas ja esta
        # medida em complexidade_replicas.csv.
        if not args.sem_nulo and i_rep == 0:
            observados = calcular_grupo(ContextoGrupo(Zn[raros], yn[raros])).valores
            with cron.fase("nulo", replica=i_rep):
                for linha in avaliar_grupo_raro(Zn, yn, raros, observados, B=args.B):
                    nulo.append({"dataset": nome, **linha})

    df_longo = pd.DataFrame(longo)
    medias = df_longo.groupby("grupo", sort=False).mean(numeric_only=True)
    desvios = df_longo.groupby("grupo", sort=False).std(numeric_only=True, ddof=1)

    linha_X = {"dataset": nome, "subamostrado": subamostrado,
               "n_replicas": len(replicas), **diag.como_dict()}
    for grupo in GRUPOS:
        for medida in MEDIDAS_EM_X:
            linha_X[f"{medida}_{grupo}"] = _valor_para_X(medias.loc[grupo].to_dict(), medida)
        linha_X[f"n_{grupo}"] = int(medias.loc[grupo, "n_grupo"])

    if subamostrado:
        for grupo in GRUPOS:
            for medida in MEDIDAS_EM_X:
                chave = f"{medida}_sq" if MEDIDAS_META[medida].agregativa else medida
                sd = desvios.loc[grupo].get(chave, np.nan)
                media = medias.loc[grupo].get(chave, np.nan)
                linha_X[f"cv_replica_{medida}_{grupo}"] = (
                    abs(sd / media) if media not in (0, np.nan) and np.isfinite(media) else np.nan
                )

    return {"X": linha_X, "longo": longo, "motivos": motivos, "nulo": nulo}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help=f"so {', '.join(DRY_RUN)}")
    parser.add_argument("--sem-nulo", action="store_true",
                        help="pula as distribuicoes nulas (diagnostico caro)")
    parser.add_argument("--B", type=int, default=B_PADRAO,
                        help="replicas por distribuicao nula")
    parser.add_argument("--memoria", action="store_true",
                        help="mede o pico real com tracemalloc (varias vezes mais lento)")
    parser.add_argument("--sufixo", default="",
                        help="sufixo dos arquivos de saida, p.ex. _dryrun")
    args = parser.parse_args()

    caminhos = sorted(DATA_DIR.glob("*.csv"))
    if args.dry_run:
        caminhos = [p for p in caminhos if p.stem in DRY_RUN]

    cron = Cronometro(medir_memoria=args.memoria)
    X, longo, motivos, nulo = [], [], [], []
    for i, csv_path in enumerate(caminhos, start=1):
        inicio = time.perf_counter()
        saida = processar(csv_path.stem, csv_path, cron, args)
        X.append(saida["X"])
        longo.extend(saida["longo"])
        motivos.extend(saida["motivos"])
        nulo.extend(saida["nulo"])
        print(f"[{i}/{len(caminhos)}] {csv_path.stem}: "
              f"{time.perf_counter() - inicio:.1f}s", flush=True)

    s = args.sufixo
    pd.DataFrame(X).set_index("dataset").to_csv(BASE_DIR / "meta-dataset" / f"X_complexidade{s}.csv")
    pd.DataFrame(longo).to_csv(BASE_DIR / "pratica-02" / "resultados" / f"complexidade{s}.csv", index=False)
    pd.DataFrame(motivos).to_csv(BASE_DIR / "pratica-02" / "resultados" / f"complexidade_motivos{s}.csv", index=False)
    if nulo:
        pd.DataFrame(nulo).to_csv(BASE_DIR / "pratica-02" / "resultados" / f"complexidade_nulo{s}.csv", index=False)
    custo = pd.DataFrame(cron.linhas)
    custo.attrs["ambiente"] = platform.platform()
    custo.to_csv(BASE_DIR / "pratica-02" / "resultados" / f"custo_extracao{s}.csv", index=False)

    print(f"\n{len(caminhos)} datasets, {custo['segundos'].sum():.1f}s de extracao")
    print(f"matriz de distancias estimada, maior dataset: "
          f"{custo['memoria_estimada_mb'].max():.0f} MB")
    if args.memoria:
        print(f"pico real medido: {custo['pico_mb'].max():.0f} MB")
    if motivos:
        print("\nindefinicoes por motivo:")
        print(pd.DataFrame(motivos)["motivo"].value_counts().to_string())


if __name__ == "__main__":
    main()
