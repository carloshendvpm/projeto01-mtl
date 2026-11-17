"""Meta-aprendizagem com warm-start (Prática 3) nas mesmas partições do AutoML.

Na Prática 3 a busca consultava a matriz de perdas, cujos valores para o
dataset alvo vinham da validação cruzada completa, incluindo as partições de
teste. Aqui a busca é refeita dentro de cada fold externo:

  * os pontos iniciais vêm dos k=3 vizinhos mais próximos em Xestendido
    (braço principal da Prática 3), usando só a matriz de perdas dos OUTROS 29
    datasets (meta-conhecimento offline). O alvo é descrito pelo Xestendido
    calculado só no treino do fold (meta_por_fold.py); na primeira versão ele
    vinha do dataset inteiro, o que deixava o teste influenciar os vizinhos;
  * cada configuração avaliada é medida por validação cruzada interna (3 folds)
    sobre o treino do fold externo: a busca nunca vê o teste;
  * o SMBO (GP Matérn 2,5 + EI sobre a grade de 60 configurações) segue a
    Prática 3, com o mesmo orçamento T = 20 avaliações;
  * a melhor configuração pela validação interna é reajustada no treino
    completo e avaliada no teste do fold.

Com --frio, registra também o SMBO com cinco iniciais aleatórias (o efeito do
meta-conhecimento na própria solução já foi medido na Prática 3).

A busca é determinística dado (treino, iniciais). Com --reaproveitar, um fold
cujas iniciais coincidem com as da versão anterior (resultados/versao_1/) copia a
linha antiga em vez de refazer a busca; o teste `test_warmstart_reaproveitado`
refaz alguns desses folds e confere que a SERA é idêntica.

Saída: resultados/warmstart_por_fold.csv
Uso: .venv/bin/python executar_warmstart.py [--jobs 6] [--datasets ...] [--reaproveitar]
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from itertools import product

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import comum as c

warnings.filterwarnings("ignore")

T, N_INIT, K_VIZ, FOLDS_INTERNOS = 20, 5, 3, 3

CONFIGS = {
    f"c{i:03d}": {"max_features": f, "min_samples_leaf": folha}
    for i, (f, folha) in enumerate(product([1.0, 0.2, 0.35, 0.5, 0.65, 0.8],
                                           [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]))
}
IDS = list(CONFIGS)
Z = np.array([[p["max_features"], np.log2(p["min_samples_leaf"]) / 4] for p in CONFIGS.values()])


def xestendido() -> pd.DataFrame:
    """Mesma construção da Prática 3 (51 descritores)."""
    return pd.read_csv(c.P3 / "Xestendido_usado.csv", index_col=0)


def vizinhos(meta_outros: pd.DataFrame, alvo: pd.Series) -> pd.Series:
    """Distância do alvo aos outros datasets; escala e imputação ajustadas só nos outros."""
    treino = meta_outros.replace([np.inf, -np.inf], np.nan)
    treino = treino.loc[:, treino.notna().any()]
    mediana = treino.median()
    escala = StandardScaler().fit(treino.fillna(mediana))
    a = escala.transform(treino.fillna(mediana))
    linha = pd.DataFrame([alvo[treino.columns].astype(float)]).replace([np.inf, -np.inf], np.nan)
    b = escala.transform(linha.fillna(mediana))
    return pd.Series(np.linalg.norm(a - b, axis=1), index=treino.index).sort_values(kind="stable")


def iniciais_similares(perdas_treino: pd.DataFrame, viz: pd.Series, k: int) -> list[str]:
    rankings = [perdas_treino[d].sort_values(kind="stable").index for d in viz.index[:k]]
    escolhidas = []
    for rodada in zip(*rankings):
        for cid in rodada:
            if cid not in escolhidas:
                escolhidas.append(cid)
            if len(escolhidas) == N_INIT:
                return escolhidas
    raise ValueError("iniciais incompletas")


def perda_interna(X, y, cid) -> tuple[float, float]:
    """SERA média da validação cruzada interna (só treino do fold externo)."""
    t0 = time.perf_counter()
    perdas = []
    for tr, va in KFold(FOLDS_INTERNOS, shuffle=True, random_state=c.SEED).split(X):
        m = RandomForestRegressor(**CONFIGS[cid], n_estimators=100, random_state=c.SEED, n_jobs=1)
        m.fit(X[tr], y[tr])
        perdas.append(c.sera(y[va], m.predict(X[va])))
    return float(np.mean(perdas)), time.perf_counter() - t0


def buscar(X, y, iniciais, seed) -> tuple[str, float, list]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    vistos, observadas, segundos = [], [], 0.0
    for cid in iniciais:
        perda, s = perda_interna(X, y, cid)
        vistos.append(IDS.index(cid)); observadas.append(perda); segundos += s
    while len(vistos) < T:
        restantes = np.array([i for i in range(len(IDS)) if i not in vistos])
        gp = GaussianProcessRegressor(kernel=Matern(length_scale=0.5, nu=2.5, length_scale_bounds="fixed"),
                                      alpha=1e-6, normalize_y=True, optimizer=None)
        gp.fit(Z[vistos], observadas)
        media, desvio = gp.predict(Z[restantes], return_std=True)
        melhora = min(observadas) - media
        z = melhora / np.maximum(desvio, 1e-12)
        ei = melhora * norm.cdf(z) + desvio * norm.pdf(z)
        ei[desvio < 1e-12] = 0
        melhores = restantes[np.isclose(ei, ei.max(), rtol=1e-10, atol=1e-14)]
        proximo = int(rng.choice(melhores))
        perda, s = perda_interna(X, y, IDS[proximo])
        vistos.append(proximo); observadas.append(perda); segundos += s
    melhor = IDS[vistos[int(np.argmin(observadas))]]
    return melhor, segundos, [IDS[i] for i in vistos]


# Datasets da mesma origem (mesmo X ou mesmo sistema); ver Seção 9 do notebook
GRUPOS = [{"a1", "a2", "a3", "a7"}, {"compactiv", "cpu_small"}, {"mortgage", "treasury"},
          {"available_power", "maximal_torque"}]


def parentes(nome: str) -> set[str]:
    return next((g - {nome} for g in GRUPOS if nome in g), set())


def rodar(nome: str, matriz: pd.DataFrame, meta: pd.DataFrame, meta_fold: pd.DataFrame,
          antigos: pd.DataFrame | None, frio: bool, sem_parentes: bool = False) -> list[dict]:
    X, y = c.carregar(nome)
    if sem_parentes:   # leave-one-group-out: o grupo do alvo sai do meta-conhecimento
        fora = list(parentes(nome))
        matriz, meta = matriz.drop(columns=fora), meta.drop(index=fora)
    linhas = []
    for fold, (tr, te) in enumerate(c.folds(X)):
        t0 = time.perf_counter()
        alvo = meta_fold.loc[(nome, fold)]
        viz = vizinhos(meta.drop(index=nome), alvo)
        iniciais = {"warmstart": iniciais_similares(matriz.drop(columns=nome), viz, K_VIZ)}
        t_iniciais = time.perf_counter() - t0 + alvo["segundos_xbase"] + alvo["segundos_complexidade"]
        if frio:
            iniciais["smbo_frio"] = np.random.default_rng(0).choice(IDS, N_INIT, replace=False).tolist()
        for metodo, ini in iniciais.items():
            base = dict(dataset=nome, fold=fold, metodo=metodo,
                        segundos_iniciais=t_iniciais if metodo == "warmstart" else 0.0,
                        vizinhos=",".join(viz.index[:K_VIZ]) if metodo == "warmstart" else "")
            if antigos is not None and (nome, fold, metodo) in antigos.index:
                velho = antigos.loc[(nome, fold, metodo)]
                if velho.avaliadas.split(",")[:N_INIT] == ini:
                    linhas.append({**base, **{k: velho[k] for k in
                                   ["sera", "config", "segundos_busca", "segundos_final", "avaliadas"]},
                                   "reaproveitado": True})
                    continue
            melhor, s_busca, ordem = buscar(X[tr], y[tr], ini, seed=0)
            t1 = time.perf_counter()
            m = RandomForestRegressor(**CONFIGS[melhor], n_estimators=100, random_state=c.SEED, n_jobs=1)
            m.fit(X[tr], y[tr])
            perda = c.sera(y[te], m.predict(X[te]))
            s_final = time.perf_counter() - t1
            linhas.append({**base, "sera": perda, "config": melhor, "segundos_busca": s_busca,
                           "segundos_final": s_final, "avaliadas": ",".join(ordem), "reaproveitado": False})
    print(nome, "concluído", flush=True)
    return linhas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--frio", action="store_true")
    ap.add_argument("--reaproveitar", action="store_true")
    ap.add_argument("--saida", default="warmstart_por_fold")
    ap.add_argument("--sem-parentes", action="store_true",
                    help="tira do meta-conhecimento os datasets da mesma origem (só datasets com parentes)")
    args = ap.parse_args()
    salvas = json.loads((c.P3 / "configuracoes.json").read_text())
    assert salvas == CONFIGS, "grade diferente da Prática 3"
    matriz = pd.read_csv(c.P3 / "matriz_perdas.csv", index_col=0)
    meta = xestendido()
    from meta_por_fold import carregar_meta_por_fold
    meta_fold = carregar_meta_por_fold()
    antigos = None
    if args.reaproveitar:
        antigos = pd.read_csv(c.SAIDA / "versao_1" / "warmstart_por_fold.csv").set_index(["dataset", "fold", "metodo"])
    nomes = args.datasets or c.datasets()
    if args.sem_parentes:
        nomes = [n for n in nomes if parentes(n)]
        args.saida = "warmstart_sem_parentes_por_fold"
    saida = Parallel(n_jobs=args.jobs)(delayed(rodar)(n, matriz, meta, meta_fold, antigos, args.frio,
                                                      args.sem_parentes) for n in nomes)
    sufixo = "" if not args.datasets or args.sem_parentes else "_parcial"
    pd.DataFrame([r for b in saida for r in b]).to_csv(c.SAIDA / f"{args.saida}{sufixo}.csv", index=False)


if __name__ == "__main__":
    main()
