"""Executa FLAML e AutoGluon nas mesmas dez partições, otimizando a SERA.

O sistema recebe só o treino do fold; a validação interna é dele. O teste é
usado uma única vez, para medir a SERA do modelo final.

Variantes:
  flaml        FLAML com pontos iniciais estáticos (sem meta-conhecimento)
  flaml_meta   FLAML com starting_points="data": portfólio zero-shot aprendido
               offline (flaml.default), escolhido pelas características do treino
  autogluon    AutoGluon (presets medium_quality: portfólio padrão + ensemble
               ponderado por seleção gulosa de Caruana)

Cada execução (sistema × orçamento × dataset × fold) grava uma linha JSON em
resultados/automl/<sistema>_<orcamento>s.jsonl; execuções já gravadas são puladas.

Uso: .venv/bin/python executar_automl.py --sistema flaml_meta --orcamento 30 \
         [--workers 4] [--ncpus 2] [--datasets ...]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

import comum as c

warnings.filterwarnings("ignore")
PASTA = c.SAIDA / "automl"
PASTA.mkdir(exist_ok=True)


# ---------------------------------------------------------------- métricas

def sera_flaml(X_val, y_val, estimator, labels, X_train, y_train,
               weight_val=None, weight_train=None, *args, **kwargs):
    """Assinatura de métrica customizada do FLAML: devolve (perda, extras)."""
    return c.sera(np.asarray(y_val), estimator.predict(X_val)), {}


def sera_score(y_true, y_pred):
    return c.sera(np.asarray(y_true), np.asarray(y_pred))


# ---------------------------------------------------------------- sistemas

def _corrigir_flaml():
    """Contorna um erro do FLAML 2.x com starting_points="data".

    Em datasets maiores o portfólio zero-shot de rf/extra_tree sugere
    hiperparâmetros (p.ex. `criterion`) que não existem no espaço de busca, e a
    checagem de validade quebra com KeyError. Tratamos esse ponto como inválido,
    que é o que o FLAML já faz com pontos fora dos limites: ele é descartado.
    """
    from flaml.automl.state import SearchState
    original = SearchState.valid_starting_point
    if getattr(original, "_corrigido", False):
        return

    def valid_starting_point(self, starting_point, search_space):
        if any(k not in search_space for k in starting_point if k != "FLAML_sample_size"):
            return False
        return original(self, starting_point, search_space)
    valid_starting_point._corrigido = True
    SearchState.valid_starting_point = valid_starting_point


def rodar_flaml(X_tr, y_tr, X_te, orcamento, ncpus, meta: bool) -> tuple[np.ndarray, dict]:
    from flaml import AutoML
    _corrigir_flaml()
    automl = AutoML()
    automl.fit(X_tr, y_tr, task="regression", metric=sera_flaml, time_budget=orcamento,
               n_jobs=ncpus, seed=c.SEED, verbose=0, log_file_name="",
               starting_points="data" if meta else "static")
    pred = automl.predict(X_te)
    # Pontos iniciais efetivamente usados: com starting_points="data", o FLAML
    # descarta as configurações zero-shot que caem fora do espaço de busca
    # (cujo limite de n_estimators depende do tamanho do treino).
    iniciais = {e: automl._search_states[e].init_config for e in automl.estimator_list}
    info = dict(modelo=automl.best_estimator, config=automl.best_config,
                ordem_estimadores=list(automl.estimator_list), iniciais=iniciais,
                perda_validacao=float(automl.best_loss),
                avaliacoes=len(automl.config_history),
                estimadores=list(automl.estimator_list), ensemble=False,
                validacao=automl._state.eval_method)
    return pred, info


def rodar_autogluon(X_tr, y_tr, X_te, orcamento, ncpus) -> tuple[np.ndarray, dict]:
    import pandas as pd
    from autogluon.core.metrics import make_scorer
    from autogluon.tabular import TabularPredictor

    scorer = make_scorer("sera", sera_score, optimum=0, greater_is_better=False)
    colunas = [f"x{i}" for i in range(X_tr.shape[1])]
    treino = pd.DataFrame(X_tr, columns=colunas).assign(y=y_tr)
    pasta = tempfile.mkdtemp(prefix="ag_")
    try:
        pred_ag = TabularPredictor(label="y", problem_type="regression", eval_metric=scorer,
                                   path=pasta, verbosity=0)
        pred_ag.fit(treino, time_limit=orcamento, presets="medium_quality",
                    num_cpus=ncpus, num_gpus=0)
        pred = pred_ag.predict(pd.DataFrame(X_te, columns=colunas)).to_numpy()
        melhor = pred_ag.model_best
        pesos = {}
        try:
            modelo = pred_ag._trainer.load_model(melhor)
            pesos = {k: float(v) for k, v in modelo._get_model_weights().items()}
        except Exception:
            pass
        lb = pred_ag.leaderboard(silent=True)
        info = dict(modelo=melhor, pesos_ensemble=pesos, ensemble=bool(pesos) and len(pesos) > 1,
                    perda_validacao=float(-lb.loc[lb.model == melhor, "score_val"].iloc[0]),
                    avaliacoes=int(len(lb)), modelos_treinados=lb.model.tolist())
        return pred, info
    finally:
        shutil.rmtree(pasta, ignore_errors=True)


# ---------------------------------------------------------------- execução

def tarefa(args) -> dict:
    sistema, orcamento, nome, fold, ncpus = args
    os.environ["OMP_NUM_THREADS"] = str(ncpus)
    X, y = c.carregar(nome)
    tr, te = c.folds(X)[fold]
    linha = dict(sistema=sistema, orcamento=orcamento, dataset=nome, fold=fold, ncpus=ncpus)
    t0 = time.perf_counter()
    try:
        if sistema == "autogluon":
            pred, info = rodar_autogluon(X[tr], y[tr], X[te], orcamento, ncpus)
        else:
            pred, info = rodar_flaml(X[tr], y[tr], X[te], orcamento, ncpus, meta=sistema == "flaml_meta")
        linha.update(sera=c.sera(y[te], pred), segundos=time.perf_counter() - t0, erro=None, **info)
    except Exception as e:  # registra a falha em vez de derrubar o lote
        linha.update(sera=None, segundos=time.perf_counter() - t0, erro=f"{type(e).__name__}: {e}",
                     traceback=traceback.format_exc()[-2000:])
    return linha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sistema", required=True, choices=["flaml", "flaml_meta", "autogluon"])
    ap.add_argument("--orcamento", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ncpus", type=int, default=2)
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--folds", nargs="*", type=int)
    args = ap.parse_args()

    arquivo = PASTA / f"{args.sistema}_{args.orcamento}s.jsonl"
    feitos = set()
    if arquivo.exists():
        for l in arquivo.read_text().splitlines():
            r = json.loads(l)
            if r.get("erro") is None:
                feitos.add((r["dataset"], r["fold"]))
    nomes = args.datasets or c.datasets()
    folds = args.folds if args.folds is not None else range(c.FOLDS)
    # datasets maiores primeiro, para não sobrar um lento no fim
    tamanhos = {n: (c.DADOS / f"{n}.csv").stat().st_size for n in nomes}
    tarefas = [(args.sistema, args.orcamento, n, f, args.ncpus)
               for n in sorted(nomes, key=tamanhos.get, reverse=True) for f in folds
               if (n, f) not in feitos]
    print(f"{len(tarefas)} execuções pendentes em {arquivo.name}", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, maxtasksperchild=1) as pool, arquivo.open("a") as saida:
        for i, linha in enumerate(pool.imap_unordered(tarefa, tarefas), 1):
            saida.write(json.dumps(linha, default=str) + "\n")
            saida.flush()
            status = f"sera={linha['sera']:.4g}" if linha["sera"] is not None else linha["erro"]
            print(f"[{i}/{len(tarefas)}] {linha['dataset']} f{linha['fold']} "
                  f"{linha['segundos']:.1f}s {status}", flush=True)


if __name__ == "__main__":
    main()
