"""Protocolo comum da Prática 4: dados, folds, SERA e regressores das práticas anteriores.

Tudo o que define o protocolo fica aqui, para que baselines, meta-aprendizagem,
DESReg e AutoML usem exatamente as mesmas partições e a mesma medida.
"""
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

AQUI = Path(__file__).resolve().parent
BASE = AQUI.parent                      # projeto-01/
DADOS = BASE / "data"
META = BASE / "meta-dataset"
P3 = BASE / "pratica-03" / "resultados"
SAIDA = AQUI / "resultados"
SAIDA.mkdir(exist_ok=True)

sys.path.insert(0, str(BASE / "lib"))
import regression_metrics as rm  # noqa: E402

SEED = 42
FOLDS = 10
ALGORITMOS = ["DT", "GB", "KNN", "Linear", "RF", "SVR"]
LIMIARES = np.arange(0, 1.01, 0.01)


def datasets() -> list[str]:
    """Os 30 datasets das práticas anteriores, na ordem de X.csv."""
    return pd.read_csv(META / "X.csv", index_col=0).index.tolist()


def carregar(nome: str) -> tuple[np.ndarray, np.ndarray]:
    """Primeira coluna é o alvo, como em todas as práticas anteriores."""
    dados = pd.read_csv(DADOS / f"{nome}.csv")
    return dados.iloc[:, 1:].to_numpy(dtype=float), dados.iloc[:, 0].to_numpy(dtype=float)


def folds(X) -> list[tuple[np.ndarray, np.ndarray]]:
    """As mesmas dez partições das Práticas 1 e 3 (KFold, shuffle, semente 42)."""
    return list(KFold(FOLDS, shuffle=True, random_state=SEED).split(X))


def pesos_sera(y_teste) -> np.ndarray:
    """Peso de cada erro quadrático na SERA (integração trapezoidal, passo 0,01).

    A relevância é estimada no alvo da própria partição avaliada, como na função
    legada `rm.sera` usada nas Práticas 1 e 3. Pela linearidade da soma, a SERA é
    o produto desses pesos pelos erros quadráticos.
    """
    relevancia = np.asarray(rm.calculate_phi(np.asarray(y_teste, dtype=float)))
    indicadores = relevancia[:, None] >= LIMIARES[None, :]
    return np.trapezoid(indicadores.astype(float), dx=0.01, axis=1)


def sera(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.dot(pesos_sera(y_true), (y_true - y_pred) ** 2))


def regressores(n_jobs: int = 1) -> dict:
    """Os seis regressores da matriz P, com configuração padrão (Prática 1)."""
    return {
        "DT": DecisionTreeRegressor(random_state=SEED),
        "GB": GradientBoostingRegressor(random_state=SEED),
        "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor()),
        "Linear": make_pipeline(StandardScaler(), LinearRegression()),
        "RF": RandomForestRegressor(random_state=SEED, n_jobs=n_jobs),
        "SVR": make_pipeline(StandardScaler(), SVR()),
    }


def ambiente() -> dict:
    import subprocess
    try:
        mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:  # pragma: no cover
        mem, cpu = None, platform.processor()
    return {"cpu": cpu, "nucleos": os.cpu_count(), "memoria_gb": mem / 2**30 if mem else None,
            "python": platform.python_version(), "sistema": platform.platform()}
