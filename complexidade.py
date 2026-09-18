"""Medidas de complexidade para regressao (Lorena et al., 2018).

Cada medida e calculada sobre um conjunto (X, y) ja normalizado em [0,1].
O modulo nao sabe nada sobre "raro" ou "normal": quem separa os grupos e
o gerar_complexidade.py.
"""

import warnings

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.linear_model import LinearRegression

# atributo constante nao correlaciona com nada; tratamos como rho = 0
warnings.filterwarnings("ignore", category=ConstantInputWarning)

RNG_SEED = 42
MIN_INSTANCIAS = 10  # abaixo disso as medidas ficam instaveis demais


def _minmax(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    lo = A.min(axis=0)
    amp = A.max(axis=0) - lo
    amp = np.where(amp == 0, 1.0, amp)
    return (A - lo) / amp


def _rho(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """|rho de Spearman| de cada atributo com o alvo."""
    n_feat = X.shape[1]
    out = np.zeros(n_feat)
    for j in range(n_feat):
        if np.all(X[:, j] == X[0, j]):
            out[j] = 0.0
            continue
        r = spearmanr(X[:, j], y).statistic
        out[j] = 0.0 if np.isnan(r) else abs(r)
    return out


# ---------------------------------------------------------------- correlacao

def C1(X, y):
    """Correlacao maxima atributo-alvo (quanto maior, mais simples)."""
    return float(_rho(X, y).max())


def C2(X, y):
    """Correlacao media atributo-alvo."""
    return float(_rho(X, y).mean())


def C3(X, y):
    """Fracao minima de exemplos a remover para atingir |rho| >= 0.9."""
    n = X.shape[0]
    n_j = []
    for j in range(X.shape[1]):
        n_j.append(_remover_ate_correlacionar(X[:, j], y, n))
    return float(min(n_j) / n)


def _remover_ate_correlacionar(x, y, n, alvo=0.9):
    """Quantos exemplos remover para |rho(x, y)| >= alvo.

    Remove iterativamente o exemplo com maior discrepancia entre o rank de x
    e o rank de y, como no algoritmo original.
    """
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    sinal = 1.0
    r = spearmanr(x, y).statistic
    if not np.isnan(r) and r < 0:
        sinal = -1.0
    rx = sinal * rx
    ativos = np.ones(n, dtype=bool)
    for removidos in range(n - 3):
        xi, yi = x[ativos], y[ativos]
        r = spearmanr(xi, yi).statistic
        if not np.isnan(r) and abs(r) >= alvo:
            return removidos
        d = np.abs(np.argsort(np.argsort(sinal * xi)) - np.argsort(np.argsort(yi)))
        idx = np.flatnonzero(ativos)[int(np.argmax(d))]
        ativos[idx] = False
    return n - 3


def C4(X, y, limiar=0.1):
    """Eficiencia coletiva dos atributos: fracao de exemplos nao explicados.

    Toma o atributo mais correlacionado, remove os exemplos com residuo
    pequeno no ajuste linear simples, repete com o atributo seguinte.
    """
    ativos = np.ones(X.shape[0], dtype=bool)
    restantes = list(range(X.shape[1]))
    while restantes and ativos.sum() > 2:
        rho = _rho(X[np.ix_(ativos, restantes)], y[ativos])
        j = restantes.pop(int(np.argmax(rho)))
        xi = X[ativos, j].reshape(-1, 1)
        yi = y[ativos]
        res = np.abs(yi - LinearRegression().fit(xi, yi).predict(xi))
        manter = res > limiar
        if manter.all():
            continue
        idx = np.flatnonzero(ativos)
        ativos[idx[~manter]] = False
    return float(ativos.sum() / X.shape[0])


# ---------------------------------------------------------------- linearidade

def _residuos_lineares(X, y):
    pred = LinearRegression().fit(X, y).predict(X)
    return np.abs(y - pred)


def L1(X, y):
    """Erro absoluto medio do regressor linear."""
    return float(_residuos_lineares(X, y).mean())


def L2(X, y):
    """Erro quadratico medio do regressor linear."""
    return float((_residuos_lineares(X, y) ** 2).mean())


def L3(X, y, rng=None):
    """Nao-linearidade do regressor linear: erro em pontos interpolados."""
    Xi, yi = _interpolar(X, y, rng)
    modelo = LinearRegression().fit(X, y)
    return float(((yi - modelo.predict(Xi)) ** 2).mean())


def _interpolar(X, y, rng=None):
    """Gera pontos novos interpolando pares de vizinhos na ordem do alvo."""
    rng = rng or np.random.default_rng(RNG_SEED)
    ordem = np.argsort(y)
    Xs, ys = X[ordem], y[ordem]
    a, b = Xs[:-1], Xs[1:]
    ya, yb = ys[:-1], ys[1:]
    alfa = rng.random((len(a), 1))
    return a + alfa * (b - a), ya + alfa[:, 0] * (yb - ya)


# ---------------------------------------------------------------- suavidade

def S1(X, y):
    """Distribuicao do alvo: variacao media de y nas arestas da MST de X."""
    D = squareform(pdist(X))
    mst = minimum_spanning_tree(D).tocoo()
    return float(np.abs(y[mst.row] - y[mst.col]).mean())


def S2(X, y):
    """Distribuicao das entradas: distancia media entre exemplos vizinhos em y."""
    Xs = X[np.argsort(y)]
    return float(np.linalg.norm(Xs[1:] - Xs[:-1], axis=1).mean())


def _predicao_1nn(Xtreino, ytreino, Xteste):
    D = np.linalg.norm(Xteste[:, None, :] - Xtreino[None, :, :], axis=2)
    return ytreino[np.argmin(D, axis=1)]


def S3(X, y):
    """Erro do vizinho mais proximo (leave-one-out, erro quadratico medio)."""
    D = squareform(pdist(X))
    np.fill_diagonal(D, np.inf)
    return float(((y - y[np.argmin(D, axis=1)]) ** 2).mean())


def S4(X, y, rng=None):
    """Nao-linearidade do vizinho mais proximo: erro em pontos interpolados."""
    Xi, yi = _interpolar(X, y, rng)
    return float(((yi - _predicao_1nn(X, y, Xi)) ** 2).mean())


# ---------------------------------------------------------- dimensionalidade

def T2(X, y=None):
    """Razao entre numero de atributos e numero de exemplos."""
    return float(X.shape[1] / X.shape[0])


MEDIDAS = {
    "C1": C1, "C2": C2, "C3": C3, "C4": C4,
    "L1": L1, "L2": L2, "L3": L3,
    "S1": S1, "S2": S2, "S3": S3, "S4": S4,
    "T2": T2,
}


def calcular_todas(X, y, normalizar=True) -> dict:
    """Todas as medidas para um conjunto. Retorna NaN se o conjunto e pequeno."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.shape[0] < MIN_INSTANCIAS:
        return {nome: np.nan for nome in MEDIDAS}
    if normalizar:
        X = _minmax(X)
        y = _minmax(y.reshape(-1, 1)).ravel()
    rng = np.random.default_rng(RNG_SEED)
    out = {}
    for nome, fn in MEDIDAS.items():
        try:
            out[nome] = fn(X, y, rng) if nome in ("L3", "S4") else fn(X, y)
        except Exception:
            out[nome] = np.nan
    return out
