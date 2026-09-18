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
    """|rho de Spearman| de cada atributo com o alvo.

    Atributo constante (ou alvo constante) devolve NaN, nao zero: a correlacao
    ali e indefinida, e trata-la como zero seria afirmar ausencia de associacao
    (ficha 4.2 do levantamento da Etapa 1). Os agregados C1/C2 ignoram os NaN.
    """
    n_feat = X.shape[1]
    out = np.full(n_feat, np.nan)
    for j in range(n_feat):
        if np.all(X[:, j] == X[0, j]):
            continue
        out[j] = abs(spearmanr(X[:, j], y).statistic)
    return out


# ---------------------------------------------------------------- correlacao

def C1(X, y):
    """Correlacao maxima atributo-alvo (quanto maior, mais simples)."""
    return _agregado(_rho(X, y), np.nanmax)


def C2(X, y):
    """Correlacao media atributo-alvo (C2.mean do levantamento)."""
    return _agregado(_rho(X, y), np.nanmean)


def _agregado(rho: np.ndarray, fn) -> float:
    """Agrega os rho validos; NaN se nenhum atributo tem correlacao definida."""
    return np.nan if np.all(np.isnan(rho)) else float(fn(rho))


def n_atributos_validos(X, y) -> int:
    """Quantos atributos tem correlacao definida (entram em C1/C2)."""
    return int(np.count_nonzero(~np.isnan(_rho(X, y))))


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
    sinal = 1.0
    r = spearmanr(x, y).statistic
    if not np.isnan(r) and r < 0:
        sinal = -1.0
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
        if np.all(np.isnan(rho)):
            break
        j = restantes.pop(int(np.nanargmax(rho)))
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

def _arestas_mst(X, y) -> np.ndarray:
    """|y_i - y_j| em cada aresta unica da MST construida sobre X."""
    mst = minimum_spanning_tree(squareform(pdist(X))).tocoo()
    return np.abs(y[mst.row] - y[mst.col])


def S1(X, y):
    """Variacao media de y nas arestas da MST (S1.mean)."""
    return float(_arestas_mst(X, y).mean())


def S1_sd(X, y):
    """Desvio padrao amostral da variacao de y nas arestas da MST (S1.sd)."""
    a = _arestas_mst(X, y)
    return float(a.std(ddof=1)) if len(a) > 1 else np.nan


def S2(X, y):
    """Distribuicao das entradas: distancia media entre exemplos vizinhos em y."""
    Xs = X[np.argsort(y)]
    return float(np.linalg.norm(Xs[1:] - Xs[:-1], axis=1).mean())


def _predicao_1nn(Xtreino, ytreino, Xteste):
    D = np.linalg.norm(Xteste[:, None, :] - Xtreino[None, :, :], axis=2)
    return ytreino[np.argmin(D, axis=1)]


def erros_1nn(X, y) -> np.ndarray:
    """Erro quadratico de 1-NN leave-one-out para CADA exemplo.

    O vizinho e buscado no conjunto inteiro passado aqui. Empates de distancia
    resolvem-se pelo menor indice (convencao da ficha 4.5), que e o que argmin
    ja faz. Serve de base para S3 e para as agregacoes restritas a um grupo.
    """
    D = squareform(pdist(X))
    np.fill_diagonal(D, np.inf)
    return (y - y[np.argmin(D, axis=1)]) ** 2


def S3(X, y):
    """Erro do vizinho mais proximo (leave-one-out, erro quadratico medio)."""
    return float(erros_1nn(X, y).mean())


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
    "S1": S1, "S1_sd": S1_sd, "S2": S2, "S3": S3, "S4": S4,
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
