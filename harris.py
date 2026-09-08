from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

BASE_DIR = Path(__file__).parent


class HARRISTree:
    def __init__(self, lam, max_depth=4, min_samples_leaf=2, max_features=None, random_state=None):
        self.lam = lam
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = np.random.default_rng(random_state)
        self.raiz = None

    def fit(self, X: pd.DataFrame, P_escalado: pd.DataFrame, R: pd.DataFrame):
        Xv, Pv, Rv = X.to_numpy(), P_escalado.to_numpy(), R.to_numpy()
        idx = np.arange(len(Xv))
        self.raiz = self._construir(Xv, Pv, Rv, idx, profundidade=0)
        return self

    def _perda_regressao(self, Pv, idx):
        return Pv[idx].var(axis=0).mean()

    def _perda_ranking(self, Rv, idx):
        if len(idx) < 2:
            return 0.0
        k = Rv.shape[1]
        c = rankdata(Rv[idx].mean(axis=0))
        c_centrado = c - c.mean()
        c_std = c.std()

        rows = Rv[idx]
        rows_centrado = rows - rows.mean(axis=1, keepdims=True)
        rows_std = rows.std(axis=1)

        with np.errstate(invalid="ignore", divide="ignore"):
            rho = (rows_centrado @ c_centrado) / (k * rows_std * c_std)
        perdas = np.where(np.isnan(rho), 1.0, 1 - rho)
        return perdas.mean() / 2.0

    def _perda_no(self, Pv, Rv, idx):
        return self.lam * self._perda_ranking(Rv, idx) + (1 - self.lam) * self._perda_regressao(Pv, idx)

    def _melhor_split(self, Xv, Pv, Rv, idx):
        n_features = Xv.shape[1]
        features = np.arange(n_features)
        if self.max_features is not None:
            k = min(self.max_features, n_features)
            features = self.rng.choice(n_features, size=k, replace=False)

        melhor, melhor_perda = None, np.inf
        for f in features:
            valores = np.unique(Xv[idx, f])
            if len(valores) < 2:
                continue
            for limiar in (valores[:-1] + valores[1:]) / 2:
                idx_esq = idx[Xv[idx, f] <= limiar]
                idx_dir = idx[Xv[idx, f] > limiar]
                if len(idx_esq) < self.min_samples_leaf or len(idx_dir) < self.min_samples_leaf:
                    continue
                perda = (
                    len(idx_esq) / len(idx) * self._perda_no(Pv, Rv, idx_esq)
                    + len(idx_dir) / len(idx) * self._perda_no(Pv, Rv, idx_dir)
                )
                if perda < melhor_perda:
                    melhor_perda, melhor = perda, (f, limiar, idx_esq, idx_dir)
        return melhor

    def _construir(self, Xv, Pv, Rv, idx, profundidade):
        rotulo = Pv[idx].mean(axis=0)
        if profundidade >= self.max_depth or len(idx) < 2 * self.min_samples_leaf:
            return {"folha": True, "rotulo": rotulo}

        split = self._melhor_split(Xv, Pv, Rv, idx)
        if split is None:
            return {"folha": True, "rotulo": rotulo}

        f, limiar, idx_esq, idx_dir = split
        return {
            "folha": False,
            "feature": f,
            "limiar": limiar,
            "esquerda": self._construir(Xv, Pv, Rv, idx_esq, profundidade + 1),
            "direita": self._construir(Xv, Pv, Rv, idx_dir, profundidade + 1),
        }

    def _prever_um(self, x):
        no = self.raiz
        while not no["folha"]:
            no = no["direita"] if x[no["feature"]] > no["limiar"] else no["esquerda"]
        return no["rotulo"]

    def predict(self, X: pd.DataFrame):
        return np.array([self._prever_um(x) for x in X.to_numpy()])


class HARRISForest:
    def __init__(self, lam, n_estimators=30, max_depth=4, min_samples_leaf=2, max_features="sqrt", random_state=42):
        self.lam = lam
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, P: pd.DataFrame, R: pd.DataFrame):
        n_features = X.shape[1]
        mf = max(1, int(np.sqrt(n_features))) if self.max_features == "sqrt" else self.max_features

        self.p_min, self.p_max = P.to_numpy().min(), P.to_numpy().max()
        P_escalado = (P - self.p_min) / (self.p_max - self.p_min)

        rng = np.random.default_rng(self.random_state)
        n = len(X)
        self.arvores = []
        for _ in range(self.n_estimators):
            idx_boot = rng.integers(0, n, size=n)
            arvore = HARRISTree(
                lam=self.lam,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=mf,
                random_state=rng.integers(0, 1_000_000),
            )
            arvore.fit(
                X.iloc[idx_boot].reset_index(drop=True),
                P_escalado.iloc[idx_boot].reset_index(drop=True),
                R.iloc[idx_boot].reset_index(drop=True),
            )
            self.arvores.append(arvore)
        self.colunas_P = P.columns
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        preds = np.mean([arvore.predict(X) for arvore in self.arvores], axis=0)
        return pd.DataFrame(preds, index=X.index, columns=self.colunas_P)


def harris(X_train, P_train, R_train, X_test, lam, **kwargs) -> pd.DataFrame:
    modelo = HARRISForest(lam=lam, **kwargs)
    modelo.fit(X_train, P_train, R_train)
    pred_P = modelo.predict(X_test)
    return pred_P.rank(axis=1, method="average", ascending=True)


def main():
    X = pd.read_csv(BASE_DIR / "X.csv", index_col=0)
    P = pd.read_csv(BASE_DIR / "P.csv", index_col=0)
    R = pd.read_csv(BASE_DIR / "R.csv", index_col=0)

    dataset_teste = "boston"
    X_train, X_test = X.drop(index=dataset_teste), X.loc[[dataset_teste]]
    P_train = P.drop(index=dataset_teste)
    R_train = R.drop(index=dataset_teste)

    print(f"Ranking real de '{dataset_teste}':")
    print(R.loc[dataset_teste].sort_values())

    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        pred = harris(X_train, P_train, R_train, X_test, lam=lam, random_state=42)
        print(f"\nHARRIS (lambda={lam}) previu para '{dataset_teste}':")
        print(pred.loc[dataset_teste].sort_values())


if __name__ == "__main__":
    main()
