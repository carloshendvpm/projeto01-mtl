import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import resreg
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).parent / "lib"))
import regression_metrics as rm  # noqa: E402
from phi import phi  # noqa: E402
from phi_ctrl_pts import phi_ctrl_pts  # noqa: E402

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
N_FOLDS = 10
SEED = 42
OVER = 0.5
UNDER = 0.5

ALGORITMOS = {
    "Linear": make_pipeline(StandardScaler(), LinearRegression()),
    "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor()),
    "SVR": make_pipeline(StandardScaler(), SVR()),
    "DT": DecisionTreeRegressor(random_state=SEED),
    "RF": RandomForestRegressor(random_state=SEED, n_jobs=-1),
    "GB": GradientBoostingRegressor(random_state=SEED),
}


def carregar(csv_path: Path):
    df = pd.read_csv(csv_path)
    y = df.iloc[:, 0].to_numpy()
    X = df.iloc[:, 1:].to_numpy()
    return X, y


def reamostrar(X_treino, y_treino):
    y_serie = pd.Series(y_treino)
    params_phi = phi_ctrl_pts(y_treino)
    relevancia = np.array(phi(y_serie, params_phi))
    X_rs, y_rs = resreg.wercs(
        X_treino, y_treino, relevancia, over=OVER, under=UNDER, random_state=SEED
    )
    return X_rs, y_rs


def rodar_dataset(csv_path: Path) -> list[dict]:
    nome_dataset = csv_path.stem
    X, y = carregar(csv_path)
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    linhas = []
    for fold, (idx_treino, idx_teste) in enumerate(kfold.split(X)):
        X_treino, X_teste = X[idx_treino], X[idx_teste]
        y_treino, y_teste = y[idx_treino], y[idx_teste]
        X_treino_rs, y_treino_rs = reamostrar(X_treino, y_treino)

        for nome_algo, modelo in ALGORITMOS.items():
            modelo.fit(X_treino_rs, y_treino_rs)
            y_pred = modelo.predict(X_teste)
            sera = rm.sera(y_teste, y_pred)
            linhas.append(
                {"dataset": nome_dataset, "algoritmo": nome_algo, "fold": fold, "sera": sera}
            )
    return linhas


def main():
    csv_paths = sorted(DATA_DIR.glob("*.csv"))
    todas_linhas = []
    for i, csv_path in enumerate(csv_paths, start=1):
        print(f"[{i}/{len(csv_paths)}] {csv_path.name}")
        todas_linhas.extend(rodar_dataset(csv_path))

    resultado = pd.DataFrame(todas_linhas)
    saida = Path(__file__).parent / "resultados_por_fold_resampling.csv"
    resultado.to_csv(saida, index=False)
    print(f"\nSalvo em {saida} ({len(resultado)} linhas)")


if __name__ == "__main__":
    main()
