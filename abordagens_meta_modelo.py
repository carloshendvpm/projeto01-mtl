from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = Path(__file__).parent
SEED = 42


def abordagem1_regressor_sobre_P(X_train, P_train, X_test) -> pd.DataFrame:
    modelo = RandomForestRegressor(random_state=SEED, n_jobs=-1)
    modelo.fit(X_train, P_train)
    pred_P = pd.DataFrame(modelo.predict(X_test), index=X_test.index, columns=P_train.columns)
    return pred_P.rank(axis=1, method="average", ascending=True)


def abordagem2_regressor_sobre_R(X_train, R_train, X_test) -> pd.DataFrame:
    modelo = RandomForestRegressor(random_state=SEED, n_jobs=-1)
    modelo.fit(X_train, R_train)
    pred_R = pd.DataFrame(modelo.predict(X_test), index=X_test.index, columns=R_train.columns)
    return pred_R.rank(axis=1, method="average", ascending=True)


def main():
    X = pd.read_csv(BASE_DIR / "X.csv", index_col=0)
    P = pd.read_csv(BASE_DIR / "P.csv", index_col=0)
    R = pd.read_csv(BASE_DIR / "R.csv", index_col=0)

    dataset_teste = "boston"
    X_train, X_test = X.drop(index=dataset_teste), X.loc[[dataset_teste]]
    P_train = P.drop(index=dataset_teste)
    R_train = R.drop(index=dataset_teste)

    pred1 = abordagem1_regressor_sobre_P(X_train, P_train, X_test)
    pred2 = abordagem2_regressor_sobre_R(X_train, R_train, X_test)

    print(f"Ranking real de '{dataset_teste}':")
    print(R.loc[dataset_teste].sort_values())
    print(f"\nAbordagem 1 previu para '{dataset_teste}':")
    print(pred1.loc[dataset_teste].sort_values())
    print(f"\nAbordagem 2 previu para '{dataset_teste}':")
    print(pred2.loc[dataset_teste].sort_values())


if __name__ == "__main__":
    main()
