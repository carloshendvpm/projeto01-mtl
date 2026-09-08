import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

sys.path.insert(0, str(Path(__file__).parent / "lib"))
from phi import phi  # noqa: E402
from phi_ctrl_pts import phi_ctrl_pts  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
LIMIAR_RELEVANCIA = 0.8


def extrair_meta_features(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    y_serie = df.iloc[:, 0]
    y = y_serie.to_numpy()
    X = df.iloc[:, 1:]

    n_instances, n_features = X.shape

    medias = X.mean(axis=0)
    desvios = X.std(axis=0)
    assimetrias = X.apply(skew)
    curtoses = X.apply(kurtosis)

    params_phi = phi_ctrl_pts(y)
    relevancia = np.array(phi(y_serie, params_phi))
    relevante = relevancia >= LIMIAR_RELEVANCIA
    mediana_y = np.median(y)
    pct_relevante = float(relevante.mean())
    pct_relevante_alta = float((relevante & (y > mediana_y)).mean())
    pct_relevante_baixa = float((relevante & (y < mediana_y)).mean())

    return {
        "dataset": csv_path.stem,
        "n_instances": n_instances,
        "n_features": n_features,
        "instances_per_feature": n_instances / n_features,
        "mean_of_means": medias.mean(),
        "mean_of_stds": desvios.mean(),
        "mean_skewness": assimetrias.mean(),
        "mean_kurtosis": curtoses.mean(),
        "y_skewness": skew(y),
        "y_kurtosis": kurtosis(y),
        "y_coef_variacao": y.std() / y.mean() if y.mean() != 0 else np.nan,
        "pct_relevante": pct_relevante,
        "pct_relevante_alta": pct_relevante_alta,
        "pct_relevante_baixa": pct_relevante_baixa,
    }


def main():
    linhas = [extrair_meta_features(p) for p in sorted(DATA_DIR.glob("*.csv"))]
    X = pd.DataFrame(linhas).set_index("dataset")
    saida = Path(__file__).parent / "X.csv"
    X.to_csv(saida)
    print(X.round(3))
    print(f"\nSalvo em {saida}")


if __name__ == "__main__":
    main()
