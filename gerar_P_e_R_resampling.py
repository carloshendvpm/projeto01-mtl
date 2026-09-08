from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent

resultados = pd.read_csv(BASE_DIR / "resultados_por_fold_resampling.csv")

P = resultados.pivot_table(index="dataset", columns="algoritmo", values="sera", aggfunc="mean")

R = P.rank(axis=1, method="average", ascending=True)

P.to_csv(BASE_DIR / "P_resampling.csv")
R.to_csv(BASE_DIR / "R_resampling.csv")

print("P_resampling (SERA médio):")
print(P.round(2))
print("\nR_resampling (ranks, 1 = melhor):")
print(R)
