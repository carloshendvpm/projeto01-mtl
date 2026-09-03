from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent

resultados = pd.read_csv(BASE_DIR / "resultados_por_fold.csv")

P = resultados.pivot_table(index="dataset", columns="algoritmo", values="sera", aggfunc="mean")

R = P.rank(axis=1, method="average", ascending=True)

P.to_csv(BASE_DIR / "P.csv")
R.to_csv(BASE_DIR / "R.csv")

print("P (SERA médio):")
print(P.round(2))
print("\nR (ranks, 1 = melhor):")
print(R)
