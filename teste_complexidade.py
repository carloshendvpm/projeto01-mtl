"""Confere a implementacao contra o exemplo numerico da Etapa 1.

Secao 5 do levantamento (pratica-02-levantamento-regressao-desbalanceada.md):
dois datasets com os mesmos resumos marginais e dificuldades locais diferentes.
Rodar com: python3 teste_complexidade.py
"""

import numpy as np

from complexidade import C2, S1, S1_sd, _minmax, erros_1nn

Z = np.array([0, 1, 4, 5, 9, 10, 15, 16, 22, 23], dtype=float).reshape(-1, 1)
Y_A = np.array([0, 0, 0, 0, 0, 0, 0, 0, 10, 10], dtype=float)
Y_B = np.array([0, 10, 0, 0, 0, 0, 0, 0, 0, 10], dtype=float)

# phi manual do exemplo: phi(0) = 0, phi(10) = 1, logo R = {i : y_i = 10}
ESPERADO = {
    "A": {"C2": 0.696311, "S1": 1 / 9, "S1_sd": 1 / 3, "S3": 0.0, "S3_R": 0.0},
    "B": {"C2": 0.087039, "S1": 3 / 9, "S1_sd": 0.5, "S3": 0.4, "S3_R": 1.0},
}


def medir(y):
    Zn = _minmax(Z)
    yn = _minmax(y.reshape(-1, 1)).ravel()
    e = erros_1nn(Zn, yn)
    return {
        "C2": C2(Zn, yn),
        "S1": S1(Zn, yn),
        "S1_sd": S1_sd(Zn, yn),
        "S3": float(e.mean()),
        "S3_R": float(e[y == 10].mean()),
    }


def main():
    falhas = 0
    for nome, y in (("A", Y_A), ("B", Y_B)):
        obtido = medir(y)
        for medida, esperado in ESPERADO[nome].items():
            ok = np.isclose(obtido[medida], esperado, atol=1e-6)
            falhas += not ok
            print(f"{'ok ' if ok else 'FALHA'} {nome}.{medida}: "
                  f"{obtido[medida]:.6f} (esperado {esperado:.6f})")
    print("\nTudo confere." if not falhas else f"\n{falhas} divergencia(s).")
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
