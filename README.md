# Práticas de meta-aprendizagem (IN1097)

Atividades práticas da disciplina IN1097, sobre meta-aprendizagem em regressão desbalanceada. Todas usam os mesmos 30 datasets, a métrica SERA e as mesmas dez partições (KFold com embaralhamento e semente 42).

| Pasta | Conteúdo |
|---|---|
| [`pratica-01/`](pratica-01/README.md) | Recomendação de regressores por meta-aprendizagem |
| [`pratica-02/`](pratica-02/LEIA-ME.md) | Meta-características de complexidade para regressão desbalanceada |
| [`pratica-03/`](pratica-03/LEIA-ME.md) | Warm-start da busca de hiperparâmetros do Random Forest |
| [`pratica-04/`](pratica-04/LEIA-ME.md) | AutoML (FLAML e AutoGluon) versus meta-aprendizagem |
| [`data/`](data/) | Os 30 datasets originais (a primeira coluna é o alvo) |
| [`meta-dataset/`](meta-dataset/README.md) | Matrizes X, P e R compartilhadas pelas práticas |
| [`codigo/`](codigo/README.md) | Scripts das Práticas 1 e 2 |
| [`lib/`](lib/) | Funções comuns: relevância, SERA e medidas de complexidade |
| [`testes/`](testes/) | Testes das funções de `lib/` |

Cada prática tem um relatório em PDF, o código e os resultados. As instruções para reproduzir estão no LEIA-ME de cada pasta.

Para rodar os testes, a partir desta pasta:

```bash
python3 -m pytest testes -q
```
