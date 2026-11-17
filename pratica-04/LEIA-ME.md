# Prática 4 — AutoML versus meta-aprendizagem em regressão desbalanceada

Este experimento usa os mesmos 30 datasets, a mesma SERA e as mesmas 10 partições das práticas anteriores: KFold com embaralhamento e semente 42.

Comparamos os seguintes métodos:

- Melhor médio.
- MtL (Abordagem 2 da Prática 1, em LODO).
- Warm-start (SMBO da Prática 3, refeito dentro de cada fold).
- DESReg e DESReg-φ.
- Três variantes de AutoML: FLAML com portfólio zero-shot, FLAML com pontos iniciais estáticos e AutoGluon `medium_quality`. Cada uma roda com 5, 30 e 120 s por fold.
- Oráculo dos seis regressores.

## Arquivos

- [Relatório em PDF](relatorio.pdf) e [fonte LaTeX](relatorio.tex).
- [Notebook com as análises e saídas executadas](pratica4_automl_vs_mtl.ipynb), gerado por `construir_notebook.py`.
- Resultados brutos por dataset e fold:
  - [AutoML, uma linha JSON por execução](resultados/automl/). Cada linha guarda a SERA, o tempo, o modelo escolhido, a configuração, os pesos do ensemble e os pontos iniciais.
  - [Seis regressores de P reexecutados](resultados/base_por_fold.csv).
  - [DESReg](resultados/desreg_por_fold.csv).
  - [Warm-start](resultados/warmstart_por_fold.csv).
  - [Todos os métodos juntos](resultados/todos_por_fold.csv).
- Tabelas consolidadas:
  - [SERA média por dataset](resultados/sera_por_dataset.csv).
  - [Perda relativa](resultados/perda_relativa_por_dataset.csv).
  - [Resumo por método](resultados/resumo_metodos.csv).
  - [Custo × desempenho](resultados/custo_desempenho.csv).
  - [Friedman](resultados/friedman.csv) e Nemenyi por orçamento (`resultados/nemenyi_*.csv`).
  - [Ganho do AutoML por dataset](resultados/ganho_automl_por_dataset.csv).
- [Figuras](resultados/figuras/): diagramas CD por orçamento, custo × desempenho e perda por dataset.

## Scripts

| Script | O que faz |
|---|---|
| `comum.py` | Protocolo comum: caminhos, folds, SERA e regressores padrão. |
| `executar_baselines.py` | Seis regressores de P e DESReg, nas mesmas partições. |
| `desreg_cache.py` | Predição vetorizada da DESReg, equivalente ao `predict` original (`testes/`). |
| `executar_warmstart.py` | SMBO com warm-start da Prática 3, dentro de cada fold e com CV interna só no treino. |
| `executar_automl.py` | FLAML e AutoGluon otimizando a SERA. Pode ser retomado: pula as execuções já gravadas. |
| `rodar_tudo.sh` | Cadeia completa das execuções de AutoML. |
| `construir_notebook.py` | Gera o notebook de análise. |

## Reproduzir

```bash
cd pratica-04
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest testes            # equivalência do cache da DESReg
.venv/bin/python executar_baselines.py --jobs 6
./rodar_tudo.sh                              # warm-start + AutoML (cerca de 24 h de parede somadas)
.venv/bin/python construir_notebook.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace pratica4_automl_vs_mtl.ipynb
pdflatex relatorio.tex && pdflatex relatorio.tex
```

O código reutiliza estes recursos da raiz do repositório:

- `../data/` (os datasets);
- `../meta-dataset/` (X, P e R);
- `../lib/` (relevância e SERA);
- `../codigo/` (meta-modelo da Prática 1 e extração de Xbase);
- `../pratica-02/resultados/` e `../pratica-03/resultados/` (Xestendido, matriz de perdas e custos).

No macOS, o XGBoost exige `brew install libomp`.

## Observações

- **Correção no FLAML.** O FLAML 2.8 com `starting_points="data"` lança `KeyError` quando o portfólio sugere um hiperparâmetro ausente no espaço de busca. `executar_automl.py` trata esses pontos como inválidos. As 28 execuções que falharam antes da correção ficaram registradas no JSONL e foram refeitas.
- **Execução cortada.** A variante FLAML sem meta-conhecimento a 120 s não foi executada, por tempo.
- **Ambiente.** As execuções rodaram num Apple M2 com 8 núcleos e 8 GB: 4 execuções de AutoML simultâneas, com 2 CPUs cada. Os tempos registrados são de parede, com `perf_counter`.
- **Sobreposição de datasets.** O portfólio zero-shot do FLAML contém `houses` (California Housing; ponto de 18.576 × 8, isto é, 90% de 20.640), que é o nosso `california`.

## Resultado em uma linha

Até 5 s por fold, o meta-conhecimento offline (MtL e Warm-start) fica à frente dos AutoML. A partir de 30 s, o AutoGluon lidera: com 120 s, tem rank 2,7 contra 3,9 da MtL, uma diferença não significativa pelo Nemenyi, e supera o oráculo dos seis regressores em 18 de 30 datasets.

A MtL custa 1,3 s por fold. Frente aos AutoML de 5 s, que têm desempenho parecido, seu custo offline (0,26 h de CPU) se paga a partir de cerca de 13 datasets novos.
