# Prática 4: AutoML versus meta-aprendizagem em regressão desbalanceada

Comparo, nos mesmos 30 datasets, com a SERA e com as mesmas dez partições das práticas anteriores:

- Melhor médio;
- MtL (Abordagem 2 da Prática 1, em leave-one-dataset-out);
- Warm-start (busca da Prática 3, refeita dentro de cada fold);
- DESReg e DESReg-φ;
- FLAML com e sem portfólio zero-shot e AutoGluon `medium_quality`, com 5, 30 e 120 s por fold;
- o oráculo dos seis regressores, como referência.

## Arquivos

- [Relatório em PDF](relatorio.pdf) e [fonte LaTeX](relatorio.tex).
- [Notebook com as análises](pratica4_automl_vs_mtl.ipynb), gerado por `construir_notebook.py`.
- Resultados brutos por dataset e fold:
  - [AutoML](resultados/automl/): uma linha JSON por execução, com SERA, tempo, modelo escolhido, configuração, pesos do ensemble e pontos iniciais;
  - [seis regressores de P](resultados/base_por_fold.csv), [DESReg](resultados/desreg_por_fold.csv) e [Warm-start](resultados/warmstart_por_fold.csv);
  - [todos os métodos juntos](resultados/todos_por_fold.csv).
- Tabelas consolidadas: [SERA por dataset](resultados/sera_por_dataset.csv), [perda relativa](resultados/perda_relativa_por_dataset.csv), [resumo por método](resultados/resumo_metodos.csv), [custo e desempenho](resultados/custo_desempenho.csv), [Friedman](resultados/friedman.csv), Nemenyi (`resultados/nemenyi_*.csv`) e [ganho do AutoGluon sobre a MtL](resultados/ganho_automl_por_dataset.csv).
- [Figuras](resultados/figuras/): diagramas de diferença crítica, custo e desempenho, perda por dataset.

## Scripts

| Script | O que faz |
|---|---|
| `comum.py` | Caminhos, folds, SERA e regressores padrão |
| `executar_baselines.py` | Seis regressores de P e DESReg |
| `desreg_cache.py` | Predição vetorizada da DESReg, equivalente ao `predict` original (ver `testes/`) |
| `executar_warmstart.py` | Busca com warm-start da Prática 3, com validação interna só no treino |
| `executar_automl.py` | FLAML e AutoGluon otimizando a SERA; pula as execuções já gravadas |
| `rodar_tudo.sh` | Roda o warm-start e todas as execuções de AutoML |
| `construir_notebook.py` | Gera o notebook de análise |

## Reproduzir

```bash
cd pratica-04
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest testes
.venv/bin/python executar_baselines.py --jobs 6
./rodar_tudo.sh
.venv/bin/python construir_notebook.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace pratica4_automl_vs_mtl.ipynb
pdflatex relatorio.tex && pdflatex relatorio.tex
```

O `rodar_tudo.sh` levou cerca de 24 h somando todas as execuções. No macOS, o XGBoost precisa de `brew install libomp`. O código usa também `../data/`, `../meta-dataset/`, `../lib/`, `../codigo/` e os resultados das Práticas 1 a 3.

## Observações

O FLAML 2.8 com `starting_points="data"` lança `KeyError` quando o portfólio sugere um hiperparâmetro que não existe no espaço de busca. Em `executar_automl.py`, esses pontos são tratados como inválidos. As 28 execuções que falharam antes da correção continuam registradas nos JSONL e foram refeitas.

O FLAML sem portfólio não foi rodado a 120 s, por falta de tempo. As execuções rodaram num Apple M2 com 8 núcleos e 8 GB, com quatro execuções de AutoML simultâneas de 2 CPUs cada. Os tempos são de relógio.

O portfólio zero-shot do FLAML inclui o California Housing (`houses`, ponto de 18.576 × 8, que é 90% de 20.640), o mesmo dataset que `california`.

## Resultado

Com 5 s por fold, MtL e Warm-start ficam à frente dos AutoML. A partir de 30 s, o AutoGluon lidera; a 120 s, tem rank 2,7 contra 3,9 da MtL (diferença não significativa pelo Nemenyi) e supera o oráculo dos seis regressores em 18 dos 30 datasets. A MtL custa 1,3 s por fold, e seu custo offline (0,26 h de CPU) se paga a partir de cerca de 13 datasets novos, quando comparada aos AutoML de 5 s.
