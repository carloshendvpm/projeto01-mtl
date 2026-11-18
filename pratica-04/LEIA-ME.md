# Prática 4: AutoML versus meta-aprendizagem em regressão desbalanceada

Comparo o meta-conhecimento das Práticas 1 a 3 com sistemas de AutoML nos mesmos 30 datasets, com a SERA e as mesmas dez partições (KFold, embaralhamento, semente 42). O relatório está em [relatorio.pdf](relatorio.pdf).

## Métodos

- Melhor médio: regressor de melhor rank médio nos outros 29 datasets;
- MtL: Abordagem 2 da Prática 1, em leave-one-dataset-out, recomendando por fold;
- Warm-start: busca da Prática 3, refeita dentro de cada fold;
- DESReg e DESReg-φ, sobre o pool de bagging dos seis regressores;
- FLAML com e sem portfólio zero-shot e AutoGluon `medium_quality`, com 5, 30 e 120 s por fold;
- oráculo dos seis regressores, só como referência.

Em todos, o teste do fold só é usado para medir a SERA. As meta-características do alvo são calculadas no treino de cada fold (`meta_por_fold.py`), e `testes/test_isolamento.py` confere que trocar o teste por ruído não muda descritores, recomendação, vizinhos nem a busca.

## Arquivos

| Script | O que faz |
|---|---|
| `comum.py` | Caminhos, folds, SERA e regressores padrão |
| `executar_baselines.py` | Seis regressores de P e DESReg |
| `desreg_cache.py` | Predição vetorizada da DESReg, equivalente ao `predict` original |
| `meta_por_fold.py` | Xbase e Xestendido do alvo calculados no treino de cada fold |
| `mtl.py` | Recomendação da MtL por fold |
| `executar_warmstart.py` | Warm-start com vizinhos e validação interna só no treino |
| `executar_automl.py` | FLAML e AutoGluon otimizando a SERA |
| `rodar_tudo.sh` | Encadeia meta-características, warm-start e AutoML |
| `construir_notebook.py` | Gera o [notebook de análise](pratica4_automl_vs_mtl.ipynb) |

O código usa também `../data/`, `../meta-dataset/`, `../lib/`, `../codigo/` e os resultados das Práticas 1 a 3.

## Reproduzir

```bash
cd pratica-04
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest testes
.venv/bin/python executar_baselines.py --jobs 6
./rodar_tudo.sh
.venv/bin/python executar_warmstart.py --jobs 6 --reaproveitar --sem-parentes
.venv/bin/python construir_notebook.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace pratica4_automl_vs_mtl.ipynb
pdflatex relatorio.tex && pdflatex relatorio.tex
```

O `rodar_tudo.sh` levou cerca de 24 h, quase todas no AutoML; as meta-características por fold, cerca de 40 min com 7 processos. Os scripts pulam execuções já gravadas. No macOS, o XGBoost precisa de `brew install libomp`.

## Resultados

- Brutos, por dataset e fold: [AutoML](resultados/automl/) (uma linha JSON por execução), [regressores de P](resultados/base_por_fold.csv), [DESReg](resultados/desreg_por_fold.csv), [Warm-start](resultados/warmstart_por_fold.csv), [meta-características](resultados/meta_por_fold.jsonl) e [todos os métodos](resultados/todos_por_fold.csv).
- Tabelas: [SERA por dataset](resultados/sera_por_dataset.csv), [resumo](resultados/resumo_metodos.csv), [Friedman](resultados/friedman.csv) e Nemenyi (`nemenyi_*.csv`), [comparações por fator](resultados/comparacoes_controladas.csv), [custo](resultados/custo_desempenho.csv), [ponto de equilíbrio](resultados/ponto_equilibrio.csv), [Spearman](resultados/spearman_ganho.csv), sensibilidade [sem `california`](resultados/sensibilidade_sem_california.csv) e [sem datasets da mesma origem](resultados/sensibilidade_parentes.csv).
- [Figuras](resultados/figuras/).

## Versão anterior

Numa primeira versão, a MtL e o Warm-start descreviam o alvo com meta-características calculadas no dataset inteiro, incluindo o fold de teste. A correção mudou a recomendação da MtL em 8 dos 300 folds e as iniciais do Warm-start em 97, sem mudar as conclusões. As tabelas e o relatório dessa versão estão em [resultados/versao_1/](resultados/versao_1/). Com `--reaproveitar`, o Warm-start copia da versão 1 os folds cujas iniciais não mudaram (coluna `reaproveitado`); a busca é determinística, e `test_warmstart_reaproveitado` refaz alguns desses folds e confere a SERA.

## Limitações de reprodução

- Os tempos são de relógio, num Apple M2 com 8 núcleos e 8 GB, com quatro AutoML simultâneos de 2 CPUs cada; em outra máquina, o AutoML chega a modelos diferentes no mesmo orçamento.
- O FLAML 2.8 com `starting_points="data"` lança `KeyError` quando o portfólio sugere um hiperparâmetro fora do espaço de busca; `executar_automl.py` descarta esses pontos. As 28 execuções que falharam antes disso continuam nos JSONL.
- O FLAML sem portfólio não foi rodado a 120 s.
- As predições não foram guardadas, então não dá para recalcular a SERA com outra estimativa de relevância.
