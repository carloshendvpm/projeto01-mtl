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
- Análises da versão 2: [comparações controladas](resultados/comparacoes_controladas.csv), [sensibilidade sem `california`](resultados/sensibilidade_sem_california.csv), [ponto de equilíbrio](resultados/ponto_equilibrio.csv), [Spearman com Holm](resultados/spearman_ganho.csv) e [exemplos raros por fold de teste](resultados/raros_por_fold_teste.csv).
- [Meta-características por fold](resultados/meta_por_fold.jsonl): Xbase e Xestendido de cada dataset calculados só no treino de cada fold, com os tempos.
- [Versão 1](resultados/versao_1/): tabelas e relatório anteriores à correção do vazamento, mantidos para comparação.
- [Figuras](resultados/figuras/): diagramas de diferença crítica, custo e desempenho, perda por dataset.

## Scripts

| Script | O que faz |
|---|---|
| `comum.py` | Caminhos, folds, SERA e regressores padrão |
| `executar_baselines.py` | Seis regressores de P e DESReg |
| `desreg_cache.py` | Predição vetorizada da DESReg, equivalente ao `predict` original (ver `testes/`) |
| `meta_por_fold.py` | Xbase e Xestendido do alvo calculados só no treino de cada fold |
| `mtl.py` | Recomendação da MtL por fold, em leave-one-dataset-out |
| `executar_warmstart.py` | Busca com warm-start da Prática 3, com vizinhos e validação interna só no treino |
| `executar_automl.py` | FLAML e AutoGluon otimizando a SERA; pula as execuções já gravadas |
| `rodar_tudo.sh` | Roda o warm-start e todas as execuções de AutoML |
| `construir_notebook.py` | Gera o notebook de análise |

## Reproduzir

```bash
cd pratica-04
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest testes
.venv/bin/python executar_baselines.py --jobs 6
./rodar_tudo.sh                      # inclui meta_por_fold.py
.venv/bin/python construir_notebook.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace pratica4_automl_vs_mtl.ipynb
pdflatex relatorio.tex && pdflatex relatorio.tex
```

O `rodar_tudo.sh` levou cerca de 24 h somando todas as execuções. No macOS, o XGBoost precisa de `brew install libomp`. O código usa também `../data/`, `../meta-dataset/`, `../lib/`, `../codigo/` e os resultados das Práticas 1 a 3.

## Versão 2: correção do vazamento

Na versão 1, a MtL e o Warm-start descreviam o dataset alvo com meta-características calculadas no dataset inteiro, incluindo o fold de teste. Agora elas são recalculadas no treino de cada fold (`meta_por_fold.py`, cerca de 40 min com 7 processos). A MtL é refeita por inteiro a partir de `base_por_fold.csv`. No Warm-start, a busca é determinística dado o treino e as iniciais; para reproduzir a versão 2 a partir da 1:

```bash
.venv/bin/python executar_warmstart.py --jobs 6 --reaproveitar
```

Os folds cujas iniciais não mudaram copiam a linha da versão 1 (coluna `reaproveitado`); os demais refazem a busca. O teste `test_warmstart_reaproveitado` refaz alguns folds reaproveitados e confere a SERA. Os testes de `testes/test_isolamento.py` trocam o teste do fold por ruído e conferem que descritores, recomendação, vizinhos e busca não mudam.

## Observações

O FLAML 2.8 com `starting_points="data"` lança `KeyError` quando o portfólio sugere um hiperparâmetro que não existe no espaço de busca. Em `executar_automl.py`, esses pontos são tratados como inválidos. As 28 execuções que falharam antes da correção continuam registradas nos JSONL e foram refeitas.

O FLAML sem portfólio não foi rodado a 120 s, por falta de tempo. As execuções rodaram num Apple M2 com 8 núcleos e 8 GB, com quatro execuções de AutoML simultâneas de 2 CPUs cada. Os tempos são de relógio.

O portfólio zero-shot do FLAML inclui o California Housing (`houses`, ponto de 18.576 × 8, que é 90% de 20.640), o mesmo dataset que `california`.

## Resultado

Com 5 s por fold, Warm-start, Melhor médio e MtL ficam à frente dos AutoML em rank. A partir de 30 s, o AutoGluon lidera; a 120 s, tem SERA menor que a da MtL em 20 dos 30 datasets (ganho mediano de 15%), mas a diferença não é significativa pelo Nemenyi ($p=0,26$) e fica no limite pelo Wilcoxon com Holm ($p=0,068$). A MtL não se distingue do Melhor médio, e o portfólio do FLAML não tem efeito significativo. A MtL custa 1,4 s por fold, e seu custo offline se paga em 13 ou 14 datasets novos contra os AutoML de 5 s em tempo de relógio (6 ou 7 em núcleo-segundos). A correção do vazamento mudou a recomendação da MtL em 8 dos 300 folds e as iniciais do Warm-start em 97, sem mudar as conclusões.
