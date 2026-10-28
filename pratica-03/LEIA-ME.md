# Prática 3 — Warm-start de HPO em regressão desbalanceada

O experimento usa Random Forest, 30 datasets, 60 configurações e validação cruzada com dez folds. A perda é SERA. Cada busca tem 20 avaliações, cinco pontos iniciais e vinte sementes. As estratégias são padrão, random search, SMBO frio, rank médio e similaridade (Xbase/Xestendido, k=1,3,5).

- [Relatório em PDF](relatorio.pdf) e [fonte LaTeX](relatorio.tex).
- [Notebook com código e saídas executadas](pratica3_hpo_rf.ipynb).
- [Matriz de perdas 60 × 30](resultados/matriz_perdas.csv).
- [Perdas e tempos dos 18.000 ajustes](resultados/perdas_por_fold.csv).
- [Grade de configurações](resultados/configuracoes.json).
- [Trajetórias completas das buscas](resultados/buscas.csv).
- [AUC por dataset, semente e método](resultados/auc.csv).
- [Friedman](resultados/friedman.csv), comparações por orçamento e pós-testes de Nemenyi em `resultados/`.
- [Comparação de X e k](resultados/resumo_x_k.csv), [ganhos por dataset](resultados/falhas.csv) e [vizinhos/configurações dos diagnósticos](resultados/diagnosticos.csv).
- [Versões das bibliotecas](resultados/versoes.json) e [verificação da equivalência da SERA](resultados/equivalencia_sera.csv).

## Reproduzir

Instale `python3 -m pip install -r pratica-03/requirements.txt` a partir da raiz. Abra o notebook e execute as células em ordem. Com `EXECUTAR_MATRIZ = False` e `EXECUTAR_BUSCAS = False`, ele carrega os resultados salvos e refaz as análises, sem repetir os treinos.

Para reconstruir a matriz, ative `EXECUTAR_MATRIZ = True`. O notebook usa seis processos, com uma thread por Random Forest, e salva cada configuração concluída em `resultados/checkpoints/`. Para reconstrução integral desde zero, use uma cópia limpa do repositório. Checkpoints locais só podem ser reaproveitados com os mesmos dados, grade, folds, sementes, versões e implementação. Eles não são necessários para consultar a matriz consolidada e não são publicados.

Para repetir as buscas, ative `EXECUTAR_BUSCAS = True`. Elas consultam a matriz, sem novos treinos do regressor. O alvo é excluído do cálculo dos ranks, seleção de vizinhos, imputação e padronização. A normalização das perdas pelo mínimo/máximo do alvo só ocorre após a busca, para a análise.

A SERA reutiliza por fold os pesos da integração trapezoidal de relevância, com o mesmo passo 0,01 da implementação anterior. A linearidade da soma permite calcular o produto desses pesos pelos erros quadráticos sem repetir 101 vezes a estimação da relevância. A equivalência com a função legada foi conferida nos dez folds do piloto.

O código está no notebook e reutiliza `../data/`, as matrizes `../meta-dataset/{X,X_complexidade,P}.csv` e as funções de relevância de `../lib/`. Não misture tabelas de outras grades.

Para recompilar o relatório, execute `pdflatex -interaction=nonstopmode -halt-on-error relatorio.tex` duas vezes dentro de `pratica-03/`.

## Interpretação

O warm-start principal (Xestendido, k=3) reduziu a AUC normalizada de 0,0705 no SMBO frio para 0,0206, uma redução de 70,8% (Nemenyi, p=0,0014). A vantagem foi significativa após cinco avaliações, mas não nos orçamentos 12 e 20. O ganho está concentrado no início da busca.

A comparação com rank médio não foi significativa na AUC nem nos três orçamentos. Assim, os resultados são compatíveis com o benefício de um portfólio de configurações geralmente boas; não isolam uma contribuição adicional das meta-características. Xestendido melhorou a AUC média em k=1 e k=3, mas piorou em k=5. A menor AUC descritiva foi de Xbase/k=5 (0,0165), sem substituir o braço principal definido previamente.

A similaridade principal melhorou a AUC em 29 datasets e piorou em a2. Os maiores ganhos ocorreram em a3, kdd_coil_1 e a7. Em a2, as iniciais transferidas privilegiaram folhas de tamanho 1 ou 2, enquanto o ótimo da grade usou max_features=0,5 e min_samples_leaf=4; ele não foi avaliado pelo warm-start em vinte passos. A distância ao vizinho mais próximo não apresentou associação significativa com o ganho (Spearman rho=0,150; p=0,428).

Para a pesquisa em seleção dinâmica de modelos para regressão desbalanceada, o resultado recomenda manter um controle por portfólio e verificar se descritores por grupo predizem diferenças de desempenho, além de descreverem os dados. O efeito observado nesta busca de um regressor fixo não estabelece o efeito das mesmas medidas sobre competência local ou seleção dinâmica.


As sementes são agregadas por dataset antes dos testes. Não significância não demonstra equivalência. A comparação de todas as variantes X/k é descritiva; os testes principais usam os cinco métodos, com Xestendido/k=3 como braço de similaridade definido previamente. O ótimo é restrito à grade e a perda após seleção não é uma estimativa externa da generalização.
