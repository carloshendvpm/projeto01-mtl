# Medidas de complexidade separadas por grupo (raros vs. normais)

**IN1097 · Prática 2, Etapa 2 · execução de 18/09/2026**

Esta é a execução do que a [Etapa 1](pratica-02-levantamento-regressao-desbalanceada.md)
levantou em 17/09. A sugestão da professora — calcular as medidas de complexidade
separadamente para dados normais e raros, além do dataset inteiro — é a
generalização da adaptação proposta na ficha 4.6 daquele documento (`S3` relevante):
lá, uma medida agregada na região relevante; aqui, todas as medidas nos três grupos.

Aqui há **extração e caracterização**. A ligação com `P` foi testada no mesmo dia,
em `avaliacao_x_estendido.py`, e **o resultado é negativo**: nenhuma variante de `X`
estendido melhora a recomendação (nenhum p abaixo de 0,10). Ver a seção 6 abaixo e
a seção 11 do documento da prática.

## 1. O que foi implementado

- **`complexidade.py`** — as 12 medidas de complexidade para regressão de
  Lorena et al. (2018): C1–C4 (correlação), L1–L3 (linearidade), S1/S1_sd/S2–S4
  (suavidade), T2 (dimensionalidade). Cobre os seis descritores selecionados na
  seção 3 da Etapa 1 (`C2.mean`, `C3.min`, `S1.mean`, `S1.sd`, `S3` global,
  `S3` relevante) e mais seis, que saem de graça da mesma base de cálculo.
- **`gerar_complexidade.py`** — para cada um dos 30 datasets: aplica φ(y),
  define R = {i : φ(yᵢ) ≥ 0,8}, e calcula as medidas **três vezes**
  (todos / normais / raros).
- **`teste_complexidade.py`** — confere a implementação contra o exemplo
  numérico da seção 5 da Etapa 1. Os dez valores batem até 1e-6: `C2` 0,696311
  e 0,087039; `S1.mean` 1/9 e 3/9; `S1.sd` 1/3 e 0,5; `S3` 0 e 0,4;
  `S3` relevante 0 e 1.
- **`analisar_complexidade.py`** — Wilcoxon pareado entre grupos, Spearman entre
  a mesma medida em grupos diferentes (redundância) e comparação das duas
  variantes de `S3`.

### Convenções, e onde elas seguem a Etapa 1

| Convenção da Etapa 1 | Implementação |
|---|---|
| Min–max de Z̃ e ỹ para [0,1]; y não entra na distância | segue; a normalização é feita **uma vez, no dataset inteiro, antes da divisão** — normalizar cada grupo à parte poria os três em escalas diferentes |
| φ sobre o y original, R = φ ≥ 0,8 | segue (φ do projeto, boxplot de Tukey, 1,5, ambas as caudas) |
| Distância euclidiana, arestas únicas da MST, desvio amostral | segue (§4.4 e nota 3 da §6) |
| Empate de distância resolve pelo menor índice | segue (`argmin`) |
| Correlação com atributo constante é **indefinida, não zero** | segue: `_rho` devolve NaN, C1/C2 agregam só os válidos e `n_atributos_validos` registra quantos entraram (§4.2 e nota 5 da §6) |
| Casos indefinidos registrados, não tratados como fáceis | grupo com menos de 10 exemplos ficaria NaN — não ocorreu; o menor grupo raro tem 10 (`cocomo_numeric`, `pdgfr`) |

**Desvio assumido:** subamostragem estratificada em 1.500 exemplos nos 6 datasets
grandes, porque S1/S3 exigem matriz de distância O(n²) — exatamente o custo
quadrático antecipado na §6. A proporção de raros é preservada e a coluna
`subamostrado` marca quais datasets foram afetados.

Saídas: `complexidade.csv` (longo), `X_complexidade.csv` (largo, pronto para
estender `X`), `complexidade_comparacao.csv`, `complexidade_grupos.png`.

## 2. As duas variantes de `S3` — e a Etapa 1 estava certa

A ficha 4.6 define `S3` relevante como: **o vizinho é buscado no dataset
inteiro**, e só a agregação é restrita a R. E adverte que restringir também os
candidatos "definiria outra medida, mais otimista quanto à disponibilidade de
vizinhos relevantes".

Calcular a medida *dentro* do grupo raro — que é a leitura literal de "medidas
calculadas separadamente para os raros" — é justamente essa outra medida. Ambas
foram calculadas (`S3` = dentro do grupo, `S3_viz_global` = definição da ficha):

| Grupo | `S3` dentro do grupo | `S3` vizinho global | ρ entre elas | p (Wilcoxon) |
|---|---:|---:|---:|---:|
| normais | 0,0076 | 0,0145 | 0,89 | <0,0001 |
| raros | 0,0286 | 0,0690 | 0,77 | <0,0001 |

A advertência se confirma nos 30 datasets: restringir os candidatos a vizinho
**reduz o erro dos raros à metade** (0,0286 contra 0,0690). Faz sentido — um
raro cercado só de casos comuns é previsto mal, mas se ele só puder consultar
outros raros, encontra alguém parecido. A versão da ficha é a que mede a
pergunta original ("os raros são difíceis no dataset como ele é?"); a versão
dentro do grupo mede a coesão interna da região rara. São perguntas diferentes,
e as duas ficam no CSV.

## 3. Os grupos são diferentes?

Médias sobre os 30 datasets e Wilcoxon pareado raros × normais:

| Medida | todos | normais | raros | p |
|---|---:|---:|---:|---:|
| C1 | 0,579 | 0,546 | 0,533 | 0,685 |
| C2 | 0,296 | 0,276 | 0,243 | 0,428 |
| C3 | 0,284 | 0,318 | 0,322 | 0,871 |
| C4 | 0,121 | 0,050 | 0,104 | 0,082 |
| L1 | 0,069 | 0,041 | 0,067 | **0,011** |
| L2 | 0,0140 | 0,0043 | 0,0128 | **0,005** |
| L3 | 0,0123 | 0,0038 | 0,0105 | **0,011** |
| S1 | 0,079 | 0,052 | 0,103 | **0,0001** |
| S1_sd | 0,101 | 0,050 | 0,121 | **<0,0001** |
| S2 | 1,224 | 1,216 | 1,249 | 0,919 |
| S3 (dentro) | 0,021 | 0,008 | 0,029 | **0,0002** |
| S4 | 0,0095 | 0,0029 | 0,0068 | 0,067 |
| T2 | 0,201 | 0,231 | 1,560 | **<0,0001** |
| S3 (viz. global) | 0,021 | 0,015 | 0,069 | **<0,0001** |

Sim, e de forma coerente: **as medidas de erro e suavidade (L1, L2, L3, S1,
S1_sd, S3) são significativamente maiores na região rara**. A região que importa
é localmente menos linear e menos suave — que é exatamente a hipótese da abertura
do roteiro de fala da Etapa 1 ("hoje eu só sei quantos casos raros existem — não
sei se são fáceis ou difíceis de prever"). A resposta empírica: são mais
difíceis, e o `S3` da ficha 4.6 é o que mostra isso com mais força (0,069 contra
0,015 nos normais, um fator de 4,8).

Duas leituras adicionais:

- **A média global já é quase a dos raros.** Em L1, "todos" dá 0,069 e os raros
  0,067, contra 0,041 dos normais. O erro agregado do dataset inteiro é em boa
  parte o erro dos poucos raros — o contrário do que a §4.5 temia ("a média pode
  ser pequena porque os numerosos casos comuns são fáceis"). Nesses 30 datasets,
  os raros puxam a média global para cima, não são diluídos por ela.
- **As correlações (C1, C2, C3) não diferem entre grupos.** Consistente com o
  que são: medidas de ordenação global, pouco sensíveis à faixa de y observada.
  Para caracterizar a região rara, a família de suavidade é a que fala.

## 4. A separação traz informação nova?

Se a medida no grupo raro fosse só uma cópia da medida no conjunto todo, ρ ≈ 1 e
nada se ganharia como meta-característica.

| Medida | ρ(raro, todos) | ρ(raro, normal) | ρ(normal, todos) |
|---|---:|---:|---:|
| C1 | 0,66 | 0,69 | 0,98 |
| C2 | 0,69 | 0,72 | 0,96 |
| C3 | 0,72 | 0,76 | 0,97 |
| C4 | 0,25 | −0,20 | 0,54 |
| L1 | 0,59 | 0,36 | 0,85 |
| L2 | 0,66 | 0,29 | 0,74 |
| L3 | 0,69 | 0,36 | 0,78 |
| S1 | 0,71 | 0,54 | 0,94 |
| S1_sd | 0,61 | 0,13 | 0,71 |
| S2 | 0,96 | 0,94 | 0,99 |
| S3 (dentro) | 0,78 | 0,42 | 0,79 |
| S4 | 0,62 | 0,33 | 0,69 |
| T2 | 0,92 | 0,89 | 0,99 |
| S3 (viz. global) | 0,97 | 0,89 | 0,94 |

1. **A versão "normal" é quase redundante** com a versão "todos" (ρ de 0,54 a
   0,99, mediana ≈ 0,94). Era de esperar: os normais são a maioria dos exemplos,
   caracterizá-los é quase caracterizar o dataset inteiro. Não vale gastar
   colunas de `X` com elas.
2. **A versão "raro" muda a ordenação dos datasets.** ρ(raro, normal) fica entre
   −0,20 e 0,54 em C4, L1, L2, L3, S1_sd, S3 e S4: um dataset pode ser fácil na
   região normal e difícil na rara, e vice-versa.
3. **Tensão entre as duas variantes de `S3`.** A da ficha 4.6 separa muito melhor
   os níveis (item 3 acima), mas ordena os datasets quase como o `S3` global
   (ρ = 0,97) — como meta-característica, acrescenta pouca ordenação nova. A
   versão dentro do grupo ordena diferente (ρ = 0,78), mas é a medida otimista.
   Qual delas ajuda a prever `P` é pergunta empírica, não de definição: ambas
   entram no teste do passo seguinte.
4. **`T2_raro` deve sair.** Média 1,56 (mais atributos que exemplos) — no grupo
   raro ela vira essencialmente um contador de raros, duplicando `pct_relevante`,
   que já está em `X`.

**Recomendação para estender `X`:** acrescentar as versões `_raro` de C4, L1, L2,
L3, S1, S1_sd, S3 e S3_viz_global; deixar de fora as `_normal` e `T2_raro`.

## 5. Ressalvas

- O grupo raro é pequeno em vários datasets (mediana 97 exemplos, mínimo 10).
  As medidas ali têm variância alta; `n_raro` está no CSV para filtrar ou ponderar.
- Os raros incluem extremos altos **e** baixos no mesmo grupo, quando φ é bimodal.
  Um grupo bimodal é, por construção, disperso — parte do S1/S3 alto pode vir daí.
  `X` já tem `pct_relevante_alta` e `pct_relevante_baixa`; separar as duas caudas
  também nas medidas é o próximo teste natural.
- A subamostragem afeta 6 datasets; ali os valores são estimativas.
- A ligação com `P` foi testada e não se sustentou — ver a seção 6.

## 6. O teste de recomendação: resultado negativo

`avaliacao_x_estendido.py` refaz o leave-one-dataset-out do Projeto 1 com quatro
variantes de `X`: `base` (13 colunas), `+compl_all` (21, controle), `+compl_raro`
(21) e `+ambos` (29). As 8 medidas acrescentadas saíram da recomendação da seção 4,
que usa apenas o lado `X` — nenhuma coluna foi escolhida olhando `P` ou `R`.

O controle `+compl_all` existe para separar duas explicações possíveis de um ganho:
a separação por grupo ajudar, ou as medidas de complexidade ajudarem.

| Abordagem | base | +compl_all | +compl_raro | +ambos |
|---|---:|---:|---:|---:|
| Abordagem 1 (Spearman) | 0,543 | 0,533 | 0,558 | 0,541 |
| Abordagem 2 (Spearman) | 0,609 | 0,615 | 0,598 | 0,617 |
| HARRIS (Spearman) | 0,621 | 0,621 | 0,611 | 0,602 |

**Nenhum p do Wilcoxon pareado contra a base fica abaixo de 0,10.** As variações
são de ±0,02 e o sinal nem é consistente entre abordagens.

Dois pontos de contexto:

- **As abordagens de agregação, que ignoram `X`, obtêm 0,616 (AR e Vitórias
  Significativas)** — acima da Abordagem 1 e empatadas com HARRIS. Um ranking fixo
  já iguala o melhor meta-modelo, o que deixa pouca margem para qualquer
  meta-característica nova aparecer.
- **A floresta usa as colunas novas e mesmo assim não melhora.** No diagnóstico de
  importância (`importancia_meta_features.csv`, ajustado aos 30 datasets), `L1_all`
  é a característica mais importante de todas (0,158) e as 8 colunas do dataset
  inteiro concentram 37% da importância. Importância não é ganho — que é a mesma
  ressalva que a Etapa 1 fez sobre a análise do Meta-IR.

Com 30 exemplos no meta-nível e 13→29 colunas, o regime é adverso a qualquer coluna
nova. A continuação natural é testar uma ou duas colunas apenas, ou seleção de
características dentro de cada partição de treino (nota 4 da §6 da Etapa 1).
