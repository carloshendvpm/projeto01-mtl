# Medidas de complexidade separadas por grupo (raros vs. normais)

Sugestão da professora a partir da Etapa 1: além de caracterizar o dataset
inteiro, calcular as medidas de complexidade **separadamente** para os dados
normais e para os dados raros, e verificar se isso traz informação adicional.

## O que foi feito

- `complexidade.py` — implementação das 12 medidas de complexidade para
  regressão de Lorena et al. (2018): correlação (C1–C4), linearidade (L1–L3),
  suavidade (S1–S4) e dimensionalidade (T2).
- `gerar_complexidade.py` — para cada um dos 30 datasets: aplica φ(y) ao alvo,
  separa raros (φ ≥ 0,8) de normais e calcula as 12 medidas **três vezes**
  (todos / normais / raros).
- `analisar_complexidade.py` — compara os grupos (Wilcoxon pareado) e mede
  redundância (Spearman entre a mesma medida em grupos diferentes).

Decisões de implementação:

- **Normalização min-max única, no dataset inteiro, antes da divisão.** Se cada
  grupo fosse normalizado separadamente, os valores dos três grupos ficariam em
  escalas diferentes e a comparação perderia o sentido.
- **Subamostragem estratificada em 1.500 exemplos** nos datasets grandes
  (S1/S3 exigem matriz de distância O(n²)). A proporção de raros é preservada.
- Grupo com menos de 10 exemplos ficaria NaN — não ocorreu: o menor grupo raro
  tem 10 exemplos (`cocomo_numeric`, `pdgfr`).

Saídas: `complexidade.csv` (formato longo), `X_complexidade.csv` (formato largo,
36 colunas `MEDIDA_grupo`, pronto para entrar no meta-dataset X),
`complexidade_comparacao.csv` e `complexidade_grupos.png`.

## Resultado 1 — os grupos são diferentes?

Médias sobre os 30 datasets e p-valor do Wilcoxon pareado:

| Medida | todos | normais | raros | p (raro×normal) |
|---|---|---|---|---|
| C1 | 0,579 | 0,546 | 0,533 | 0,685 |
| C2 | 0,294 | 0,274 | 0,238 | 0,339 |
| C3 | 0,284 | 0,318 | 0,322 | 0,871 |
| C4 | 0,121 | 0,050 | 0,104 | 0,082 |
| L1 | 0,069 | 0,041 | 0,067 | **0,011** |
| L2 | 0,0140 | 0,0043 | 0,0128 | **0,005** |
| L3 | 0,0123 | 0,0038 | 0,0105 | **0,011** |
| S1 | 0,079 | 0,052 | 0,103 | **0,0001** |
| S2 | 1,224 | 1,216 | 1,249 | 0,919 |
| S3 | 0,021 | 0,008 | 0,029 | **0,0002** |
| S4 | 0,0095 | 0,0029 | 0,0068 | 0,067 |
| T2 | 0,201 | 0,231 | 1,560 | **<0,0001** |

Padrão claro e coerente: **as medidas de erro/suavidade (L1, L2, L3, S1, S3)
são significativamente maiores na região rara**. Ou seja, a região rara é
localmente mais difícil — menos linear e menos suave — do que a região normal.
E note que a média no conjunto **todo** fica sempre colada na média do grupo
**raro**, não na do grupo normal (L1: 0,069 todos vs. 0,067 raros vs. 0,041
normais). A dificuldade medida no dataset inteiro já é, em boa parte, a
dificuldade dos raros: os poucos exemplos raros dominam o erro agregado.

As medidas de correlação (C1–C3) não diferem entre grupos, o que faz sentido:
são medidas de ranking global, pouco sensíveis a qual faixa de y se observa.

## Resultado 2 — a separação traz informação nova?

Se a medida no grupo raro fosse só uma cópia da medida no conjunto todo, a
correlação de Spearman entre elas seria ≈ 1 e nada se ganharia como
meta-característica.

| Medida | ρ(raro, todos) | ρ(raro, normal) | ρ(normal, todos) |
|---|---|---|---|
| C1 | 0,66 | 0,69 | 0,98 |
| C2 | 0,69 | 0,72 | 0,96 |
| C3 | 0,72 | 0,76 | 0,97 |
| C4 | 0,25 | −0,20 | 0,54 |
| L1 | 0,59 | 0,36 | 0,85 |
| L2 | 0,66 | 0,29 | 0,74 |
| L3 | 0,69 | 0,36 | 0,78 |
| S1 | 0,71 | 0,54 | 0,94 |
| S2 | 0,96 | 0,94 | 0,99 |
| S3 | 0,78 | 0,42 | 0,79 |
| S4 | 0,62 | 0,33 | 0,69 |
| T2 | 0,92 | 0,89 | 0,99 |

Duas leituras:

1. **A versão "normal" é quase redundante** com a versão "todos" (ρ de 0,69 a
   0,99, mediana ≈ 0,94). Faz sentido: os normais são a maioria dos exemplos,
   então caracterizá-los é quase o mesmo que caracterizar o dataset inteiro.
2. **A versão "raro" carrega informação nova.** ρ(raro, normal) fica entre
   −0,20 e 0,54 para C4, L1, L2, L3, S3 e S4 — um dataset pode ser fácil na
   região normal e difícil na rara, e vice-versa; a ordenação dos datasets muda.
   S2 e T2 são as exceções (ρ alto), e T2 no grupo raro vira essencialmente
   um contador de quantos raros existem (média 1,56: mais atributos que
   exemplos), o que duplica `pct_relevante` — **descartar T2_raro**.

**Conclusão prática:** vale acrescentar ao meta-dataset X as versões `_raro`
das medidas de erro/suavidade (C4, L1, L2, L3, S1, S3, S4). As versões
`_normal` podem ser deixadas de fora sem perda relevante, e T2_raro deve sair.

## Ressalvas

- O grupo raro é pequeno em vários datasets (mediana de 97 exemplos, mínimo de
  10). As medidas ali têm variância alta; `n_raro` está no CSV para permitir
  filtrar ou ponderar.
- Os raros incluem extremos altos **e** baixos juntos, quando o φ é bimodal.
  Um grupo raro bimodal é, por construção, disperso — parte do S1/S3 alto pode
  vir daí. Separar raros-altos de raros-baixos é o próximo teste natural.
- A subamostragem em 1.500 afeta 6 datasets; os valores ali são estimativas.
- Nada disso foi ainda ligado a P (desempenho): o próximo passo é refazer o
  leave-one-dataset-out com X ampliado e ver se as meta-características `_raro`
  melhoram a predição de desempenho.
