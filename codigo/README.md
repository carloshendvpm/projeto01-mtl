# Código das Práticas 1 e 2

Scripts que geram as matrizes e as análises das Práticas 1 e 2. As Práticas 3 e 4 têm o código nas próprias pastas; a Prática 4 também importa `abordagens_meta_modelo.py` e `gerar_X.py` daqui.

| Etapa | Scripts |
|---|---|
| Construir Xbase | `gerar_X.py` |
| Avaliar os regressores | `gerar_matriz_P.py`, `gerar_matriz_P_resampling.py` |
| Construir P e R a partir dos folds | `gerar_P_e_R.py`, `gerar_P_e_R_resampling.py` |
| Recomendar algoritmos | `abordagens_agregacao.py`, `abordagens_meta_modelo.py`, `harris.py` |
| Avaliar recomendações | `avaliacao_leave_one_out.py`, `comparar_resampling.py` |
| Extrair complexidade por grupo | `gerar_complexidade.py` |
| Montar as variantes de X | `montar_x.py` |
| Analisar complexidade e descritores | `analisar_complexidade.py`, `analise1_meta_alvo.py`, `analise2_redundancia.py` |
| Avaliar Xestendido | `avaliacao_x_estendido.py`, `comparar_variantes.py`, `selecao_meta_features.py` |
| Gerar figuras e tabelas da Prática 1 | `gerar_resultados_relatorio.py`, `diagrama_diferenca_critica.py` |

Os caminhos são relativos à raiz do projeto, então os scripts podem ser chamados de qualquer pasta:

```bash
python3 codigo/montar_x.py
```

Os scripts que geram matrizes treinam muitos modelos e podem levar horas.
