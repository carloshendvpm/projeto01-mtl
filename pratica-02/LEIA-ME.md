# Prática 2: meta-características para regressão desbalanceada

- [Relatório em PDF](relatorio.pdf) e [fonte LaTeX](relatorio.tex).
- [Notebook](atv-02-mtl.ipynb), também disponível no [Google Colab](https://colab.research.google.com/drive/1rmAUTeKFGBM2pG6Rl0QFPk2mKrp471cN?usp=sharing).
- `avaliar_colab.py` e `comparacao_estatistica_colab.py`: avaliação da recomendação e comparação estatística, usadas no Colab.
- `resultados/`: medidas de complexidade, custos de extração e diagnósticos.
- `figuras/`: curvas de perda e diagrama de diferença crítica.

## Reproduzir

No repositório, o notebook lê os dados de `../data/`, as matrizes de `../meta-dataset/` e os módulos de `../codigo/`. Para rodar no Colab, coloque em `entradas/` os arquivos listados na primeira célula.

Para recompilar o relatório, rode duas vezes dentro desta pasta:

```bash
pdflatex -interaction=nonstopmode -halt-on-error relatorio.tex
```

## Resultado

Estendi Xbase com medidas de complexidade calculadas no alvo inteiro e, separadamente, nos exemplos comuns e nos raros. As médias das abordagens com Random Forest melhoraram: na Abordagem 2, a área da curva de perda caiu de 0,032 para 0,023. O teste de Friedman, porém, não rejeitou a igualdade em nenhuma abordagem (p = 0,1105; 0,2352; 0,4514, com α = 0,05/3), e o Nemenyi não apontou nenhum par significativo (CD = 1,842).

Os tempos citados no relatório são os registrados no Colab. A avaliação com seleção de atributos foi refeita depois de corrigir a ordem das colunas no seletor, e só a versão corrigida entra nas conclusões.
