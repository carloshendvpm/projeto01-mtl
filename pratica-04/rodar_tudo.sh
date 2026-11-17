#!/bin/zsh
# Encadeia as execuções da Prática 4 (sequenciais, para que os tempos não se contaminem).
# Uso: ./rodar_tudo.sh  (a partir de pratica-04/). Pode ser interrompido e retomado.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
[[ -f resultados/warmstart_por_fold.csv ]] || $PY executar_warmstart.py --jobs 6
for ORC in 5 30 120; do $PY executar_automl.py --sistema flaml_meta --orcamento $ORC --workers 4 --ncpus 2; done
for ORC in 5 30; do $PY executar_automl.py --sistema flaml --orcamento $ORC --workers 4 --ncpus 2; done
for ORC in 5 30 120; do $PY executar_automl.py --sistema autogluon --orcamento $ORC --workers 4 --ncpus 2; done
echo FIM
