"""Gera pratica4_automl_vs_mtl.ipynb (análises). Execute depois de rodar_tudo.sh.

    .venv/bin/python construir_notebook.py && .venv/bin/jupyter nbconvert --to notebook \
        --execute --inplace pratica4_automl_vs_mtl.ipynb
"""
import nbformat as nbf

C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip()))


md("""
# Prática 4 — AutoML versus meta-aprendizagem (regressão desbalanceada)

Mesmo problema, medida e protocolo das Práticas 1–3: 30 datasets de regressão
desbalanceada, **SERA** (menor é melhor) e as mesmas **dez partições** (KFold,
`shuffle`, semente 42). O AutoML recebe só o treino de cada fold.

As execuções pesadas ficam em scripts (`executar_baselines.py`,
`executar_warmstart.py`, `executar_automl.py`, encadeados por `rodar_tudo.sh`);
este notebook só lê os resultados brutos de `resultados/` e faz as análises.
""")

code(r"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display
from scipy.stats import friedmanchisquare, spearmanr, studentized_range
import scikit_posthocs as sp

import comum as c
sys.path.insert(0, str(c.BASE / 'codigo'))
from abordagens_meta_modelo import abordagem2_regressor_sobre_R
from gerar_X import extrair_meta_features

R = c.SAIDA
FIG = R / 'figuras'; FIG.mkdir(exist_ok=True)
DATASETS = c.datasets()
pd.set_option('display.width', 160)
print(json.dumps(c.ambiente(), indent=1))
""")

md("## 1. Resultados brutos por fold")
code(r"""
base = pd.read_csv(R / 'base_por_fold.csv')
desreg = pd.read_csv(R / 'desreg_por_fold.csv')
ws = pd.read_csv(R / 'warmstart_por_fold.csv')
ws['segundos'] = ws.segundos_busca + ws.segundos_final + ws.segundos_iniciais / c.FOLDS

linhas = []
for arq in sorted((R / 'automl').glob('*.jsonl')):
    linhas += [json.loads(l) for l in arq.read_text().splitlines()]
automl = pd.DataFrame(linhas)
falhas = automl[automl.erro.notna()]
automl = automl[automl.erro.isna()].drop_duplicates(['sistema', 'orcamento', 'dataset', 'fold'], keep='last')
automl['metodo'] = automl.sistema + '_' + automl.orcamento.astype(str) + 's'
print('falhas registradas (reexecutadas depois):', len(falhas))
display(automl.groupby(['sistema', 'orcamento']).agg(execucoes=('sera', 'size'),
        segundos_medios=('segundos', 'mean'), segundos_max=('segundos', 'max')).round(1))
assert (automl.groupby('metodo').size() == 300).all(), 'há execuções faltando'
""")

md("""
**Mesmo protocolo de fato.** A matriz P das práticas anteriores foi gerada por nós,
nas mesmas partições; ainda assim os seis regressores foram reexecutados aqui
(scikit-learn {versão atual}), junto com o AutoML, para que tempos e perdas venham
do mesmo ambiente. A conferência abaixo compara com a P original.
""")
code(r"""
P = base.pivot_table(index='dataset', columns='metodo', values='sera').loc[DATASETS, c.ALGORITMOS]
P_antiga = pd.read_csv(c.META / 'P.csv', index_col=0).loc[DATASETS, c.ALGORITMOS]
dif = (P / P_antiga - 1).abs()
print('diferença relativa mediana com a P da Prática 1:', f'{dif.stack().median():.2e}',
      '| máxima:', f'{dif.stack().max():.3f}')
print('o melhor algoritmo coincide em', int((P.idxmin(axis=1) == P_antiga.idxmin(axis=1)).sum()), 'de 30 datasets')
Rk = P.rank(axis=1)
""")

md("""
## 2. Os métodos comparados

1. **Melhor médio** — algoritmo de melhor rank médio nos outros 29 datasets, configuração padrão.
2. **MtL** — Abordagem 2 da Prática 1 (RF sobre R, com Xbase; Xestendido não trouxe ganho na Prática 2),
   *leave-one-dataset-out*, top-1, configuração padrão.
3. **Warm-start** — SMBO da Prática 3 (Xestendido, k=3, T=20) com avaliação interna no treino do fold.
4. **AutoML** — FLAML (com e sem meta-conhecimento) e AutoGluon, em 5, 30 e 120 s por fold.
5. **Oráculo** — melhor dos seis algoritmos de P no dataset (teto da seleção, não é método).

Extra, ligado ao tema da pesquisa: **DESReg** (seleção dinâmica sobre o pool de bagging
dos seis regressores), com competência padrão e com competência ponderada pela relevância (`phi`).
""")
code(r"""
Xbase = pd.read_csv(c.META / 'X.csv', index_col=0).loc[DATASETS]
rec, custo_rec = {}, {}
import time
for alvo in DATASETS:
    t0 = time.perf_counter()
    mf = extrair_meta_features(c.DADOS / f'{alvo}.csv')           # custo online: meta-características
    t_mf = time.perf_counter() - t0
    t0 = time.perf_counter()
    ranking = abordagem2_regressor_sobre_R(Xbase.drop(index=alvo), Rk.drop(index=alvo), Xbase.loc[[alvo]]).loc[alvo]
    t_mm = time.perf_counter() - t0                                  # treinar/consultar o meta-modelo (29 linhas)
    rec[alvo] = ranking.idxmin()
    custo_rec[alvo] = dict(meta_caracteristicas=t_mf, meta_modelo=t_mm)
rec = pd.Series(rec, name='MtL')
melhor_medio = pd.Series({d: Rk.drop(index=d).mean().idxmin() for d in DATASETS}, name='Melhor_medio')
display(pd.concat([melhor_medio, rec, P.idxmin(axis=1).rename('Oraculo')], axis=1).T)
""")

code(r"""
def por_fold(algs: pd.Series, nome):
    m = base.merge(algs.rename('alg').reset_index().rename(columns={'index': 'dataset'}), on='dataset')
    m = m[m.metodo == m.alg].assign(metodo=nome)
    return m[['dataset', 'fold', 'metodo', 'sera', 'segundos']]

oraculo = P.idxmin(axis=1)
folds = pd.concat([
    por_fold(melhor_medio, 'Melhor_medio'),
    por_fold(rec, 'MtL'),
    ws.assign(metodo='Warm-start')[['dataset', 'fold', 'metodo', 'sera', 'segundos']],
    desreg[['dataset', 'fold', 'metodo', 'sera', 'segundos']],
    automl[['dataset', 'fold', 'metodo', 'sera', 'segundos']],
    por_fold(oraculo, 'Oraculo'),
])
# custo online da recomendação: meta-características e meta-modelo, uma vez por dataset (rateado nos folds)
extra = pd.DataFrame(custo_rec).T.sum(axis=1) / c.FOLDS
folds.loc[folds.metodo == 'MtL', 'segundos'] += folds.loc[folds.metodo == 'MtL', 'dataset'].map(extra).to_numpy()
folds.to_csv(R / 'todos_por_fold.csv', index=False)

L = folds.pivot_table(index='dataset', columns='metodo', values='sera').loc[DATASETS]
TEMPO = folds.pivot_table(index='dataset', columns='metodo', values='segundos').loc[DATASETS]
METODOS_FIXOS = ['Melhor_medio', 'MtL', 'Warm-start', 'DESReg', 'DESReg_phi']
AUTOML = sorted(automl.metodo.unique(), key=lambda m: (m.split('_')[0], int(m.split('_')[-1][:-1]), m))
""")

md("## 3. Análise 1 — desempenho")
code(r"""
# Perda relativa ao melhor resultado observado no dataset (entre todos os métodos e o oráculo)
melhor = L.min(axis=1)
REL = L.div(melhor, axis=0) - 1
resumo = pd.DataFrame({'perda_rel_media': REL.mean(), 'perda_rel_mediana': REL.median(),
                       'rank_medio': L.drop(columns='Oraculo').rank(axis=1).mean(),
                       'vitorias': (L.drop(columns='Oraculo').rank(axis=1) == 1).sum(),
                       'segundos_por_fold': TEMPO.mean()}).sort_values('rank_medio')
display(resumo.round(3))
resumo.to_csv(R / 'resumo_metodos.csv')
L.to_csv(R / 'sera_por_dataset.csv'); REL.to_csv(R / 'perda_relativa_por_dataset.csv')
""")

code(r"""
ROT = {'Melhor_medio': 'Melhor médio', 'MtL': 'MtL', 'Warm-start': 'Warm-start',
       'DESReg': 'DESReg', 'DESReg_phi': 'DESReg-φ'}
def rotulo(m):
    if m in ROT: return ROT[m]
    s, o = m.rsplit('_', 1)
    return {'flaml': 'FLAML', 'flaml_meta': 'FLAML+zs', 'autogluon': 'AutoGluon'}[s] + f' {o}'

testes = []
for orc in [5, 30, 120]:
    cols = METODOS_FIXOS + [m for m in AUTOML if m.endswith(f'_{orc}s')]
    tab = L[cols]
    est, p = friedmanchisquare(*[tab[col] for col in cols])
    k, n = len(cols), len(tab)
    cd = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))
    pv = sp.posthoc_nemenyi_friedman(tab.to_numpy()); pv.index = pv.columns = cols
    ranks = tab.rank(axis=1).mean()
    testes.append(dict(orcamento=orc, metodos=k, friedman=est, p=p, cd=cd))
    plt.figure(figsize=(8, 2.4))
    sp.critical_difference_diagram(ranks.rename(rotulo), pv.rename(index=rotulo, columns=rotulo),
                                   label_props={'fontsize': 9})
    ax = plt.gca(); ini = ranks.min()
    ax.plot([ini, ini + cd], [1.6, 1.6], color='black', lw=2)
    ax.text(ini + cd / 2, 2.0, f'CD = {cd:.2f}', ha='center', fontsize=9); ax.set_ylim(top=3.0)
    plt.title(f'AutoML com {orc} s por fold — Friedman p = {p:.2g}', fontsize=10, pad=18)
    plt.tight_layout(); plt.savefig(FIG / f'cd_{orc}s.pdf', bbox_inches='tight'); plt.show()
    pv.to_csv(R / f'nemenyi_{orc}s.csv')
    print(f'{orc}s: p(MtL vs AutoML):', {m: round(pv.loc['MtL', m], 4) for m in cols if m not in METODOS_FIXOS})
testes = pd.DataFrame(testes); testes.to_csv(R / 'friedman.csv', index=False); display(testes.round(4))
""")

code(r"""
# Dataset a dataset: perda relativa de cada método (heatmap)
ordem = resumo.index.tolist()
fig, ax = plt.subplots(figsize=(11, 8))
im = ax.imshow(np.log10(1 + REL[ordem].to_numpy()), aspect='auto', cmap='viridis')
ax.set_xticks(range(len(ordem)), [rotulo(m) if m != 'Oraculo' else 'Oráculo' for m in ordem], rotation=60, ha='right', fontsize=8)
ax.set_yticks(range(len(DATASETS)), DATASETS, fontsize=8)
plt.colorbar(im, label='log10(1 + perda relativa)'); plt.tight_layout()
plt.savefig(FIG / 'perda_por_dataset.pdf', bbox_inches='tight'); plt.show()
""")

md("## 4. Análise 2 — desempenho e custo")
code(r"""
# Custo offline (segundos de CPU somados), do conjunto das práticas:
#  * MtL: construir P (6 regressores × 10 folds × 30 datasets, medido agora) e Xbase (30 extrações)
#  * Warm-start: matriz de perdas 60 × 30 da Prática 3 e extração de Xestendido (Prática 2)
custo_ext = pd.read_csv(c.BASE / 'pratica-02' / 'resultados' / 'custo_extracao.csv')
custo_ext = custo_ext[custo_ext.fase != 'nulo']           # distribuições nulas são diagnóstico, não descritor
t_xest = custo_ext.groupby('dataset').segundos.sum().reindex(DATASETS)
t_xbase = pd.Series({d: v['meta_caracteristicas'] for d, v in custo_rec.items()})
tempo_p3 = json.loads((c.P3 / 'tempo_matriz.json').read_text())
offline = {'MtL': base.segundos.sum() + t_xbase.sum(),
           'Warm-start': tempo_p3['segundos_ajustes'] + t_xest.sum()}
print({k: f'{v/3600:.2f} h de CPU' for k, v in offline.items()})

# Warm-start: o custo online inclui extrair Xestendido do alvo (rateado nos folds)
TEMPO['Warm-start'] = TEMPO['Warm-start'] + t_xest / c.FOLDS

custo = pd.DataFrame({'segundos_por_fold': TEMPO.drop(columns='Oraculo').mean(),
                      'perda_rel_media': REL.drop(columns='Oraculo').mean(),
                      'perda_rel_mediana': REL.drop(columns='Oraculo').median(),
                      'rank_medio': L.drop(columns='Oraculo').rank(axis=1).mean()})
display(custo.sort_values('segundos_por_fold').round(3))
custo.to_csv(R / 'custo_desempenho.csv')

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for eixo, y, nome in [(ax[0], 'rank_medio', 'Rank médio (menor é melhor)'), (ax[1], 'perda_rel_mediana', 'Perda relativa mediana (menor é melhor)')]:
    for sis, cor in [('flaml_meta', 'C0'), ('flaml', 'C9'), ('autogluon', 'C1')]:
        sub = custo.loc[[m for m in AUTOML if m.rsplit('_', 1)[0] == sis]].sort_values('segundos_por_fold')
        eixo.plot(sub.segundos_por_fold, sub[y], 'o-', color=cor, label=rotulo(sub.index[0]).split()[0])
    for m, mk in zip(METODOS_FIXOS, ['s', 'D', '^', 'v', 'P']):
        eixo.scatter(custo.loc[m, 'segundos_por_fold'], custo.loc[m, y], marker=mk, s=60, color='k')
        eixo.annotate(rotulo(m), (custo.loc[m, 'segundos_por_fold'], custo.loc[m, y]), xytext=(4, -11 if m == 'Melhor_medio' else 4), textcoords='offset points', fontsize=8)
    eixo.set_xscale('log'); eixo.set_xlabel('Custo online por fold (s, escala log)'); eixo.set_ylabel(nome)
ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(FIG / 'custo_desempenho.pdf', bbox_inches='tight'); plt.show()
""")

code(r"""
# Com que orçamento o AutoML supera a MtL? Por dataset: menor orçamento em que a SERA média do AutoML < a da MtL.
supera = {}
for sis in ['flaml_meta', 'flaml', 'autogluon']:
    cols = [m for m in AUTOML if m.rsplit('_', 1)[0] == sis]
    for alvo in ['MtL', 'Warm-start']:
        primeiro = L[cols].lt(L[alvo], axis=0).idxmax(axis=1).where(L[cols].lt(L[alvo], axis=0).any(axis=1), 'nunca')
        supera[(sis, alvo)] = primeiro.value_counts().reindex(cols + ['nunca'], fill_value=0)
display(pd.DataFrame(supera).fillna(0).astype(int))

# Ponto de equilíbrio do custo offline: menor n de datasets novos com offline + online_sol × n ≤ online_AutoML × n.
# A comparação justa é com o AutoML de desempenho equivalente (rank médio parecido); contra AutoML
# melhores, a economia é troca de qualidade por custo, e não ganho líquido.
for sol in ['MtL', 'Warm-start']:
    for m in AUTOML:
        dif = (TEMPO[m].mean() - TEMPO[sol].mean()) * c.FOLDS        # por dataset (10 folds)
        if dif > 0:
            rel = 'melhor' if custo.loc[m, 'rank_medio'] < custo.loc[sol, 'rank_medio'] else 'pior'
            print(f'{sol} vs {rotulo(m)} (AutoML {rel} em rank): economia {dif:.0f} s/dataset, '
                  f'offline se paga após {int(np.ceil(offline[sol] / dif))} datasets novos')
""")

md("## 5. Análise 3 — o que o AutoML escolheu")
code(r"""
FAMILIA = {'rf': 'RF', 'RandomForestMSE': 'RF', 'RandomForest': 'RF', 'lgbm': 'GB', 'xgboost': 'GB', 'xgb_limitdepth': 'GB',
           'catboost': 'GB', 'LightGBM': 'GB', 'LightGBMXT': 'GB', 'LightGBMLarge': 'GB', 'CatBoost': 'GB', 'XGBoost': 'GB',
           'extra_tree': 'ExtraTrees', 'ExtraTreesMSE': 'ExtraTrees', 'sgd': 'Linear', 'LinearModel': 'Linear',
           'KNeighborsUnif': 'KNN', 'KNeighborsDist': 'KNN', 'NeuralNetFastAI': 'Rede neural', 'NeuralNetTorch': 'Rede neural'}
def base_ag(m): return m.split('_BAG')[0].replace('_L1', '')

def familias(r):
    if r.sistema == 'autogluon' and isinstance(r.pesos_ensemble, dict) and r.pesos_ensemble:
        return sorted({FAMILIA.get(base_ag(k), k) for k in r.pesos_ensemble})
    return [FAMILIA.get(base_ag(r.modelo), r.modelo)]
automl['familias'] = automl.apply(familias, axis=1)
def modelo_base(r):
    # AutoGluon: um WeightedEnsemble com um único peso é, na prática, aquele modelo
    if r.sistema != 'autogluon':
        return base_ag(r.modelo)
    if r.ensemble:
        return 'WeightedEnsemble'
    pesos = r.pesos_ensemble if isinstance(r.pesos_ensemble, dict) and r.pesos_ensemble else {r.modelo: 1}
    return base_ag(next(iter(pesos)))
automl['modelo_base'] = automl.apply(modelo_base, axis=1)

display(automl.groupby('metodo').modelo_base.value_counts(normalize=True).unstack(fill_value=0).round(2))
ag = automl[automl.sistema == 'autogluon']
print('AutoGluon: fração de folds com ensemble de >1 modelo:', ag.ensemble.astype(float).groupby(ag.orcamento).mean().round(2).to_dict())
print('AutoGluon: modelos por ensemble (média):', ag.groupby('orcamento').pesos_ensemble.apply(lambda s: np.mean([len(p) if isinstance(p, dict) else 1 for p in s])).round(2).to_dict())

exp = automl.explode('familias')
# fração de folds em que cada família aparece no modelo final (um ensemble pode ter várias)
freq = pd.crosstab([exp.metodo, exp.dataset, exp.fold], exp.familias).clip(upper=1).groupby('metodo').mean()
print('fração de folds em que a família aparece no modelo final:'); display(freq.round(2))
fora_de_P = automl.familias.apply(lambda f: bool(set(f) - set(c.ALGORITMOS))).groupby(automl.metodo).mean()
print('fração de folds com alguma família fora dos 6 algoritmos de P:'); display(fora_de_P.round(2))

concorda = exp.assign(rec=exp.dataset.map(rec)).assign(igual=lambda d: d.familias == d.rec).groupby(['metodo', 'dataset', 'fold']).igual.any().groupby('metodo').mean()
print('fração de folds em que a família recomendada pela MtL aparece no modelo final do AutoML:'); display(concorda.round(2))
""")

code(r"""
# Pré-processamento: o FLAML não faz engenharia de atributos para dados numéricos; o AutoGluon aplica
# seu gerador de atributos automático (todos os 30 datasets são numéricos, então ele só tipa/remove constantes).
# Onde o AutoML mais ganhou da MtL, e o que já sabíamos desses datasets?
ref = 'flaml_meta_120s' if 'flaml_meta_120s' in L else AUTOML[-1]
# AutoGluon a 120 s sozinho: escolher o melhor AutoML por dataset olharia o teste
ganho = (L['MtL'] - L['autogluon_120s']) / L['MtL']
dist = pd.read_csv(c.P3 / 'distancias.csv').query("meta == 'estendido'").set_index('dataset').distancia
p1 = pd.read_csv(c.BASE / 'pratica-01' / 'resultados' / 'avaliacao_por_dataset.csv').query("abordagem == 'Abordagem_2'").set_index('dataset')
tab = pd.DataFrame({'ganho_rel_AutoGluon_120s': ganho,
                    'MtL_acertou_oraculo': rec == oraculo,
                    'perda_meta_modelo_P1': p1.perda_t1,
                    'distancia_vizinho_P3': dist,
                    'pct_relevante': Xbase.pct_relevante, 'n': Xbase.n_instances, 'p': Xbase.n_features,
                    'oraculo_vs_MtL': L['MtL'] / L['Oraculo'] - 1,
                    'automl_vs_oraculo': L['autogluon_120s'] / L['Oraculo'] - 1}).loc[DATASETS]
display(tab.sort_values('ganho_rel_AutoGluon_120s', ascending=False).round(3))
for col in ['perda_meta_modelo_P1', 'distancia_vizinho_P3', 'pct_relevante', 'n', 'p']:
    rho, p = spearmanr(tab.ganho_rel_AutoGluon_120s, tab[col], nan_policy='omit')
    print(f'Spearman ganho × {col}: rho={rho:.3f}, p={p:.3g}')
print('Datasets em que o AutoGluon 120 s supera o ORÁCULO dos 6 algoritmos:', int((tab.automl_vs_oraculo < 0).sum()), 'de 30',
      '| ganho mediano sobre o oráculo:', round(float(-tab.automl_vs_oraculo.median()), 3))
tab.to_csv(R / 'ganho_automl_por_dataset.csv')
""")

md("## 6. Meta-conhecimento dentro do AutoML e sobreposição com os nossos datasets")
code(r"""
# FLAML com e sem o portfólio zero-shot, no mesmo orçamento
for orc in [5, 30, 120]:
    a, b = f'flaml_meta_{orc}s', f'flaml_{orc}s'
    if a in L and b in L:
        d = (L[b] - L[a]) / L[b]
        print(f'{orc}s: zero-shot melhor em {(d > 0).sum()}/30 datasets; ganho relativo mediano {d.median():+.3f}')

# Quantos estimadores receberam configuração zero-shot efetiva (diferente do ponto estático)?
fm = automl[automl.sistema == 'flaml_meta'].copy()
estaticos = automl[automl.sistema == 'flaml'].set_index(['dataset', 'fold', 'orcamento']).iniciais
def efetivos(r):
    est = estaticos.get((r.dataset, r.fold, r.orcamento))
    if not isinstance(est, dict): return np.nan
    return sum(json.dumps(r.iniciais.get(e), sort_keys=True, default=str) != json.dumps(est.get(e), sort_keys=True, default=str) for e in r.iniciais)
fm['zero_shot_efetivos'] = fm.apply(efetivos, axis=1)
print('estimadores com ponto inicial zero-shot efetivo (média por fold):')
display(fm.groupby('dataset').zero_shot_efetivos.mean().sort_values(ascending=False).head(10))
print('primeiro estimador da busca (sugerido pelo zero-shot):', fm.ordem_estimadores.str[0].value_counts().to_dict())
""")

code(r"""
# O portfólio do FLAML foi minerado em datasets do OpenML/AMLB. Os nomes das configurações e os
# 22 vizinhos (meta-características em escala original) permitem checar a sobreposição.
import flaml, os
port = json.load(open(Path(flaml.__file__).parent / 'default' / 'all' / 'regression.json'))
centro, escala = np.array(port['preprocessing']['center']), np.array(port['preprocessing']['scale'])
viz = pd.DataFrame([np.array(v['features']) * escala + centro for v in port['neighbors']], columns=port['meta_feature_names'])
viz[['NumberOfInstances', 'NumberOfFeatures']] = viz[['NumberOfInstances', 'NumberOfFeatures']].round().astype(int)
print('configurações do portfólio (estimador/dataset de origem):', port['configsource'])
# As meta-características do portfólio foram medidas no treino (90% do dataset, como nos nossos folds):
# comparamos com 0,9 × n e também com n.
nossos = Xbase[['n_instances', 'n_features']]
casam = []
for d, r in nossos.iterrows():
    for fator in (1.0, 0.9):
        perto = (viz.NumberOfInstances - fator * r.n_instances).abs() / (fator * r.n_instances) < 0.005
        if (perto & (viz.NumberOfFeatures == r.n_features)).any():
            casam.append((d, int(r.n_instances), int(r.n_features), fator))
print('datasets nossos com ponto do portfólio de mesmo n (±0,5%, n ou 0,9n) e mesmo p:', casam)
display(viz.sort_values('NumberOfInstances'))
""")

nb = nbf.v4.new_notebook(cells=C, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, "pratica4_automl_vs_mtl.ipynb")
print("notebook gerado:", len(C), "células")
