"""Medidas de complexidade para regressao (Lorena et al., 2018).

Cada medida e calculada sobre um conjunto (Z, y) JA normalizado em [0,1]. O
modulo nao sabe nada sobre "raro" ou "normal": quem separa os grupos e o
gerar_complexidade.py. A normalizacao acontece uma vez, no dataset inteiro,
antes de dividir os grupos -- normalizar cada grupo a parte poria os tres em
escalas diferentes e a comparacao perderia sentido (ficha 4.7).

Tres pontos em que este modulo difere da definicao literal do artigo, todos
deliberados e documentados na medida correspondente:

1. Squashing v/(1+v) no nivel do ELEMENTO (cada aresta, cada erro, cada
   residuo), como a revisao de maio/2026 do ECoL, e so entao a agregacao.
   Cada medida agregativa devolve dois numeros do mesmo calculo: o bruto,
   comparavel com Lorena/Meta-IR, e o `_sq`, que NAO e comparavel com numeros
   publicados e existe para por todas as medidas na mesma escala [0,1).
2. L1/L2/L3 canonicos usam validacao cruzada dentro do conjunto. A definicao
   literal (residuo in-sample) fica como variante `_treino`: com p >= n o
   ajuste linear e exato e o residuo e zero por construcao, o que tornava a
   medida um artefato do tamanho do grupo, nao uma medida de linearidade.
3. Medidas que dependem de vizinhanca ou de ajuste de modelo tem variante
   `_vizglobal`: a MST, o vizinho ou o modelo vem do dataset inteiro e so a
   agregacao e restrita ao grupo. E a leitura da ficha 4.6 -- "a dificuldade
   da regiao dentro do problema como ele e" -- e a unica imune ao
   confundimento de tamanho, porque nao reconstroi a geometria num
   subconjunto esparso.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, LeaveOneOut

warnings.filterwarnings("ignore", category=ConstantInputWarning)

RNG_SEED = 42

# Abaixo de MIN_CALCULAVEL nao se calcula nada. Entre MIN_CALCULAVEL e
# MIN_CONFIAVEL calcula-se, mas a linha sai marcada com confiabilidade_baixa.
MIN_CALCULAVEL = 10
MIN_CONFIAVEL = 20

# Margem exigida para que a validacao cruzada de L1/L2/L3 seja definida:
# cada fold de treino precisa de pelo menos p + MARGEM_CV exemplos.
MARGEM_CV = 10

MOTIVOS = frozenset(
    {
        "grupo_vazio",
        "grupo_pequeno",
        "p_maior_que_n",
        "alvo_constante",
        "sem_atributo_valido",
        "aresta_unica",
        "identico_ao_primario",
    }
)


class MedidaIndefinida(Exception):
    """A medida nao esta definida para este conjunto, e sabemos por que.

    Qualquer outra excecao propaga e aborta a execucao: um NaN por bug seria
    indistinguivel de um NaN por indefinicao legitima, que foi exatamente o
    defeito do `except Exception: np.nan` da primeira tentativa.
    """

    def __init__(self, motivo: str):
        if motivo not in MOTIVOS:
            raise ValueError(f"motivo fora do vocabulario: {motivo!r}")
        self.motivo = motivo
        super().__init__(motivo)


def minmax(A: np.ndarray) -> np.ndarray:
    """Min-max por coluna. Amplitude zero vira coluna de zeros."""
    A = np.asarray(A, dtype=float)
    lo = A.min(axis=0)
    amp = A.max(axis=0) - lo
    amp = np.where(amp == 0, 1.0, amp)
    return (A - lo) / amp


def squash(v: np.ndarray | float) -> np.ndarray | float:
    """v / (1 + v), aplicada elemento a elemento antes de agregar."""
    return v / (1.0 + v)


# ------------------------------------------------------------------ contexto


@dataclass
class ContextoGrupo:
    """Um conjunto (Z, y) normalizado, com os calculos caros memoizados.

    S1, S2, S3 e S4 precisavam de `pdist` e C1..C4 de `_rho`; na primeira
    tentativa cada um recalculava o seu, quatro e cinco vezes por grupo. Com o
    contexto sao uma `pdist` e um `rho` por grupo, o que e pre-requisito para
    o custo das distribuicoes nulas (200 recalculos por dataset e medida).
    """

    Z: np.ndarray
    y: np.ndarray
    rng_seed: int = RNG_SEED
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.Z = np.asarray(self.Z, dtype=float)
        self.y = np.asarray(self.y, dtype=float).ravel()
        if self.Z.ndim != 2:
            raise ValueError("Z precisa ser 2-D")
        if len(self.y) != self.Z.shape[0]:
            raise ValueError("Z e y com numeros de linhas diferentes")

    @property
    def n(self) -> int:
        return self.Z.shape[0]

    @property
    def p(self) -> int:
        return self.Z.shape[1]

    def _memo(self, chave, fn):
        if chave not in self._cache:
            self._cache[chave] = fn()
        return self._cache[chave]

    @property
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.rng_seed)

    @property
    def D(self) -> np.ndarray:
        """Matriz de distancias euclidianas, densa. O(n^2) em memoria."""
        return self._memo("D", lambda: squareform(pdist(self.Z)))

    @property
    def rho(self) -> np.ndarray:
        """|rho de Spearman| de cada atributo com o alvo; NaN se indefinido.

        Atributo constante devolve NaN, nao zero: a correlacao ali e
        indefinida, e trata-la como zero seria afirmar ausencia de associacao
        (ficha 4.2).
        """
        return self._memo("rho", self._calcular_rho)

    def _calcular_rho(self) -> np.ndarray:
        out = np.full(self.p, np.nan)
        if self.n == 0 or np.all(self.y == self.y[0]):
            return out
        for j in range(self.p):
            coluna = self.Z[:, j]
            if np.all(coluna == coluna[0]):
                continue
            out[j] = abs(spearmanr(coluna, self.y).statistic)
        return out

    @property
    def n_atributos_validos(self) -> int:
        return int(np.count_nonzero(~np.isnan(self.rho)))

    @property
    def arestas_mst(self) -> np.ndarray:
        """|y_i - y_j| em cada aresta unica da MST construida sobre Z.

        `.tocoo()` de uma MST devolve exatamente n-1 arestas, uma orientacao
        por aresta -- a implementacao do ECoL coleta as duas orientacoes da
        matriz simetrica, o que preserva a media mas altera o desvio amostral.
        Aqui o desvio usa arestas unicas e divisor m-1 (nota 3 da secao 6).
        """
        return self._memo("mst", self._calcular_mst)

    def _calcular_mst(self) -> np.ndarray:
        mst = minimum_spanning_tree(self.D).tocoo()
        return np.abs(self.y[mst.row] - self.y[mst.col])

    @property
    def ordem_y(self) -> np.ndarray:
        """Ordenacao por y, ESTAVEL.

        S2, L3 e S4 dependem desta ordem. Com empates em y -- comuns quando o
        alvo e discretizado -- o quicksort padrao do numpy escolhe uma ordem
        arbitraria e dependente de plataforma, e as medidas deixam de ser
        reprodutiveis. A ordenacao estavel resolve os empates pelo indice.
        """
        return self._memo("ordem_y", lambda: np.argsort(self.y, kind="stable"))

    @property
    def erros_1nn(self) -> np.ndarray:
        """Erro quadratico de 1-NN leave-one-out para CADA exemplo.

        O vizinho e buscado dentro do conjunto passado aqui. Empates de
        distancia resolvem-se pelo menor indice (convencao da ficha 4.5), que
        e o que `argmin` ja faz.
        """
        return self._memo("erros_1nn", self._calcular_erros_1nn)

    def _calcular_erros_1nn(self) -> np.ndarray:
        D = self.D.copy()
        np.fill_diagonal(D, np.inf)
        return (self.y - self.y[np.argmin(D, axis=1)]) ** 2

    @property
    def interpolados(self) -> tuple[np.ndarray, np.ndarray]:
        """Pontos sinteticos entre pares consecutivos na ordenacao de y."""
        return self._memo("interp", self._calcular_interpolados)

    def _calcular_interpolados(self):
        Zs, ys = self.Z[self.ordem_y], self.y[self.ordem_y]
        a, b = Zs[:-1], Zs[1:]
        ya, yb = ys[:-1], ys[1:]
        alfa = self.rng.random((len(a), 1))
        return a + alfa * (b - a), ya + alfa[:, 0] * (yb - ya)

    def exigir_minimo(self, minimo: int = MIN_CALCULAVEL) -> None:
        if self.n == 0:
            raise MedidaIndefinida("grupo_vazio")
        if self.n < minimo:
            raise MedidaIndefinida("grupo_pequeno")

    def exigir_atributo_valido(self) -> None:
        if self.n_atributos_validos == 0:
            raise MedidaIndefinida("sem_atributo_valido")

    def exigir_alvo_variavel(self) -> None:
        if np.all(self.y == self.y[0]):
            raise MedidaIndefinida("alvo_constante")


def predicao_1nn(Z_treino, y_treino, Z_teste) -> np.ndarray:
    D = np.linalg.norm(Z_teste[:, None, :] - Z_treino[None, :, :], axis=2)
    return y_treino[np.argmin(D, axis=1)]


# --------------------------------------------------------------- correlacao
# Nao agregam elemento a elemento e ja estao em [0,1]: sem variante `_sq`.


def C1(ctx: ContextoGrupo) -> float:
    """Correlacao maxima atributo-alvo. Quanto maior, mais simples."""
    ctx.exigir_minimo()
    ctx.exigir_atributo_valido()
    return float(np.nanmax(ctx.rho))


def C2(ctx: ContextoGrupo) -> float:
    """Correlacao media atributo-alvo (C2.mean do levantamento).

    A media percorre um numero variavel de atributos validos entre datasets
    (de 5 a 282 na primeira extracao), logo sua precisao varia por linha;
    `n_atributos_validos` sai no CSV junto para permitir a ressalva.
    """
    ctx.exigir_minimo()
    ctx.exigir_atributo_valido()
    return float(np.nanmean(ctx.rho))


def C3(ctx: ContextoGrupo) -> float:
    """Fracao minima de exemplos a remover para atingir |rho| >= 0.9.

    Divide pelo n do grupo, logo e incomparavel entre grupos de tamanhos muito
    diferentes -- a mesma ressalva que tirou T2 de X vale aqui, e esta
    registrada no relatorio.
    """
    ctx.exigir_minimo()
    ctx.exigir_atributo_valido()
    n = ctx.n
    minimo = min(_remover_ate_correlacionar(ctx.Z[:, j], ctx.y, n) for j in range(ctx.p))
    return float(minimo / n)


def _remover_ate_correlacionar(x, y, n, alvo=0.9) -> int:
    """Quantos exemplos remover para |rho(x, y)| >= alvo.

    Remove iterativamente o exemplo de maior discrepancia entre o rank de x e
    o rank de y, como no algoritmo suplementar do artigo. Satura em n-3.
    """
    sinal = 1.0
    r = spearmanr(x, y).statistic
    if not np.isnan(r) and r < 0:
        sinal = -1.0
    ativos = np.ones(n, dtype=bool)
    for removidos in range(n - 3):
        xi, yi = x[ativos], y[ativos]
        r = spearmanr(xi, yi).statistic
        if not np.isnan(r) and abs(r) >= alvo:
            return removidos
        d = np.abs(np.argsort(np.argsort(sinal * xi)) - np.argsort(np.argsort(yi)))
        idx = np.flatnonzero(ativos)[int(np.argmax(d))]
        ativos[idx] = False
    return n - 3


def C4(ctx: ContextoGrupo, limiar: float = 0.1) -> float:
    """Eficiencia coletiva: fracao de exemplos nao explicados.

    Toma o atributo mais correlacionado, remove os exemplos de residuo pequeno
    no ajuste linear simples, repete com o atributo seguinte.
    """
    ctx.exigir_minimo()
    ctx.exigir_atributo_valido()
    Z, y = ctx.Z, ctx.y
    ativos = np.ones(ctx.n, dtype=bool)
    restantes = list(range(ctx.p))
    while restantes and ativos.sum() > 2:
        sub = ContextoGrupo(Z[np.ix_(ativos, restantes)], y[ativos])
        rho = sub.rho
        if np.all(np.isnan(rho)):
            break
        j = restantes.pop(int(np.nanargmax(rho)))
        xi = Z[ativos, j].reshape(-1, 1)
        yi = y[ativos]
        res = np.abs(yi - LinearRegression().fit(xi, yi).predict(xi))
        manter = res > limiar
        if manter.all():
            continue
        idx = np.flatnonzero(ativos)
        ativos[idx[~manter]] = False
    return float(ativos.sum() / ctx.n)


# -------------------------------------------------------------- linearidade


def residuos_treino(ctx: ContextoGrupo) -> np.ndarray:
    """|residuo| in-sample: a definicao literal de Lorena et al.

    Diagnostico de fidelidade, nao canonico. Com p >= n o ajuste e exato e
    estes residuos sao ~0 por construcao.
    """
    ctx.exigir_minimo()
    ctx.exigir_alvo_variavel()
    pred = LinearRegression().fit(ctx.Z, ctx.y).predict(ctx.Z)
    return np.abs(ctx.y - pred)


def residuos_cv(ctx: ContextoGrupo) -> np.ndarray:
    """|residuo| fora da amostra, por validacao cruzada dentro do grupo.

    K = 5 se n >= 50, senao leave-one-out. Indefinida se cada fold de treino
    nao tiver p + MARGEM_CV exemplos: com p proximo de n o ajuste volta a ser
    exato e a medida deixa de medir dificuldade. Correcao por graus de
    liberdade nao serve de alternativa -- com p >= n o denominador n-p-1 fica
    <= 0 e nao esta definido.
    """
    ctx.exigir_minimo()
    ctx.exigir_alvo_variavel()
    n, p = ctx.n, ctx.n_atributos_validos
    cv = KFold(n_splits=5, shuffle=True, random_state=ctx.rng_seed) if n >= 50 else LeaveOneOut()
    n_folds = cv.get_n_splits(ctx.Z)
    n_treino_minimo = n - int(np.ceil(n / n_folds))
    if n_treino_minimo < p + MARGEM_CV:
        raise MedidaIndefinida("p_maior_que_n")
    residuos = np.empty(n)
    for treino, teste in cv.split(ctx.Z):
        modelo = LinearRegression().fit(ctx.Z[treino], ctx.y[treino])
        residuos[teste] = np.abs(ctx.y[teste] - modelo.predict(ctx.Z[teste]))
    return residuos


def residuos_vizglobal(ctx_completo: ContextoGrupo, mascara: np.ndarray) -> np.ndarray:
    """|residuo| do modelo ajustado no dataset INTEIRO, restrito ao grupo.

    Onde n >> p quase sempre vale, logo e o fallback informativo quando a
    validacao cruzada dentro do grupo nao cabe.
    """
    ctx_completo.exigir_minimo()
    ctx_completo.exigir_alvo_variavel()
    pred = LinearRegression().fit(ctx_completo.Z, ctx_completo.y).predict(ctx_completo.Z)
    return np.abs(ctx_completo.y - pred)[mascara]


def erros_interpolados_lineares(ctx: ContextoGrupo) -> np.ndarray:
    """Erro quadratico do modelo linear do grupo nos pontos interpolados (L3)."""
    ctx.exigir_minimo()
    ctx.exigir_alvo_variavel()
    Zi, yi = ctx.interpolados
    modelo = LinearRegression().fit(ctx.Z, ctx.y)
    return (yi - modelo.predict(Zi)) ** 2


def erros_interpolados_lineares_vizglobal(ctx_completo, mascara) -> np.ndarray:
    """L3 com interpolacoes do grupo e modelo linear do dataset inteiro."""
    ctx_completo.exigir_minimo()
    ctx_completo.exigir_alvo_variavel()
    ctx_grupo = ContextoGrupo(ctx_completo.Z[mascara], ctx_completo.y[mascara])
    ctx_grupo.exigir_minimo()
    Zi, yi = ctx_grupo.interpolados
    modelo = LinearRegression().fit(ctx_completo.Z, ctx_completo.y)
    return (yi - modelo.predict(Zi)) ** 2


# ----------------------------------------------------------------- suavidade


def arestas_mst(ctx: ContextoGrupo) -> np.ndarray:
    """|delta y| nas arestas da MST construida DENTRO do grupo (S1, S1_sd).

    Atencao ao confundimento de tamanho: uma MST sobre 10% dos pontos tem
    arestas mais longas por geometria, e |delta y| maior, independentemente de
    dificuldade. E por isso que a comparacao entre grupos exige a distribuicao
    nula de `lib/nulo_tamanho.py`, e que existe a variante `_vizglobal`.
    """
    ctx.exigir_minimo()
    arestas = ctx.arestas_mst
    if len(arestas) < 2:
        raise MedidaIndefinida("aresta_unica")
    return arestas


def arestas_mst_vizglobal(ctx_completo: ContextoGrupo, mascara: np.ndarray) -> np.ndarray:
    """Arestas da MST do dataset inteiro com AMBOS os extremos no grupo.

    Nao reconstroi geometria em subconjunto esparso, logo nao sofre o
    confundimento de tamanho. Em troca, o numero de arestas internas ao grupo
    nao e controlado: um grupo disperso pode ter pouquissimas.
    """
    ctx_completo.exigir_minimo()
    mst = minimum_spanning_tree(ctx_completo.D).tocoo()
    interna = mascara[mst.row] & mascara[mst.col]
    if np.count_nonzero(interna) < 2:
        raise MedidaIndefinida("aresta_unica")
    y = ctx_completo.y
    return np.abs(y[mst.row[interna]] - y[mst.col[interna]])


def distancias_vizinhos_em_y(ctx: ContextoGrupo) -> np.ndarray:
    """Distancia em Z entre exemplos consecutivos na ordenacao de y (S2)."""
    ctx.exigir_minimo()
    Zs = ctx.Z[ctx.ordem_y]
    return np.linalg.norm(Zs[1:] - Zs[:-1], axis=1)


def distancias_vizinhos_em_y_vizglobal(ctx_completo, mascara) -> np.ndarray:
    """S2 sobre os pares consecutivos em y do dataset inteiro, internos ao grupo."""
    ctx_completo.exigir_minimo()
    ordem = ctx_completo.ordem_y
    Zs = ctx_completo.Z[ordem]
    interna = mascara[ordem][:-1] & mascara[ordem][1:]
    if not np.any(interna):
        raise MedidaIndefinida("grupo_pequeno")
    return np.linalg.norm(Zs[1:] - Zs[:-1], axis=1)[interna]


def erros_1nn_grupo(ctx: ContextoGrupo) -> np.ndarray:
    """Erro quadratico de 1-NN com vizinhos buscados so dentro do grupo (S3)."""
    ctx.exigir_minimo()
    return ctx.erros_1nn


def erros_1nn_vizglobal(ctx_completo: ContextoGrupo, mascara: np.ndarray) -> np.ndarray:
    """S3 da ficha 4.6: vizinho do dataset inteiro, agregacao restrita ao grupo.

    Mais pessimista para os raros, porque um caso raro pode nao ter nenhum
    vizinho raro por perto -- e foi a variante que mais separou os grupos na
    primeira extracao (fator 4,8).
    """
    ctx_completo.exigir_minimo()
    if not np.any(mascara):
        raise MedidaIndefinida("grupo_vazio")
    return ctx_completo.erros_1nn[mascara]


def erros_interpolados_1nn(ctx: ContextoGrupo) -> np.ndarray:
    """Erro quadratico do 1-NN do grupo nos pontos interpolados (S4)."""
    ctx.exigir_minimo()
    Zi, yi = ctx.interpolados
    return (yi - predicao_1nn(ctx.Z, ctx.y, Zi)) ** 2


def erros_interpolados_1nn_vizglobal(ctx_completo, mascara) -> np.ndarray:
    """S4 com interpolacoes do grupo e 1-NN buscado no dataset inteiro."""
    ctx_completo.exigir_minimo()
    ctx_grupo = ContextoGrupo(ctx_completo.Z[mascara], ctx_completo.y[mascara])
    ctx_grupo.exigir_minimo()
    Zi, yi = ctx_grupo.interpolados
    return (yi - predicao_1nn(ctx_completo.Z, ctx_completo.y, Zi)) ** 2


# ------------------------------------------------------------ dimensionalidade


def T2(ctx: ContextoGrupo) -> float:
    """Exemplos por dimensao, n/d, como na equacao 13 de Lorena et al.

    Atencao ao sentido: e a UNICA medida do conjunto em que valores menores
    indicam mais complexidade (dados mais esparsos), e a unica que o artigo
    manda NAO normalizar. A primeira implementacao usava d/n, o reciproco, o
    que invertia a leitura.

    Fora de X: no grupo raro e proporcional a |R| e duplica `pct_relevante`,
    que ja esta na base. Mantida para a comparacao entre grupos e como teste
    de sanidade da analise de redundancia, onde T2_all deve agrupar com
    `instances_per_feature` -- sao agora a mesma quantidade.
    """
    if ctx.n == 0:
        raise MedidaIndefinida("grupo_vazio")
    if ctx.p == 0:
        raise MedidaIndefinida("sem_atributo_valido")
    return float(ctx.n / ctx.p)


# ------------------------------------------------------------------ registro


def _media(v: np.ndarray) -> float:
    return float(np.mean(v))


def _media_quadrados(v: np.ndarray) -> float:
    return float(np.mean(v**2))


def _desvio(v: np.ndarray) -> float:
    return float(np.std(v, ddof=1))


@dataclass(frozen=True)
class Medida:
    """Uma medida, com a sua fonte de elementos e o seu agregador.

    Fonte unica de verdade: extracao, analise e testes leem daqui. Medidas
    agregativas (`elementos` preenchido) geram o valor bruto e o `_sq`, do
    mesmo calculo intermediario. Medidas escalares geram so o bruto.
    """

    nome: str
    familia: str
    agregador: callable | None = None
    elementos: callable | None = None
    elementos_vizglobal: callable | None = None
    escalar: callable | None = None
    elementos_treino: callable | None = None
    em_X: bool = True
    nota: str = ""

    @property
    def agregativa(self) -> bool:
        return self.elementos is not None


MEDIDAS_META: dict[str, Medida] = {
    m.nome: m
    for m in [
        Medida("C1", "correlacao", escalar=C1),
        Medida("C2", "correlacao", escalar=C2),
        Medida("C3", "correlacao", escalar=C3, nota="divide pelo n do grupo"),
        Medida("C4", "correlacao", escalar=C4),
        Medida(
            "L1",
            "linearidade",
            agregador=_media,
            elementos=residuos_cv,
            elementos_vizglobal=residuos_vizglobal,
            elementos_treino=residuos_treino,
        ),
        Medida(
            "L2",
            "linearidade",
            agregador=_media_quadrados,
            elementos=residuos_cv,
            elementos_vizglobal=residuos_vizglobal,
            elementos_treino=residuos_treino,
        ),
        Medida(
            "L3",
            "linearidade",
            agregador=_media,
            elementos=erros_interpolados_lineares,
            elementos_vizglobal=erros_interpolados_lineares_vizglobal,
        ),
        Medida(
            "S1",
            "suavidade",
            agregador=_media,
            elementos=arestas_mst,
            elementos_vizglobal=arestas_mst_vizglobal,
        ),
        Medida(
            "S1_sd",
            "suavidade",
            agregador=_desvio,
            elementos=arestas_mst,
            elementos_vizglobal=arestas_mst_vizglobal,
        ),
        Medida(
            "S2",
            "suavidade",
            agregador=_media,
            elementos=distancias_vizinhos_em_y,
            elementos_vizglobal=distancias_vizinhos_em_y_vizglobal,
        ),
        Medida(
            "S3",
            "suavidade",
            agregador=_media,
            elementos=erros_1nn_grupo,
            elementos_vizglobal=erros_1nn_vizglobal,
        ),
        Medida(
            "S4",
            "suavidade",
            agregador=_media,
            elementos=erros_interpolados_1nn,
            elementos_vizglobal=erros_interpolados_1nn_vizglobal,
        ),
        Medida(
            "T2",
            "dimensionalidade",
            escalar=T2,
            em_X=False,
            nota="degenera em contador de raros; fora de X",
        ),
    ]
}

MEDIDAS_EM_X = tuple(nome for nome, m in MEDIDAS_META.items() if m.em_X)


@dataclass
class Resultado:
    """Valores de uma medida num conjunto, com o motivo quando indefinida."""

    valores: dict[str, float]
    motivos: dict[str, str]


def _tentar(fn, *args) -> tuple[float, str | None]:
    try:
        return fn(*args), None
    except MedidaIndefinida as e:
        return np.nan, e.motivo


def calcular_grupo(ctx: ContextoGrupo, medidas=None) -> Resultado:
    """Todas as medidas DENTRO do conjunto, brutas e squashed.

    Colunas produzidas por medida agregativa M: `M` (bruta), `M_sq`
    (squashed), `M_treino`/`M_treino_sq` quando a medida tem variante
    in-sample. Medidas escalares produzem so `M`.
    """
    medidas = medidas or MEDIDAS_META.keys()
    valores: dict[str, float] = {}
    motivos: dict[str, str] = {}

    for nome in medidas:
        m = MEDIDAS_META[nome]
        if m.agregativa:
            _registrar_agregativa(valores, motivos, nome, m, m.elementos, ctx)
            if m.elementos_treino is not None:
                _registrar_agregativa(
                    valores, motivos, f"{nome}_treino", m, m.elementos_treino, ctx
                )
        else:
            valor, motivo = _tentar(m.escalar, ctx)
            valores[nome] = valor
            if motivo:
                motivos[nome] = motivo
    return Resultado(valores, motivos)


def calcular_vizglobal(
    ctx_completo: ContextoGrupo, mascara: np.ndarray, medidas=None
) -> Resultado:
    """Variantes `_vizglobal`: geometria do dataset inteiro, agregacao no grupo.

    Quando a mascara e toda verdadeira o valor e IDENTICAMENTE igual ao
    primario, e nesse caso a celula sai NaN com motivo
    `identico_ao_primario` -- na primeira tentativa `S3_viz_global_all` e
    `S3_all` eram a mesma coluna e as duas entraram em X.
    """
    medidas = medidas or MEDIDAS_META.keys()
    mascara = np.asarray(mascara, dtype=bool)
    todos = bool(mascara.all())
    valores: dict[str, float] = {}
    motivos: dict[str, str] = {}

    for nome in medidas:
        m = MEDIDAS_META[nome]
        if m.elementos_vizglobal is None:
            continue
        chave = f"{nome}_vizglobal"
        if todos:
            valores[chave] = np.nan
            valores[f"{chave}_sq"] = np.nan
            motivos[chave] = "identico_ao_primario"
            continue
        _registrar_agregativa(
            valores, motivos, chave, m, m.elementos_vizglobal, ctx_completo, mascara
        )
    return Resultado(valores, motivos)


def _registrar_agregativa(valores, motivos, chave, medida, fonte, *args) -> None:
    try:
        elementos = np.asarray(fonte(*args), dtype=float)
    except MedidaIndefinida as e:
        valores[chave] = np.nan
        valores[f"{chave}_sq"] = np.nan
        motivos[chave] = e.motivo
        return
    valores[chave] = medida.agregador(elementos)
    valores[f"{chave}_sq"] = medida.agregador(squash(elementos))
