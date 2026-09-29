"""Figuras do relatório final, geradas do zero a partir das trajetórias.

Uso:
    python -m molsim.figuras                  # todas, e reescreve LEGENDAS.md
    python -m molsim.figuras --only C_its     # uma ou mais, pelo nome

Saída em figs/relatorio/: cada figura em PDF (vetorial, para LaTeX) e PNG
300 dpi (para Word), já no tamanho de impressão — 16 cm de largura para
figura de página inteira, 8 cm para meia coluna. Não precisa reescalar no
relatório, e as fontes saem com 7,5–9 pt.

Convenções:
- vírgula decimal nos eixos e nas legendas (troque VIRGULA para False);
- os 4 estados de referência têm sempre as mesmas 4 cores (CORES_ESTADO);
  regiões de Ramachandran, variantes do K-means e temperaturas usam outras
  cores ou tons de cinza, para que "azul = estado 0" valha no relatório todo;
- tudo com semente fixa: rodar de novo dá as mesmas figuras e os mesmos
  números de legenda.

O prefixo do nome diz a parte do relatório (S sistema, A descritores globais,
D diedros, B K-means periódico, E estados, C cinética); o relatório numera
as figuras na ordem em que as usar.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter, ScalarFormatter  # noqa: E402

from molsim import kinetics as kin  # noqa: E402
from molsim import parametros as P  # noqa: E402
from molsim.data import (  # noqa: E402
    DIHEDRAL_NAMES, load_features, ramachandran_region, read_xyz, state_name,
)
from molsim.kmeans_variants import (  # noqa: E402
    VARIANTS, best_of, canonical_states, chord_sq, convergence_program,
    kmeans_periodic, match_states,
)

SAIDA = P.RAIZ / "figs" / "relatorio"
VIRGULA = True

CM = 1 / 2.54
INTEIRA = 16 * CM
MEIA = 8 * CM

CORES_ESTADO = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CORES_REGIAO = {"α": "#e87ba4", "β": "#008300", "P": "#4a3aa7", "L": "#e34948"}
NOMES_REGIAO = {"α": "α_R", "β": "β", "P": "PPII", "L": "α_L"}
CORES_VARIANTE = {"tutorial": "#e34948", "mista": "#4a3aa7", "corda": "#008300"}
MARCA_VARIANTE = {"tutorial": "X", "mista": "s", "corda": "o"}
CINZA = "#8a8984"
TINTA = "#2b2b2a"
CPK = {"C": "#909090", "N": "#3050F8", "O": "#FF0D0D", "H": "#F4F4F4"}
RAIO = {"C": 42, "N": 42, "O": 42, "H": 14}
RESIDUOS = [f"Ala{i}" for i in range(1, 7)]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.4, "lines.markersize": 4,
    "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
})


# --------------------------------------------------------------------------
# Formatação
# --------------------------------------------------------------------------
def br(x: float, casas: int = 2) -> str:
    """Número com vírgula decimal."""
    s = f"{x:.{casas}f}"
    return s.replace(".", ",") if VIRGULA else s


def pct(x: float, casas: int = 0) -> str:
    return br(100 * x, casas) + "%"


class _Virgula(ScalarFormatter):
    def __call__(self, x, pos=None):
        s = super().__call__(x, pos)
        return s.replace(".", ",") if VIRGULA else s


def _log_fmt(v, _pos):
    s = f"{v:g}"
    return s.replace(".", ",") if VIRGULA else s


def _formatar_eixos(fig) -> None:
    for ax in fig.axes:
        if ax.name == "polar":
            continue
        for eixo in (ax.xaxis, ax.yaxis):
            if eixo.get_scale() == "log":
                eixo.set_major_formatter(FuncFormatter(_log_fmt))
                eixo.set_minor_formatter(NullFormatter())
            elif isinstance(eixo.get_major_formatter(), ScalarFormatter):
                eixo.set_major_formatter(_Virgula())


def salvar(fig, nome: str) -> None:
    _formatar_eixos(fig)
    SAIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAIDA / f"{nome}.pdf")
    fig.savefig(SAIDA / f"{nome}.png")
    plt.close(fig)


def letra(ax, s: str) -> None:
    """Rótulo de painel (a), (b), ... no canto superior esquerdo."""
    ax.text(-0.02, 1.02, f"({s})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, fontweight="bold", color=TINTA)


def eixo_rama(ax, rotulo_x=True, rotulo_y=True) -> None:
    # marcas em ±90 e 0: com ±180 os rótulos de painéis vizinhos se encostam
    ax.set(xlim=(-180, 180), ylim=(-180, 180), xticks=[-90, 0, 90], yticks=[-90, 0, 90])
    ax.set_aspect("equal")
    ax.grid(False)
    if rotulo_x:
        ax.set_xlabel("φ (°)")
    if rotulo_y:
        ax.set_ylabel("ψ (°)")


def fronteiras_rama(ax) -> None:
    kw = dict(color=CINZA, lw=0.5, zorder=3)
    ax.axvline(0, **kw)
    ax.plot([-180, 0], [50, 50], **kw)
    ax.plot([-180, 0], [-120, -120], **kw)
    ax.plot([-100, -100], [50, 180], **kw)
    ax.plot([-100, -100], [-180, -120], **kw)


# --------------------------------------------------------------------------
# Dados: tudo calculado uma vez, sob demanda
# --------------------------------------------------------------------------
class Dados:
    @cached_property
    def df500(self):
        return load_features(P.TRAJ_500K)

    @cached_property
    def df300(self):
        return load_features(P.TRAJ_300K)

    @cached_property
    def X5(self):
        return self.df500[DIHEDRAL_NAMES].values

    @cached_property
    def X3(self):
        return self.df300[DIHEDRAL_NAMES].values

    @cached_property
    def xyz500(self):
        return read_xyz(P.TRAJ_500K)

    @cached_property
    def can(self):
        return canonical_states(self.X5, k=P.K_ESTADOS)

    @property
    def lab5(self):
        return self.can.labels

    @cached_property
    def nomes(self):
        return [state_name(c) for c in self.can.centers]

    @cached_property
    def can3(self):
        return canonical_states(self.X3, k=P.K_ESTADOS)

    @cached_property
    def nomes3(self):
        return [state_name(c) for c in self.can3.centers]

    @cached_property
    def pops(self):
        return kin.populations(self.lab5, P.K_ESTADOS)

    @cached_property
    def dG(self):
        k = P.K_ESTADOS
        dg = kin.free_energy(self.pops, 500.0)
        boot = kin.block_bootstrap(self.lab5, lambda l: kin.free_energy(kin.populations(l, k), 500.0),
                                   n_blocks=P.BOOT_BLOCOS, n_boot=P.BOOT_N_POP)
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        return dg, lo, hi

    @cached_property
    def permanencia(self):
        c = kin.count_matrix(self.lab5, 1, P.K_ESTADOS)
        return np.diag(kin.transition_matrix(c, reversible=False))

    @cached_property
    def sementes(self):
        """Mesmo procedimento do notebook 05 §1.2."""
        k = P.K_ESTADOS
        objetivos, perm_s = [], []
        for s in range(P.N_SEMENTES):
            r = kmeans_periodic(self.X5, k, "corda", max_iter=200, seed=s)
            perm = match_states(self.can.centers, r.centers)
            novo = np.empty(k, dtype=int)
            novo[perm] = np.arange(k)
            l = novo[r.labels]
            objetivos.append(r.objective_history[-1])
            perm_s.append(np.diag(kin.transition_matrix(kin.count_matrix(l, 1, k), reversible=False)))
        objetivos = np.array(objetivos)
        return dict(objetivos=objetivos, perm=np.array(perm_s),
                    n_minimos=np.unique(objetivos.round(3)).size,
                    frac_melhor=np.isclose(objetivos, objetivos.min()).mean())

    @cached_property
    def cotovelo(self):
        ks = np.arange(2, 15)
        obj = np.array([best_of(self.X5, k, n_init=50).objective_history[-1] for k in ks])
        return ks, obj

    @cached_property
    def convergencia(self):
        linhas = []
        for k in range(2, 13):
            for v in VARIANTS:
                st = convergence_program(self.X5, k=k, variant=v, n_init=50, seed0=0, max_iter=200)
                linhas.append(dict(k=k, variante=v, ciclo=st.rate_cycled,
                                   violacao=st.monotonic_violations / st.n_init))
        return pd.DataFrame(linhas)

    @cached_property
    def its_k4(self):
        return kin.implied_timescales(self.lab5, P.LAGS, P.K_ESTADOS)

    @cached_property
    def micro(self):
        return {km: best_of(self.X5, km, n_init=P.N_INIT_MICRO) for km in P.K_MICRO}

    @cached_property
    def its_micro(self):
        return {km: kin.implied_timescales(r.labels, P.LAGS, km) for km, r in self.micro.items()}

    @cached_property
    def boot_micro(self):
        return {km: kin.bootstrap_timescales(r.labels, P.LAG_REF, km, n_blocks=P.BOOT_BLOCOS,
                                             n_boot=P.BOOT_N)
                for km, r in self.micro.items()}

    @cached_property
    def macro(self):
        km = max(P.K_MICRO)
        r = self.micro[km]
        return kin.spectral_split(r.labels, P.LAG_REF, km)[r.labels]

    @cached_property
    def tem_L(self):
        return (self.X5[:, 0::2] > 0).any(axis=1)

    @cached_property
    def knn300(self):
        from sklearn.neighbors import KNeighborsClassifier
        pred = KNeighborsClassifier(n_neighbors=15).fit(imersao(self.X5), self.lab5).predict(imersao(self.X3))
        return kin.populations(pred, P.K_ESTADOS)

    @cached_property
    def medoides(self):
        """Frame de cada estado com a menor soma de distâncias de corda aos
        outros membros do estado."""
        out = []
        for s in range(P.K_ESTADOS):
            idx = np.flatnonzero(self.lab5 == s)
            Xs = self.X5[idx]
            soma = np.array([chord_sq(Xs, x).sum() for x in Xs])
            out.append(int(idx[np.argmin(soma)]))
        return out


def imersao(X):
    t = np.radians(X)
    return np.hstack([np.cos(t), np.sin(t)])


# --------------------------------------------------------------------------
# Registro
# --------------------------------------------------------------------------
@dataclass
class Figura:
    nome: str
    titulo: str
    secao: str
    tipo: str          # "principal" ou "apêndice"
    questao: str       # questão do exercício que a figura sustenta, ou ""
    conferir: str      # onde os números da legenda são impressos
    gerar: Callable[[Dados], str]


FIGURAS: list[Figura] = []


def figura(nome, titulo, secao, tipo="principal", questao="", conferir=""):
    def deco(f):
        FIGURAS.append(Figura(nome, titulo, secao, tipo, questao, conferir, f))
        return f
    return deco


# --------------------------------------------------------------------------
# Renderização simples de estrutura (projeção 2D com profundidade)
# --------------------------------------------------------------------------
BACKBONE = [1] + [3 + 6 * i + j for i in range(6) for j in (0, 2, 4)] + [39]


def ligacoes(coords: np.ndarray, names: list[str]) -> list[tuple[int, int]]:
    d = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    h = np.array([n[0] == "H" for n in names])
    corte = np.where(h[:, None] | h[None, :], 1.25, 1.75)
    i, j = np.nonzero(np.triu(d < corte, k=1))
    return list(zip(i.tolist(), j.tolist()))


def kabsch(movel: np.ndarray, ref: np.ndarray, idx) -> np.ndarray:
    """Superpõe `movel` a `ref` usando só os átomos `idx`; devolve `movel` movido."""
    cm, cr = movel[idx].mean(0), ref[idx].mean(0)
    h = (movel[idx] - cm).T @ (ref[idx] - cr)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1, 1, d]) @ vt
    return (movel - cm) @ rot + cr


def eixos_de_vista(coords: np.ndarray) -> np.ndarray:
    """Base ortonormal DIREITA com o eixo mais longo da molécula na horizontal.
    Uma base esquerda espelharia a molécula — trocaria L por D na figura."""
    c = coords[BACKBONE] - coords[BACKBONE].mean(0)
    _, vec = np.linalg.eigh(c.T @ c)
    base = vec[:, ::-1]
    if np.linalg.det(base) < 0:
        base[:, 2] *= -1
    return base


def desenhar_molecula(ax, coords, names, bonds, base, rotulos=True, destaque=None) -> None:
    xyz = (coords - coords[BACKBONE].mean(0)) @ base
    x, y, z = xyz.T
    itens = [(z[a] + z[b]) / 2 - 0.01 for a, b in bonds]
    ordem = np.argsort(np.r_[itens, z])
    zorder = np.empty_like(ordem)
    zorder[ordem] = np.arange(len(ordem))
    nb = len(bonds)
    if destaque is not None:
        a, b = destaque
        ax.plot([x[a], x[b]], [y[a], y[b]], color=CORES_REGIAO["α"], lw=7, alpha=0.55,
                solid_capstyle="round", zorder=0)
    for k, (a, b) in enumerate(bonds):
        ax.plot([x[a], x[b]], [y[a], y[b]], color="#5a5955", lw=1.8, solid_capstyle="round",
                zorder=2 + zorder[k])
    for i, n in enumerate(names):
        e = n[0]
        ax.scatter(x[i], y[i], s=RAIO[e], color=CPK[e], edgecolors=TINTA, linewidths=0.4,
                   zorder=2 + zorder[nb + i])
    if rotulos:
        for r in range(6):
            ca = 3 + 6 * r + 2
            ax.annotate(RESIDUOS[r], (x[ca], y[ca]), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=6.5, color=TINTA, zorder=500)
        ax.annotate("Ace", (x[0], y[0]), xytext=(-8, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=6.5, color=TINTA, zorder=500)
        ax.annotate("NH₂", (x[39], y[39]), xytext=(8, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=6.5, color=TINTA, zorder=500)
    ax.set_aspect("equal")
    ax.axis("off")


def titulo_com_marca(ax, texto: str, cor: str) -> None:
    """Título em tinta com um quadrado da cor do estado ao lado: a cor identifica,
    o texto continua legível."""
    from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea
    from matplotlib.patches import Rectangle

    marca = DrawingArea(8, 8)
    marca.add_artist(Rectangle((0, 0), 8, 8, facecolor=cor, edgecolor="none"))
    rotulo = TextArea(texto, textprops=dict(color=TINTA, fontsize=8.5))
    caixa = HPacker(children=[marca, rotulo], align="center", pad=0, sep=4)
    ax.add_artist(AnchoredOffsetbox(loc="lower center", child=caixa, pad=0, frameon=False,
                                    bbox_to_anchor=(0.5, 1.0), bbox_transform=ax.transAxes,
                                    borderpad=0.3))


def legenda_elementos(ax, loc="lower right") -> None:
    alcas = [Line2D([], [], marker="o", ls="", markersize=5, markerfacecolor=CPK[e],
                    markeredgecolor=TINTA, markeredgewidth=0.4, label=e) for e in "CNOH"]
    ax.legend(handles=alcas, loc=loc, ncol=4, handletextpad=0.1, columnspacing=0.8)


# --------------------------------------------------------------------------
# S — o sistema
# --------------------------------------------------------------------------
@figura("S_estrutura", "Estrutura do Ace-(Ala)₆-NH₂ e a ligação cis",
        "Metodologia › Sistema e dados", questao="apoia Q1",
        conferir="notebook 05 §0 (tabela de ω)")
def fig_estrutura(d: Dados) -> str:
    names, coords = d.xyz500
    c0 = coords[0]
    bonds = ligacoes(c0, names)
    fig, ax = plt.subplots(figsize=(INTEIRA, 5.2 * CM))
    desenhar_molecula(ax, c0, names, bonds, eixos_de_vista(c0), destaque=(19, 21))
    legenda_elementos(ax, loc="lower left")
    xyz = (c0 - c0[BACKBONE].mean(0)) @ eixos_de_vista(c0)
    meio = (xyz[19, :2] + xyz[21, :2]) / 2
    ax.annotate("ligação cis Ala3–Ala4", meio, xytext=(0, -26), textcoords="offset points",
                ha="center", fontsize=7, color=TINTA,
                arrowprops=dict(arrowstyle="-", color=TINTA, lw=0.6))
    salvar(fig, "S_estrutura")
    return (f"Ace-(Ala)₆-NH₂ em modelo de átomo unido ({len(names)} sítios; só os H polares "
            "são explícitos), no primeiro frame da trajetória de 500 K. Em destaque, a ligação "
            "peptídica entre Ala3 e Ala4, que é *cis* em todos os frames das duas trajetórias "
            "fornecidas; as demais são *trans* (a da acetila passa por um único episódio *cis* "
            "curto a 500 K). A ligação *cis* dobra a cadeia no meio. Renderização por projeção "
            "nos eixos principais do esqueleto.")


# --------------------------------------------------------------------------
# A — descritores globais (Fase A)
# --------------------------------------------------------------------------
@figura("A_rg_rmsd", "Raio de giro e RMSD ao longo da trajetória de 500 K",
        "Resultados › Descritores globais", questao="Q2",
        conferir="notebook 01, seção 'Calculation of features, part 1'")
def fig_rg_rmsd(d: Dados) -> str:
    df = d.df500
    t = df["time"].values / 1000
    fig = plt.figure(figsize=(INTEIRA, 7.2 * CM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], wspace=0.28, hspace=0.12)
    a1 = fig.add_subplot(gs[0, 0])
    a2 = fig.add_subplot(gs[1, 0], sharex=a1)
    a3 = fig.add_subplot(gs[:, 1])
    a1.plot(t, df["radgyr"], color=TINTA, lw=0.5)
    a1.set_ylabel("Rg (Å)")
    a1.tick_params(labelbottom=False)
    a2.plot(t, df["rmsd"], color=TINTA, lw=0.5)
    a2.set_ylabel("RMSD (Å)")
    a2.set_xlabel("tempo (ns)")
    sc = a3.scatter(df["radgyr"], df["rmsd"], c=t, cmap="Blues", s=5, lw=0, rasterized=True,
                    vmin=-5, vmax=t.max())
    a3.set_xlabel("Rg (Å)")
    a3.set_ylabel("RMSD (Å)")
    cb = fig.colorbar(sc, ax=a3, fraction=0.05, pad=0.02)
    cb.set_label("tempo (ns)")
    cb.ax.set_ylim(0, t.max())
    letra(a1, "a")
    letra(a2, "b")
    letra(a3, "c")
    salvar(fig, "A_rg_rmsd")
    rg, rm = df["radgyr"], df["rmsd"]
    return (f"(a) Raio de giro e (b) RMSD em relação ao primeiro frame, após superposição "
            f"ótima, ao longo dos 20 ns a 500 K (1 frame = 10 ps). (c) Mapa Rg × RMSD colorido "
            f"pelo tempo. O Rg varia entre {br(rg.min())} e {br(rg.max())} Å e o RMSD entre "
            f"{br(rm.min())} e {br(rm.max())} Å; a molécula alterna entre patamares, e os "
            "pontos se agrupam em ilhas no mapa porque cada frame foi minimizado antes de ser "
            "gravado.")


@figura("A_kmeans_rg_rmsd", "K-means do scikit-learn em (Rg, RMSD)",
        "Resultados › Descritores globais", questao="Q4",
        conferir="notebook 01, seções 'K-Means clustering' e 'Assessment of the clustering'")
def fig_kmeans_rg_rmsd(d: Dados) -> str:
    from sklearn.cluster import KMeans

    X = d.df500[["radgyr", "rmsd"]].values
    ks = np.arange(2, 15)
    inercia = [KMeans(n_clusters=k, random_state=0).fit(X).inertia_ for k in ks]
    km = KMeans(n_clusters=4, random_state=0).fit(X)
    ordem = np.argsort(-np.bincount(km.labels_))
    novo = np.empty(4, dtype=int)
    novo[ordem] = np.arange(4)
    lab = novo[km.labels_]
    cent = km.cluster_centers_[ordem]

    fig = plt.figure(figsize=(INTEIRA, 4.6 * CM))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.25, 1, 1, 1, 1], wspace=0.12)
    a0 = fig.add_subplot(gs[0])
    a0.plot(ks, inercia, "o-", color=TINTA, ms=3)
    a0.axvline(4, color=CINZA, lw=0.8, ls="--")
    a0.set(xlabel="K", ylabel="inércia (Å²)")
    letra(a0, "a")
    eixos = [fig.add_subplot(gs[1])]
    eixos += [fig.add_subplot(gs[i + 1], sharex=eixos[0], sharey=eixos[0]) for i in range(1, 4)]
    for g, ax in enumerate(eixos):
        ax.scatter(X[:, 0], X[:, 1], s=2, color=CINZA, alpha=0.25, lw=0, rasterized=True)
        ax.scatter(X[lab == g, 0], X[lab == g, 1], s=3, color=TINTA, lw=0, rasterized=True)
        ax.scatter(*cent[g], marker="x", s=30, color=CORES_REGIAO["L"], lw=1.4, zorder=5)
        ax.set_title(f"grupo {g + 1} ({pct(np.mean(lab == g))})", fontsize=7.5)
        ax.set_xlabel("Rg (Å)")
        if g == 0:
            ax.set_ylabel("RMSD (Å)")
            letra(ax, "b")
        else:
            ax.tick_params(labelleft=False)
    salvar(fig, "A_kmeans_rg_rmsd")
    faixa = np.ptp(X, axis=0)
    return ("(a) Cotovelo do K-means do scikit-learn sobre (Rg, RMSD), sem padronização, como "
            "no notebook do exercício (`random_state=0`). (b) Os quatro grupos para K = 4, um "
            "por painel (em preto; em cinza, a trajetória inteira; ×, o centroide). Sem "
            f"padronização, o RMSD, que varia numa faixa {br(faixa[1] / faixa[0], 1)} vezes "
            "maior que a do Rg, pesa mais na distância: três grupos são faixas de RMSD, e só o "
            "das conformações mais estendidas (Rg ≈ 5 Å) se separa pelo Rg.")


# --------------------------------------------------------------------------
# D — ângulos diedros
# --------------------------------------------------------------------------
@figura("D_ramachandran_500_300", "Mapas de Ramachandran por resíduo, 500 K e 300 K",
        "Resultados › Ângulos diedros", conferir="notebook 05 §0 (tabela de ocupação)")
def fig_rama_temp(d: Dados) -> str:
    fig, axes = plt.subplots(2, 6, figsize=(INTEIRA, 6.4 * CM), sharex=True, sharey=True)
    bins = np.linspace(-180, 180, 61)
    # rampa de um tom só, começando num azul já visível: 1 frame não pode sumir
    azul = LinearSegmentedColormap.from_list("azul", plt.get_cmap("Blues")(np.linspace(0.4, 1, 256)))
    for lin, (X, temp) in enumerate(((d.X5, "500 K"), (d.X3, "300 K"))):
        for r in range(6):
            ax = axes[lin, r]
            ax.hist2d(X[:, 2 * r], X[:, 2 * r + 1], bins=[bins, bins], cmap=azul, cmin=1,
                      norm=LogNorm(vmin=1, vmax=400), rasterized=True)
            fronteiras_rama(ax)
            eixo_rama(ax, rotulo_x=lin == 1, rotulo_y=r == 0)
            if lin == 0:
                ax.set_title(RESIDUOS[r])
            if r == 0:
                ax.set_ylabel(f"{temp}\nψ (°)")
    for s, (xx, yy) in {"α": (-150, -40), "β": (-160, 130), "P": (-60, 130), "L": (90, 20)}.items():
        axes[0, 0].text(xx, yy, s, fontsize=6.5, color=TINTA, ha="center", va="center")
    fig.subplots_adjust(wspace=0.08, hspace=0.12)
    salvar(fig, "D_ramachandran_500_300")
    return ("Densidade de frames no plano (φ, ψ) de cada resíduo, em escala logarítmica, a "
            "500 K (acima) e a 300 K (abaixo). As linhas cinza delimitam as regiões usadas nos "
            "nomes dos estados: α (α_R), β, P (PPII) e L (α_L, φ > 0). Ala3 fica restrita à "
            "região estendida e Ala4 nunca chega a α_L, efeito da ligação *cis* entre elas; a "
            "300 K a região α_L não é visitada.")


@figura("D_ocupacao_regioes", "Ocupação das regiões de Ramachandran por resíduo",
        "Resultados › Ângulos diedros", conferir="notebook 05 §0 (tabela de ocupação)")
def fig_ocupacao(d: Dados) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(INTEIRA, 5.0 * CM), sharey=True)
    for ax, (X, temp), lt in zip(axes, ((d.X5, "500 K"), (d.X3, "300 K")), "ab"):
        reg = ramachandran_region(X[:, 0::2], X[:, 1::2])
        esq = np.zeros(6)
        for rg in "αβPL":
            f = (reg == rg).mean(axis=0)
            ax.barh(np.arange(6), f, left=esq, color=CORES_REGIAO[rg], edgecolor="white",
                    lw=0.8, height=0.72, label=NOMES_REGIAO[rg])
            for i in range(6):
                if f[i] >= 0.12:
                    ax.text(esq[i] + f[i] / 2, i, pct(f[i]), ha="center", va="center",
                            fontsize=6.5, color="white")
            esq += f
        ax.set_yticks(range(6), RESIDUOS)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
        ax.set_title(temp)
        ax.grid(False)
        letra(ax, lt)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.1), ncol=4, handlelength=1)
    salvar(fig, "D_ocupacao_regioes")
    return ("Fração do tempo que cada resíduo passa em cada região de Ramachandran, (a) a "
            "500 K e (b) a 300 K. A 300 K não há α_L em nenhum resíduo; Ala5 fica quase só em "
            "α_R, e Ala3 fica sempre na região estendida nas duas temperaturas.")


@figura("D_rg_temperaturas", "Distribuição do raio de giro a 500 K e 300 K",
        "Resultados › Descritores globais", tipo="apêndice",
        conferir="molsim/data.py (massas por elemento)")
def fig_rg_temp(d: Dados) -> str:
    fig, ax = plt.subplots(figsize=(MEIA, 5.2 * CM))
    bins = np.linspace(3.3, 5.2, 58)
    ax.hist(d.df300["radgyr"], bins=bins, density=True, color=CINZA, alpha=0.5, label="300 K")
    ax.hist(d.df500["radgyr"], bins=bins, density=True, histtype="step", color=TINTA, lw=1.2,
            label="500 K")
    ax.set(xlabel="raio de giro (Å)", ylabel="densidade (1/Å)")
    ax.legend()
    salvar(fig, "D_rg_temperaturas")
    return ("Distribuição do raio de giro nas duas temperaturas, com massas atribuídas pelo "
            "elemento. Com os nomes de tipo GROMOS do arquivo de 300 K, o MDAnalysis atribui "
            "massa zero a CH3, CH1 e NH1, e esta distribuição sairia errada sem aviso.")


# --------------------------------------------------------------------------
# B — K-means periódico (contribuição)
# --------------------------------------------------------------------------
@figura("B_media_circular", "Média aritmética × média circular perto de ±180°",
        "Metodologia › K-means periódico", conferir="tests/test_kmeans_variants.py")
def fig_media_circular(d: Dados) -> str:
    ang = np.array([-170.0, 175.0, 170.0, -175.0])
    fig = plt.figure(figsize=(MEIA, 7.4 * CM))
    ax = fig.add_subplot(projection="polar")
    ax.scatter(np.radians(ang), np.ones(4), s=26, color=TINTA, zorder=3, label="ângulos")
    ax.scatter([0], [1], s=60, marker=MARCA_VARIANTE["tutorial"], color=CORES_VARIANTE["tutorial"],
               zorder=4, label="média aritmética: 0°")
    ax.scatter([np.pi], [1], s=90, marker="o", facecolor="none", edgecolor=CORES_VARIANTE["corda"],
               lw=1.6, zorder=4, label="média circular: 180°")
    ax.set_ylim(0, 1.15)
    ax.set_yticks([])
    ax.set_xticks(np.radians([0, 90, 180, 270]), ["0°", "90°", "±180°", "−90°"])
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1)
    salvar(fig, "B_media_circular")
    return ("Os ângulos −170°, 175°, 170° e −175° estão todos a menos de 10° de ±180°. A média "
            "aritmética (0°) cai do lado oposto do círculo, com Σd² = 119 050 grau²; a média "
            "circular, μ = atan2(⟨sin θ⟩, ⟨cos θ⟩), dá 180° e Σd² = 250 grau². É a atualização "
            "da variante `tutorial` (a do material de referência) que produz o primeiro "
            "resultado.")


@figura("B_convergencia", "Ciclo limite e monotonicidade das três variantes",
        "Resultados › K-means periódico", questao="Q6",
        conferir="notebook 02, 'Comparação das três variantes'")
def fig_convergencia(d: Dados) -> str:
    tab = d.convergencia
    fig, axes = plt.subplots(1, 2, figsize=(INTEIRA, 5.4 * CM), sharex=True)
    for ax, col, titulo, lt in ((axes[0], "ciclo", "presas em ciclo limite", "a"),
                                (axes[1], "violacao", "com o objetivo subindo", "b")):
        for v in VARIANTS:
            sub = tab[tab.variante == v]
            ax.plot(sub.k, sub[col], marker=MARCA_VARIANTE[v], color=CORES_VARIANTE[v], ms=4,
                    lw=1.2, label=v)
        ax.set(xlabel="K", ylim=(-0.03, 1.03))
        ax.set_title(titulo)
        letra(ax, lt)
    axes[0].set_ylabel("fração das 50 inicializações")
    axes[0].legend(loc="upper left")
    salvar(fig, "B_convergencia")
    tut = tab[(tab.variante == "tutorial") & (tab.k >= 3)]
    return (f"Programa de medida sobre os 12 diedros a 500 K, com 50 inicializações por K. "
            f"(a) A variante `tutorial` fica presa em ciclo limite em {pct(tut.ciclo.min())} a "
            f"{pct(tut.ciclo.max())} das inicializações para K ≥ 3; `mista` e `corda` nunca. "
            f"(b) Só em `corda`, em que a atualização é o minimizador exato da métrica de "
            f"atribuição, o objetivo nunca sobe entre iterações "
            f"({int(tab[tab.variante == 'corda'].violacao.sum() * 50)} violações em "
            f"{50 * tab.k.nunique()} rodadas).")


@figura("B_historico_objetivo", "Objetivo a cada iteração, mesma inicialização",
        "Resultados › K-means periódico", tipo="apêndice", questao="Q6",
        conferir="molsim/kmeans_variants.py (objective_history)")
def fig_historico(d: Dados) -> str:
    semente, k = 2, 4
    fig, axes = plt.subplots(1, 3, figsize=(INTEIRA, 4.8 * CM))
    unid = {"tutorial": "Σ d² (10⁶ grau²)", "mista": "Σ d² (10⁶ grau²)", "corda": "Σ(1 − cos Δθ)"}
    desc = {}
    for ax, v, lt in zip(axes, VARIANTS, "abc"):
        r = kmeans_periodic(d.X5, k, v, max_iter=200, seed=semente)
        h = np.array(r.objective_history) / (1e6 if v != "corda" else 1)
        it = np.arange(1, len(h) + 1)
        ax.plot(it, h, "-", color=CORES_VARIANTE[v], lw=1.2,
                marker=MARCA_VARIANTE[v], ms=3.5)
        sobe = np.r_[False, np.diff(h) > 1e-12]
        ax.scatter(it[sobe], h[sobe], s=46, facecolor="none", edgecolor=TINTA, lw=0.9, zorder=5)
        estado = "ciclo limite" if r.cycled else ("convergiu" if r.converged else "não parou")
        desc[v] = (estado, int(sobe.sum()), len(h))
        n = int(sobe.sum())
        ax.set_title(f"{v}\n{estado} · {n} {'subida' if n == 1 else 'subidas'}", fontsize=8)
        ax.set(xlabel="iteração", ylabel=unid[v])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        letra(ax, lt)
    fig.subplots_adjust(wspace=0.55)
    salvar(fig, "B_historico_objetivo")
    return (f"Objetivo de cada variante a cada iteração, com K = {k} e a mesma inicialização "
            f"(semente {semente}); círculos marcam iterações em que o objetivo subiu. Cada "
            "painel tem a sua própria unidade: as variantes otimizam objetivos diferentes e "
            "não são comparáveis pela altura das curvas. `tutorial` alterna entre duas "
            f"partições até repetir uma já visitada ({desc['tutorial'][0]}); `corda` desce a "
            "cada passo, como a teoria garante.")


@figura("B_cotovelo_corda", "Cotovelo da variante coerente",
        "Resultados › Escolha de K", questao="Q4", conferir="notebook 05 §1.0")
def fig_cotovelo(d: Dados) -> str:
    ks, obj = d.cotovelo
    ganho = -np.diff(obj) / obj[:-1]
    fig, ax = plt.subplots(figsize=(MEIA, 5.4 * CM))
    ax.plot(ks, obj, "o-", color=TINTA, ms=3.5)
    ax.axvline(P.K_ESTADOS, color=CINZA, lw=0.8, ls="--")
    ax.annotate("K = 4 (usado)", (P.K_ESTADOS, obj.max()), xytext=(4, 0),
                textcoords="offset points", color=CINZA, va="top", fontsize=7)
    ax.set(xlabel="K", ylabel="objetivo final  Σ(1 − cos Δθ)")
    salvar(fig, "B_cotovelo_corda")
    i4 = list(ks).index(P.K_ESTADOS)
    i9 = list(ks).index(9)
    return (f"Objetivo final da variante `corda` (melhor de 50 inicializações) em função de K. "
            f"O ganho relativo ao passar de K para K+1 cai só de {pct(ganho[i4 - 1])} (3→4) para "
            f"{pct(ganho[i4])} (4→5) e fica entre {pct(ganho[i4:i9].min())} e "
            f"{pct(ganho[i4:i9].max())} até K = 9; depois cai para ~{pct(ganho[i9:].mean())}. "
            "O joelho em K = 4 é fraco: K = 4 foi mantido para comparação com o material de "
            "referência, não porque o cotovelo o determine.")


@figura("B_ari_metodos", "Concordância entre métodos de agrupamento (ARI)",
        "Resultados › Comparação com outros métodos", questao="Q7",
        conferir="notebook 02, 'Baselines: GMM, aglomerativo (Ward) e DBSCAN'")
def fig_ari(d: Dados) -> str:
    from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture

    X = d.X5
    metodos = {
        "corda": kmeans_periodic(X, 4, "corda", seed=0).labels,
        "mista": kmeans_periodic(X, 4, "mista", seed=0).labels,
        "K-means\n(sklearn)": KMeans(n_clusters=4, random_state=0, n_init=10).fit(X).labels_,
        "GMM": GaussianMixture(n_components=4, random_state=0).fit_predict(X),
        "Ward": AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(X),
        "DBSCAN": DBSCAN(eps=30.0, min_samples=12).fit(X).labels_,
    }
    nomes = list(metodos)
    m = np.array([[adjusted_rand_score(metodos[a], metodos[b]) for b in nomes] for a in nomes])
    fig, ax = plt.subplots(figsize=(9.5 * CM, 8 * CM))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(nomes)), nomes, rotation=45, ha="right")
    ax.set_yticks(range(len(nomes)), nomes)
    ax.grid(False)
    for i in range(len(nomes)):
        for j in range(len(nomes)):
            ax.text(j, i, br(m[i, j]), ha="center", va="center", fontsize=6.8,
                    color="white" if m[i, j] > 0.6 else TINTA)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("ARI")
    salvar(fig, "B_ari_metodos")
    ari_ref = adjusted_rand_score(metodos["corda"], d.lab5)
    return ("Índice de Rand ajustado entre as partições de seis métodos sobre os 12 diedros a "
            "500 K (1 = partições idênticas, 0 = concordância de acaso). Só `corda` e `mista` "
            "tratam a periodicidade; GMM, Ward e o K-means do scikit-learn usam distância "
            "euclidiana nos ângulos crus, e o DBSCAN (eps = 30°, min_samples = 12) escolhe "
            "sozinho o número de grupos. A partição `corda` desta figura (semente 0) coincide "
            f"com a partição de referência (ARI = {br(ari_ref)}).")


@figura("B_sementes", "Reprodutibilidade da partição K = 4",
        "Resultados › Escolha de K", questao="Q3", conferir="notebook 05 §1.2")
def fig_sementes(d: Dados) -> str:
    sm = d.sementes
    perm = sm["perm"]
    fig, ax = plt.subplots(figsize=(10 * CM, 6 * CM))
    rng = np.random.default_rng(0)
    for i in range(P.K_ESTADOS):
        x = i + rng.uniform(-0.18, 0.18, len(perm))
        ax.scatter(x, 100 * perm[:, i], s=12, color=CORES_ESTADO[i], alpha=0.7,
                   edgecolor="white", linewidth=0.4)
        ax.plot([i - 0.3, i + 0.3], [100 * d.permanencia[i]] * 2, color=TINTA, lw=1.6)
    ax.set_xticks(range(P.K_ESTADOS), [f"estado {i}\n{n}" for i, n in enumerate(d.nomes)])
    ax.set_ylabel("P(permanecer após 10 ps) (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    salvar(fig, "B_sementes")
    return (f"Probabilidade de permanecer em cada estado após 10 ps, para {P.N_SEMENTES} "
            "inicializações da variante `corda` com K = 4 (pontos), alinhadas aos estados de "
            "referência pelo centroide, contra a partição de referência (traço). As "
            f"{P.N_SEMENTES} sementes chegam a {sm['n_minimos']} mínimos locais distintos, e só "
            f"{pct(sm['frac_melhor'])} delas ao de menor objetivo: o mesmo estado vai de "
            "passageiro a quase permanente conforme a inicialização.")


# --------------------------------------------------------------------------
# E — os estados
# --------------------------------------------------------------------------
@figura("E_ramachandran_estados", "Assinatura de Ramachandran de cada estado",
        "Resultados › Os estados", questao="Q5", conferir="notebook 05 §1.1")
def fig_rama_estados(d: Dados) -> str:
    k = P.K_ESTADOS
    fig, axes = plt.subplots(k, 6, figsize=(INTEIRA, 11.4 * CM), sharex=True, sharey=True)
    for s in range(k):
        sel = d.lab5 == s
        for r in range(6):
            ax = axes[s, r]
            phi, psi = d.X5[:, 2 * r], d.X5[:, 2 * r + 1]
            ax.scatter(phi, psi, s=1, color=CINZA, alpha=0.18, lw=0, rasterized=True)
            ax.scatter(phi[sel], psi[sel], s=1.5, color=CORES_ESTADO[s], alpha=0.7, lw=0,
                       rasterized=True)
            eixo_rama(ax, rotulo_x=s == k - 1, rotulo_y=False)
            if s == 0:
                ax.set_title(RESIDUOS[r])
            if r == 0:
                ax.set_ylabel(f"estado {s}\n{d.nomes[s]}", fontsize=7.5)
    fig.subplots_adjust(wspace=0.08, hspace=0.1)
    salvar(fig, "E_ramachandran_estados")
    return ("Cada linha é um estado da partição de referência (K = 4, variante `corda`) e "
            "cada coluna um resíduo: em cinza a trajetória inteira de 500 K, em cor os frames "
            "do estado. O nome resume o centroide com uma letra por resíduo, de Ala1 a Ala6: "
            "α (α_R), β, P (PPII) e L (α_L). Os eixos vão de −180° a 180°.")


@figura("E_dpca_estados", "Os estados na PCA dos diedros",
        "Resultados › Os estados", tipo="apêndice", conferir="notebook 05 §1.1")
def fig_dpca(d: Dados) -> str:
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2).fit(imersao(d.X5))
    Y = pca.transform(imersao(d.X5))
    fig, axes = plt.subplots(1, P.K_ESTADOS, figsize=(INTEIRA, 4.4 * CM), sharex=True, sharey=True)
    for s, ax in enumerate(axes):
        ax.scatter(Y[:, 0], Y[:, 1], s=1.5, color=CINZA, alpha=0.18, lw=0, rasterized=True)
        ax.scatter(Y[d.lab5 == s, 0], Y[d.lab5 == s, 1], s=2, color=CORES_ESTADO[s], alpha=0.7,
                   lw=0, rasterized=True)
        ax.set_title(f"estado {s}  {d.nomes[s]}", fontsize=7.5)
        ax.set_xlabel("PC1")
        ax.grid(False)
    axes[0].set_ylabel("PC2")
    fig.subplots_adjust(wspace=0.08)
    salvar(fig, "E_dpca_estados")
    return ("Projeção nos dois primeiros componentes principais dos diedros, calculados na "
            "imersão (cos θ, sin θ) para respeitar a periodicidade (PCA nos ângulos crus trataria "
            f"−179° e 179° como opostos). PC1 e PC2 explicam "
            f"{pct(pca.explained_variance_ratio_.sum())} da variância; os estados se sobrepõem "
            "na projeção, que perde a maior parte da informação.")


@figura("E_estabilidade", "Energia livre e estabilidade dos estados",
        "Resultados › Os estados", questao="Q7", conferir="notebook 05 §1.1 (tabela)")
def fig_estabilidade(d: Dados) -> str:
    dg, lo, hi = d.dG
    k = P.K_ESTADOS
    fig, axes = plt.subplots(1, 2, figsize=(INTEIRA, 5.6 * CM))
    x = np.arange(k)
    rot = [f"estado {i}\n{n}" for i, n in enumerate(d.nomes)]
    ax = axes[0]
    # pontos, não barras: três dos quatro valores são ~0 e barras sumiriam
    for i in range(k):
        ax.errorbar(x[i], dg[i], yerr=[[dg[i] - lo[i]], [hi[i] - dg[i]]], fmt="o", ms=5,
                    color=CORES_ESTADO[i], ecolor=TINTA, capsize=3, lw=0.9, mec=TINTA, mew=0.4)
    ax.axhline(0, color=CINZA, lw=0.6)
    ax.set_xticks(x, rot)
    ax.set_xlim(-0.6, k - 0.4)
    ax.set_ylabel("ΔG (kcal/mol)")
    ax.grid(axis="x", visible=False)
    letra(ax, "a")
    ax = axes[1]
    b = ax.bar(x, 100 * d.permanencia, color=CORES_ESTADO, width=0.6)
    ax.bar_label(b, labels=[pct(p) for p in d.permanencia], padding=2, fontsize=7)
    ax.set_xticks(x, rot)
    ax.set_ylabel("P(permanecer após 10 ps) (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    letra(ax, "b")
    fig.subplots_adjust(wspace=0.3)
    salvar(fig, "E_estabilidade")
    return (f"(a) Energia livre relativa ΔG = −k_B T ln(p_i/p_max) a 500 K, com intervalo de "
            f"95% por bootstrap em {P.BOOT_BLOCOS} blocos contíguos. Os estados 0, 1 e 2 têm "
            f"populações entre {pct(d.pops[:3].min())} e {pct(d.pops[:3].max())} e não se "
            f"distinguem em ΔG; o estado 3 ({pct(d.pops[3])}) fica {br(dg[3])} kcal/mol acima. "
            f"(b) Probabilidade de o estado seguinte (10 ps depois) ser o mesmo.")


@figura("E_medoides", "Estruturas representativas (medoides) dos estados",
        "Resultados › Os estados", questao="Q5", conferir="figs/relatorio/estruturas/")
def fig_medoides(d: Dados) -> str:
    names, coords = d.xyz500
    frames = d.medoides
    ref = coords[frames[0]]
    alinhadas = [kabsch(coords[f], ref, BACKBONE) for f in frames]
    base = eixos_de_vista(ref)
    bonds = ligacoes(ref, names)
    pasta = SAIDA / "estruturas"
    pasta.mkdir(parents=True, exist_ok=True)
    for s, (f, c) in enumerate(zip(frames, alinhadas)):
        with open(pasta / f"estado_{s}_medoide.xyz", "w", encoding="utf-8") as fh:
            fh.write(f"{len(names)}\nestado {s} ({d.nomes[s]}), frame {f} de trajectory.xyz\n")
            for n, (x, y, z) in zip(names, c):
                fh.write(f"{n:<4s} {x:12.5f} {y:12.5f} {z:12.5f}\n")
    fig, axes = plt.subplots(2, 2, figsize=(INTEIRA, 9.5 * CM))
    for s, ax in enumerate(axes.flat):
        desenhar_molecula(ax, alinhadas[s], names, bonds, base, rotulos=(s == 0))
        titulo_com_marca(ax, f"estado {s}  {d.nomes[s]}  ({pct(d.pops[s])})", CORES_ESTADO[s])
    alcas = [Line2D([], [], marker="o", ls="", markersize=5, markerfacecolor=CPK[e],
                    markeredgecolor=TINTA, markeredgewidth=0.4, label=e) for e in "CNOH"]
    fig.legend(handles=alcas, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01),
               handletextpad=0.1, columnspacing=0.8)
    fig.subplots_adjust(wspace=0.05, hspace=0.15)
    salvar(fig, "E_medoides")
    return ("Medoide de cada estado — o frame com a menor soma de distâncias de corda aos "
            "demais membros —, superposto ao do estado 0 pelos átomos do esqueleto e visto na "
            f"mesma orientação (frames {', '.join(map(str, frames))} da trajetória de 500 K). "
            "Mesmo com todos os resíduos na região estendida (estado 3), a cadeia se dobra: é "
            "a ligação *cis* entre Ala3 e Ala4 que a vira. As coordenadas estão em "
            "`figs/relatorio/estruturas/` para renderização em VMD, VESTA ou PyMOL.")


# --------------------------------------------------------------------------
# C — critério cinético
# --------------------------------------------------------------------------
@figura("C_residencia", "Tempos de residência nos estados",
        "Resultados › Critério cinético", tipo="apêndice", conferir="notebook 05 §2.1")
def fig_residencia(d: Dados) -> str:
    dw = kin.dwell_times(d.lab5, P.K_ESTADOS)
    fig, ax = plt.subplots(figsize=(MEIA * 1.15, 6 * CM))
    for i, v in enumerate(dw):
        t = np.arange(1, v.max() + 2)
        ax.step(t * P.DT_PS, [(v >= ti).mean() for ti in t], where="post", color=CORES_ESTADO[i],
                lw=1.2, label=f"{i}  {d.nomes[i]}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlabel="duração da visita t (ps)", ylabel="fração das visitas ≥ t")
    ax.legend(title="estado", loc="lower left")
    salvar(fig, "C_residencia")
    med = [v.mean() * P.DT_PS for v in dw]
    return ("Fração das visitas a cada estado que duram pelo menos t (primeira e última visita "
            "da trajetória descartadas). As durações médias vão de "
            f"{br(min(med), 0)} a {br(max(med), 0)} ps; só o estado 2, o único com resíduos em "
            "α_L, tem visitas longas (cauda até ~1 ns).")


@figura("C_its", "Escalas de tempo implícitas: K = 4 contra microestados",
        "Resultados › Critério cinético", questao="Q7", conferir="notebook 05 §2.2 e §3")
def fig_its(d: Dados) -> str:
    lags = np.array(P.LAGS)
    paineis = [("K = 4 (geométrico)", d.its_k4, None)]
    paineis += [(f"{km} microestados", d.its_micro[km], d.boot_micro[km]) for km in P.K_MICRO]
    fig, axes = plt.subplots(1, 3, figsize=(INTEIRA, 5.8 * CM), sharey=True)
    for ax, (titulo, its, boot), lt in zip(axes, paineis, "abc"):
        ax.fill_between(lags, 0.5, lags, color=CINZA, alpha=0.18, lw=0)
        ax.plot(lags, lags, color=CINZA, lw=0.6)
        for p in range(min(5, its.shape[1]) - 1, -1, -1):
            ax.plot(lags, its[:, p], "o-", ms=2.8, lw=1.1,
                    color=CORES_ESTADO[0] if p == 0 else CINZA, zorder=3 if p == 0 else 2)
        if boot is not None:
            j = list(lags).index(P.LAG_REF)
            for p in range(2):
                lo, hi = np.nanpercentile(boot[:, p], [2.5, 97.5])
                t = its[j, p]
                ax.errorbar(P.LAG_REF * 1.07, t, yerr=[[t - lo], [hi - t]], fmt="none",
                            capsize=2.5, color=TINTA, lw=0.9, zorder=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.5, 100)
        ax.set_xlabel("tempo de atraso τ (frames)")
        ax.set_title(titulo)
        letra(ax, lt)
    axes[0].set_ylabel("escala de tempo implícita (frames)")
    fig.subplots_adjust(wspace=0.1)
    salvar(fig, "C_its")
    km = max(P.K_MICRO)
    j = list(lags).index(P.LAG_REF)
    b = d.boot_micro[km]
    lo1, hi1 = np.nanpercentile(b[:, 0], [2.5, 97.5])
    lo2, hi2 = np.nanpercentile(b[:, 1], [2.5, 97.5])
    t1, t2 = d.its_micro[km][j, :2]
    return ("Escalas de tempo implícitas t_i(τ) = −τ / ln λ_i(τ) em função do tempo de atraso "
            "(1 frame = 10 ps); em azul, o processo mais lento; na região cinza, t < τ, os "
            "processos não são resolvidos. (a) Com os 4 estados geométricos a escala mais lenta "
            f"sobe de {br(d.its_k4[0, 0], 1)} para {br(d.its_k4[-1, 0], 1)} frames, sem patamar: "
            "a partição não é markoviana. (b, c) Com microestados, um processo fica claramente "
            f"separado dos demais: com {km} microestados e τ = {P.LAG_REF} frames, "
            f"t₁ = {br(t1 * P.DT_PS / 1000, 2)} ns (IC 95%: {br(lo1 * P.DT_PS / 1000, 2)}–"
            f"{br(hi1 * P.DT_PS / 1000, 2)} ns) e t₂ = {br(t2 * P.DT_PS / 1000, 2)} ns "
            f"({br(lo2 * P.DT_PS / 1000, 2)}–{br(hi2 * P.DT_PS / 1000, 2)} ns). Barras: IC 95% "
            f"por bootstrap em {P.BOOT_BLOCOS} blocos.")


@figura("C_macroestados_alfaL", "O processo lento: entrada e saída de α_L",
        "Resultados › Critério cinético", questao="Q7", conferir="notebook 05 §4")
def fig_macro(d: Dados) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11 * CM, 5.8 * CM), sharey=True)
    cor = {0: CINZA, 1: CORES_ESTADO[2]}
    for ax, res, lt in zip(axes, (2, 5), "ab"):
        phi, psi = d.X5[:, 2 * (res - 1)], d.X5[:, 2 * (res - 1) + 1]
        for m in (0, 1):
            sel = d.macro == m
            ax.scatter(phi[sel], psi[sel], s=3, color=cor[m], alpha=0.5, lw=0, rasterized=True,
                       label=f"macroestado {m}")
        ax.axvline(0, color=TINTA, lw=0.5)
        eixo_rama(ax, rotulo_y=res == 2)
        ax.set_title(f"Ala{res}")
        letra(ax, lt)
    axes[0].legend(loc="lower left", markerscale=3, handletextpad=0.2)
    fig.subplots_adjust(wspace=0.08)
    salvar(fig, "C_macroestados_alfaL")
    fl = [(d.X5[d.macro == m][:, 0::2] > 0).mean(0) for m in (0, 1)]
    return (f"Divisão dos {max(P.K_MICRO)} microestados em dois macroestados pelo sinal do "
            "autovetor do processo mais lento, vista no Ramachandran de (a) Ala2 e (b) Ala5. O "
            f"macroestado 1 coincide com 'algum resíduo em α_L (φ > 0)' em "
            f"{pct(np.mean(d.macro == d.tem_L), 1)} dos frames: no macroestado 1, Ala2 está em "
            f"α_L em {pct(fl[1][1])} e Ala5 em {pct(fl[1][4])} dos frames, contra "
            f"{pct(fl[0][1])} e {pct(fl[0][4])} no macroestado 0.")


@figura("C_rotulos_tempo", "Sequência temporal dos estados, 500 K e 300 K",
        "Resultados › Comparação 500 K × 300 K", questao="Q7",
        conferir="notebook 05 §5.2")
def fig_rotulos(d: Dados) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(INTEIRA, 6.4 * CM))
    for ax, lab, nms, temp in ((axes[0], d.lab5, d.nomes, "500 K"),
                               (axes[1], d.can3.labels, d.nomes3, "300 K")):
        t = np.arange(len(lab)) * P.DT_PS / 1000
        for s in range(P.K_ESTADOS):
            sel = lab == s
            cor = CORES_ESTADO[s] if temp == "500 K" else TINTA
            ax.scatter(t[sel], np.full(sel.sum(), s), marker="|", s=40, color=cor, lw=0.7,
                       rasterized=True)
        ax.set_yticks(range(P.K_ESTADOS), [f"{s}  {n}" for s, n in enumerate(nms)], fontsize=7)
        ax.set_ylim(P.K_ESTADOS - 0.5, -0.5)
        ax.set_xlim(-0.1, t.max() + 0.1)
        ax.set_title(f"{temp} — estados próprios desta temperatura", loc="left")
        ax.grid(axis="y", visible=False)
    axes[1].set_xlabel("tempo (ns)")
    fig.subplots_adjust(hspace=0.55)
    salvar(fig, "C_rotulos_tempo")
    ala1 = ramachandran_region(d.X3[:, 0], d.X3[:, 1]) == "α"
    frames = np.flatnonzero(ala1)
    return ("Estado em cada frame. A 500 K (acima, cores dos estados de referência), o estado 2 "
            "— o único com α_L — aparece em blocos longos. A 300 K (abaixo, em preto: estes "
            "estados são outros, agrupados na própria trajetória de 300 K), os estados com Ala1 "
            f"em α_R só aparecem entre {br(frames[0] * P.DT_PS / 1000, 1)} e "
            f"{br(frames[-1] * P.DT_PS / 1000, 1)} ns: um único evento lento em 10 ns, o que "
            "impede estimar populações de equilíbrio a 300 K. O KNN treinado a 500 K põe "
            f"{pct(d.knn300.max(), 1)} dos frames de 300 K num único estado de 500 K.")


# --------------------------------------------------------------------------
# Catálogo
# --------------------------------------------------------------------------
ORDEM_SECOES = ["S", "A", "D", "B", "E", "C"]
NOME_SECAO = {"S": "Sistema", "A": "Descritores globais (Fase A)", "D": "Ângulos diedros",
              "B": "K-means periódico (contribuição)", "E": "Os estados", "C": "Critério cinético"}

TABELAS = """\
## Tabelas sugeridas

Os números vêm impressos nos notebooks; copie de lá.

| tabela | onde |
|---|---|
| ω das seis ligações peptídicas, 500 K e 300 K | notebook 05 §0 |
| ocupação das regiões de Ramachandran por resíduo | notebook 05 §0 |
| estados de referência: nome, população, ΔG com IC 95%, P(permanecer) | notebook 05 §1.1 |
| convergência por variante e K (ciclo, violações, iterações) | notebook 02, "Comparação das três variantes" |
| ARI entre métodos | notebook 02, "Baselines" |
| escalas de tempo implícitas com IC 95% (20 e 40 microestados) | notebook 05 §3 |
| populações a 300 K pelo KNN (k = 1, 5, 15) | notebook 05 §5.1 |
| estados próprios de 300 K: visitas, primeiro e último frame | notebook 05 §5.2 |

## Mapa das questões do exercício

| questão | figuras |
|---|---|
| Q1 — por que não usar as coordenadas cruas | S_estrutura (argumento conceitual) |
| Q2 — o que se observa em Rg e RMSD | A_rg_rmsd |
| Q3 — rodar o K-means várias vezes | B_sementes |
| Q4 — um bom valor de K | A_kmeans_rg_rmsd, B_cotovelo_corda |
| Q5 — as estruturas de um cluster se parecem? | E_ramachandran_estados, E_medoides |
| Q6 — medida de convergência | B_convergencia, B_historico_objetivo |
| Q7 — melhores features e número de estados | C_its, C_macroestados_alfaL, B_ari_metodos, E_estabilidade, C_rotulos_tempo |
"""


def escrever_catalogo(legendas: dict[str, str]) -> Path:
    linhas = [
        "# Figuras do relatório",
        "",
        "Gerado por `python -m molsim.figuras` — não edite à mão; mude o código e rode de "
        "novo. Cada figura existe em PDF (vetorial, para LaTeX) e PNG 300 dpi (para Word), já "
        "no tamanho de impressão: 16 cm de largura as de página inteira, 8 cm as de meia "
        "coluna. As legendas são **sugestões**, com os números calculados na mesma execução "
        "que desenhou a figura.",
        "",
        "Em LaTeX: `\\includegraphics{figs/relatorio/NOME.pdf}` sem `width=` (a figura já tem "
        "o tamanho final).",
        "",
    ]
    for sec in ORDEM_SECOES:
        figs = [f for f in FIGURAS if f.nome.startswith(sec + "_")]
        if not figs:
            continue
        linhas += [f"## {sec} — {NOME_SECAO[sec]}", ""]
        for f in figs:
            linhas += [
                f"### `{f.nome}` — {f.titulo}",
                "",
                f"![{f.nome}]({f.nome}.png)",
                "",
                f"- **uso:** {f.tipo} · **onde:** {f.secao}"
                + (f" · **questão:** {f.questao}" if f.questao else ""),
                f"- **conferir números em:** {f.conferir}" if f.conferir else "",
                "",
                f"**Legenda sugerida.** {legendas[f.nome]}",
                "",
            ]
    linhas.append(TABELAS)
    caminho = SAIDA / "LEGENDAS.md"
    caminho.write_text("\n".join(l for l in linhas if l is not None), encoding="utf-8")
    return caminho


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", help="gerar só estas figuras (pelo nome)")
    args = ap.parse_args()

    dados = Dados()
    alvo = FIGURAS if not args.only else [f for f in FIGURAS if f.nome in args.only]
    if args.only and len(alvo) != len(args.only):
        conhecidas = ", ".join(f.nome for f in FIGURAS)
        raise SystemExit(f"figura desconhecida; as disponíveis são: {conhecidas}")

    legendas = {}
    for f in alvo:
        legendas[f.nome] = f.gerar(dados)
        print(f"  {f.nome}")
    if not args.only:
        print(f"-> {escrever_catalogo(legendas)}")
    else:
        for nome, leg in legendas.items():
            print(f"\n{nome}: {leg}")


if __name__ == "__main__":
    main()
