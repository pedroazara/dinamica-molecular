"""K-means periódico: as três variantes registradas na §3 do PLANO.md.

O algoritmo de Lloyd converge porque atribuição e atualização reduzem o
MESMO objetivo: a atribuição move cada ponto para o centro mais próximo, e a
atualização recoloca o centro no minimizador exato da soma das distâncias dos
seus membros. A §7.8 do MolSim corrige a atribuição dos ângulos diedros com
distância de imagem mínima (L=360°), mas mantém a atualização como "the mean
of the positions of its cluster members" — média aritmética. As duas escolhas
são incompatíveis: a média aritmética não é o minimizador da distância de
imagem mínima na descontinuidade ±180°.

Três variantes (PLANO.md §3.2):

    variante    atribuição              atualização        coerente
    tutorial    imagem mínima (L=360)   média aritmética    não
    mista       imagem mínima (L=360)   média circular      parcialmente
    corda       soma(1 - cos Δθ)        média circular      sim

Só em `corda` a atualização é o minimizador exato da métrica de atribuição:
μ = atan2(⟨sin θ⟩, ⟨cos θ⟩) minimiza Σ(1 − cos(θ − μ)). Equivale a K-means
euclidiano sobre a imersão (cos θ, sin θ).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

VARIANTS = ("tutorial", "mista", "corda")


# --------------------------------------------------------------------------
# Distâncias e médias
# --------------------------------------------------------------------------
def min_image_delta(x1, x2, lbox: float = 360.0):
    """Diferença x1 - x2 corrigida por imagem mínima numa caixa de lado `lbox`."""
    d = np.asarray(x1) - np.asarray(x2)
    return d - lbox * np.round(d / lbox)


def min_image_sq(x1, x2, lbox: float = 360.0):
    """Distância euclidiana ao quadrado, com correção de imagem mínima."""
    d = min_image_delta(x1, x2, lbox)
    return np.sum(d * d, axis=-1)


def chord_sq(x1, x2, lbox: float = 360.0):
    """Σ(1 - cos Δθ): a distância cujo minimizador exato é a média circular."""
    factor = 2 * np.pi / lbox
    dtheta = (np.asarray(x1) - np.asarray(x2)) * factor
    return np.sum(1.0 - np.cos(dtheta), axis=-1)


def arithmetic_mean(x, lbox: float = 360.0):
    del lbox  # mantém a assinatura igual à de circular_mean, p/ despacho uniforme
    return np.mean(x, axis=0)


def circular_mean(x, lbox: float = 360.0):
    """μ = atan2(⟨sin θ⟩, ⟨cos θ⟩): minimizador exato de chord_sq, não de min_image_sq."""
    factor = 2 * np.pi / lbox
    theta = np.asarray(x) * factor
    s = np.mean(np.sin(theta), axis=0)
    c = np.mean(np.cos(theta), axis=0)
    return np.arctan2(s, c) / factor


_ASSIGN = {"tutorial": min_image_sq, "mista": min_image_sq, "corda": chord_sq}
_UPDATE = {"tutorial": arithmetic_mean, "mista": circular_mean, "corda": circular_mean}


# --------------------------------------------------------------------------
# K-means com instrumentação
# --------------------------------------------------------------------------
@dataclass
class KMeansResult:
    variant: str
    k: int
    labels: np.ndarray
    centers: np.ndarray
    n_members: np.ndarray
    objective_history: list[float] = field(repr=False)
    n_iter: int
    converged: bool
    cycled: bool
    seed: int

    @property
    def monotonic(self) -> bool:
        """Verdadeiro se o objetivo nunca subiu entre iterações consecutivas.

        Só é garantido por construção quando atribuição e atualização são
        coerentes (variante `corda`); nas outras duas é uma medida empírica.
        """
        h = self.objective_history
        return all(h[i + 1] <= h[i] + 1e-9 for i in range(len(h) - 1))


def kmeans_periodic(features, k: int, variant: str = "corda", lbox: float = 360.0,
                     max_iter: int = 100, seed: int = 0) -> KMeansResult:
    """K-means sobre features periódicas (ângulos em graus, período `lbox`).

    Instrumenta o que a §3.3 do PLANO.md pede medir: histórico do objetivo
    (para checar monotonicidade), número de iterações, e detecção de ciclo
    limite guardando as partições (tuplas de rótulos) já visitadas — uma
    partição repetida significa que o algoritmo está preso, não convergindo.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variante desconhecida: {variant!r}, use uma de {VARIANTS}")

    features = np.asarray(features, dtype=float)
    n = len(features)
    assign_dist = _ASSIGN[variant]
    update = _UPDATE[variant]

    rng = np.random.default_rng(seed)
    centers = features[rng.choice(n, size=k, replace=False)].copy()

    labels = np.full(n, -1, dtype=int)  # sentinela: nenhum rótulo real é -1
    objective_history: list[float] = []
    seen_partitions: set[tuple[int, ...]] = set()
    converged = False
    cycled = False
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        dists = np.stack([assign_dist(features, centers[ic], lbox) for ic in range(k)], axis=1)
        new_labels = dists.argmin(axis=1)
        objective_history.append(float(dists[np.arange(n), new_labels].sum()))

        if np.array_equal(new_labels, labels):
            labels = new_labels
            converged = True
            break

        partition = tuple(new_labels.tolist())
        if partition in seen_partitions:
            labels = new_labels
            cycled = True
            break
        seen_partitions.add(partition)
        labels = new_labels

        new_centers = centers.copy()
        for ic in range(k):
            members = labels == ic
            if members.any():
                new_centers[ic] = update(features[members], lbox)
        centers = new_centers

    n_members = np.array([(labels == ic).sum() for ic in range(k)])
    return KMeansResult(variant=variant, k=k, labels=labels, centers=centers,
                         n_members=n_members, objective_history=objective_history,
                         n_iter=n_iter, converged=converged, cycled=cycled, seed=seed)


# --------------------------------------------------------------------------
# Programa de medida (§3.3)
# --------------------------------------------------------------------------
@dataclass
class ConvergenceStats:
    variant: str
    k: int
    n_init: int
    rate_converged: float
    rate_cycled: float
    rate_maxed_out: float
    monotonic_violations: int
    mean_n_iter: float
    results: list[KMeansResult] = field(repr=False)


def convergence_program(features, k: int, variant: str, n_init: int = 50,
                         lbox: float = 360.0, max_iter: int = 100,
                         seed0: int = 0) -> ConvergenceStats:
    """Roda `n_init` inicializações e tabula taxa de convergência, ciclo limite
    e violações de monotonicidade — o programa de medida da §3.3 do PLANO.md.
    """
    results = [kmeans_periodic(features, k, variant, lbox, max_iter, seed=seed0 + i)
               for i in range(n_init)]
    n = len(results)
    n_converged = sum(r.converged for r in results)
    n_cycled = sum(r.cycled for r in results)
    return ConvergenceStats(
        variant=variant, k=k, n_init=n,
        rate_converged=n_converged / n,
        rate_cycled=n_cycled / n,
        rate_maxed_out=(n - n_converged - n_cycled) / n,
        monotonic_violations=sum(not r.monotonic for r in results),
        mean_n_iter=sum(r.n_iter for r in results) / n,
        results=results,
    )


def discontinuity_fraction(angles, threshold_deg: float = 40.0) -> float:
    """Fração de amostras a menos de `threshold_deg` da descontinuidade ±180°.

    Assume ângulos já no intervalo (-180, 180], convenção usual do MDAnalysis
    para diedros. Aceita qualquer shape (achata antes de medir).
    """
    angles = np.asarray(angles, dtype=float).ravel()
    dist_to_edge = 180.0 - np.abs(angles)
    return float(np.mean(dist_to_edge < threshold_deg))
