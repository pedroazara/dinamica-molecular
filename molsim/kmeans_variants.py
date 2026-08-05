"""Três variantes de K-means para dados angulares, e a coerência de cada uma.

O algoritmo de Lloyd converge porque **os dois passos reduzem o mesmo
objetivo**: a atribuição reduz ao mover cada ponto para o centro mais próximo,
e a atualização reduz porque o novo centroide é o *minimizador* da soma das
distâncias dos membros. Se a regra de atualização não for o minimizador da
métrica usada na atribuição, essa garantia se perde.

Ângulos diedros são periódicos, o que dá três combinações possíveis:

    variante    atribuição            atualização        coerente?
    ---------------------------------------------------------------
    tutorial    imagem mínima (L=360) média aritmética   não
    mista       imagem mínima (L=360) média circular     parcialmente
    corda       1 - cos(Δθ)           média circular     SIM

A variante `tutorial` é a receita da §7.8 do exercício MolSim. A média
aritmética de ângulos é indefinida na descontinuidade ±180°: a média de −170°
e +170° é 0°, quando o ponto médio circular é 180°.

A variante `corda` é a única em que a atualização minimiza exatamente a métrica
da atribuição — a média circular μ = atan2(⟨sin θ⟩, ⟨cos θ⟩) é o minimizador de
Σ(1 − cos(θ − μ)). Equivale a rodar K-means euclidiano sobre a imersão
(cos θ, sin θ).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BOX = 360.0


# --------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------
def minimum_image(delta: np.ndarray, box: float = BOX) -> np.ndarray:
    """Convenção de imagem mínima: leva a diferença para (−box/2, box/2]."""
    return delta - box * np.round(delta / box)


def dist2_minimum_image(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """(n, k) distâncias ao quadrado com imagem mínima. Graus."""
    return (minimum_image(X[:, None, :] - C[None, :, :]) ** 2).sum(-1)


def dist_chord(R: np.ndarray, C: np.ndarray) -> np.ndarray:
    """(n, k) distâncias de corda Σ(1 − cos Δθ). Radianos."""
    return (1.0 - np.cos(R[:, None, :] - C[None, :, :])).sum(-1)


def circular_mean(rad: np.ndarray, axis: int = 0) -> np.ndarray:
    """μ = atan2(⟨sin θ⟩, ⟨cos θ⟩) — minimizador de Σ(1 − cos(θ − μ))."""
    return np.arctan2(np.sin(rad).mean(axis), np.cos(rad).mean(axis))


# --------------------------------------------------------------------------
@dataclass
class KMeansResult:
    centers: np.ndarray          # graus
    labels: np.ndarray
    n_iter: int
    converged: bool
    cycled: bool                 # entrou em ciclo limite (não converge nunca)
    objective: np.ndarray        # histórico do objetivo da própria variante
    inertia: float


def kmeans_angular(X_deg: np.ndarray, k: int, variant: str = "corda",
                   seed: int | None = 0, max_iter: int = 500) -> KMeansResult:
    """K-means para features angulares em graus.

    `variant` ∈ {"tutorial", "mista", "corda"} — ver docstring do módulo.
    Detecta ciclos limite guardando as partições já visitadas: se uma partição
    se repete, o algoritmo está preso e nunca vai convergir.
    """
    if variant not in ("tutorial", "mista", "corda"):
        raise ValueError(f"variante desconhecida: {variant}")

    rng = np.random.default_rng(seed)
    start = rng.choice(len(X_deg), k, replace=False)
    R = np.radians(X_deg)

    C = (R[start] if variant == "corda" else X_deg[start].astype(float)).copy()
    labels = np.full(len(X_deg), -1)
    seen: set[bytes] = set()
    history: list[float] = []
    converged = cycled = False
    it = 0

    for it in range(max_iter):
        D = dist_chord(R, C) if variant == "corda" else dist2_minimum_image(X_deg, C)
        new = D.argmin(1)

        if (new == labels).all():
            converged = True
            break
        labels = new

        key = labels.tobytes()
        if key in seen:
            cycled = True
            break
        seen.add(key)

        for j in range(k):
            sel = labels == j
            if not sel.any():
                continue
            if variant == "tutorial":
                C[j] = X_deg[sel].mean(0)
            elif variant == "mista":
                C[j] = np.degrees(circular_mean(R[sel]))
            else:
                C[j] = circular_mean(R[sel])

        D = dist_chord(R, C) if variant == "corda" else dist2_minimum_image(X_deg, C)
        history.append(float(D[np.arange(len(X_deg)), labels].sum()))

    centers = np.degrees(C) if variant == "corda" else C
    inertia = float(dist2_minimum_image(X_deg, centers)[
        np.arange(len(X_deg)), labels].sum())

    return KMeansResult(centers, labels, it, converged, cycled,
                        np.array(history), inertia)


def monotonicity_violations(history: np.ndarray, rtol: float = 1e-9) -> int:
    """Quantas vezes o objetivo SUBIU — deve ser 0 num Lloyd coerente."""
    if len(history) < 2:
        return 0
    tol = rtol * max(1.0, abs(float(history[0])))
    return int((np.diff(history) > tol).sum())
