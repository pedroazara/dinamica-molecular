"""Pipeline completo de análise do hexâmero de alanina.

Cobre as tarefas 1–10 do capítulo 7 do MolSim, com a variante coerente do
K-means como método principal:

    1. descritores (raio de giro, RMSD, 12 diedros)
    2. análise exploratória e mapas de Ramachandran por resíduo
    3. seleção de k (cotovelo + silhueta)
    4. agrupamento nas três variantes, com diagnóstico de convergência
    5. energias livres relativas   ΔG_i = −k_B T ln(n_i/N)
    6. matriz de transições entre estados
    7. estruturas representativas (medoide de cada estado)

    python -m molsim.analyze

A silhueta é calculada sobre a imersão (cos θ, sin θ): a distância euclidiana
nessa imersão é proporcional à distância de corda usada no agrupamento, então
a métrica da seleção de modelo é a mesma do algoritmo — não uma euclidiana
sobre ângulos crus, que reintroduziria o erro de periodicidade.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data import (dihedrals, masses, radius_of_gyration, read_xyz,
                   rmsd_to_reference)
from .kmeans_variants import kmeans_angular, monotonicity_violations

KB_KCAL = 0.0019872041  # kcal/(mol·K)
DEFAULT_TRAJ = Path(r"G:/Meu Drive/ufla/ic/exercises/trajectory.xyz")


# --------------------------------------------------------------------------
def embedding(X_deg: np.ndarray) -> np.ndarray:
    """(n, 2d) imersão (cos θ, sin θ) — lineariza a topologia do toro."""
    r = np.radians(X_deg)
    return np.hstack([np.cos(r), np.sin(r)])


def best_of_n(X: np.ndarray, k: int, variant: str, n_init: int = 10):
    """Melhor de n_init inicializações, pelo objetivo da própria variante."""
    best = None
    for s in range(n_init):
        r = kmeans_angular(X, k, variant=variant, seed=s)
        score = r.objective[-1] if len(r.objective) else np.inf
        if best is None or score < best[0]:
            best = (score, r)
    return best[1]


def free_energies(labels: np.ndarray, k: int, temperature: float):
    n = np.bincount(labels, minlength=k).astype(float)
    p = n / n.sum()
    with np.errstate(divide="ignore"):
        dg = -KB_KCAL * temperature * np.log(p / p.max())
    return n.astype(int), p, dg


def transition_matrix(labels: np.ndarray, k: int) -> np.ndarray:
    """Matriz de transição linha-estocástica, com lag de 1 frame."""
    T = np.zeros((k, k))
    np.add.at(T, (labels[:-1], labels[1:]), 1.0)
    row = T.sum(1, keepdims=True)
    return np.divide(T, row, out=np.zeros_like(T), where=row > 0)


def medoids(X_deg: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> list[int]:
    """Frame real mais próximo de cada centroide (o centroide não é estrutura)."""
    U, C = embedding(X_deg), embedding(centers)
    out = []
    for j in range(len(centers)):
        idx = np.flatnonzero(labels == j)
        if len(idx) == 0:
            out.append(-1); continue
        out.append(int(idx[np.argmin(((U[idx] - C[j]) ** 2).sum(1))]))
    return out


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj", type=Path, default=DEFAULT_TRAJ)
    ap.add_argument("--temperature", type=float, default=500.0)
    ap.add_argument("--kmax", type=int, default=15)
    ap.add_argument("--figs", type=Path,
                    default=Path(__file__).resolve().parents[1] / "figs" / "molsim")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import silhouette_score

    args.figs.mkdir(parents=True, exist_ok=True)

    # ---- 1. descritores -------------------------------------------------
    coords, names = read_xyz(args.traj)
    m = masses(names)
    X, dih_names = dihedrals(coords)
    rg = radius_of_gyration(coords, m)
    rmsd = rmsd_to_reference(coords, m)
    t = np.arange(len(coords)) * 10.0
    U = embedding(X)

    print(f"trajetória : {args.traj.name}")
    print(f"frames     : {len(coords)}  ({t[-1]/1000:.0f} ns a cada 10 ps)")
    print(f"raio de giro: {rg.mean():.3f} ± {rg.std():.3f} Å  "
          f"[{rg.min():.2f}, {rg.max():.2f}]")
    print(f"RMSD        : {rmsd.mean():.3f} ± {rmsd.std():.3f} Å  "
          f"[{rmsd.min():.2f}, {rmsd.max():.2f}]")

    # ---- 2. figuras exploratórias --------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(t / 1000, rg, lw=0.5); ax[0].set(xlabel="t (ns)",
                                                ylabel="raio de giro (Å)")
    ax[1].plot(t / 1000, rmsd, lw=0.5, color="C1")
    ax[1].set(xlabel="t (ns)", ylabel="RMSD ao frame 0 (Å)")
    ax[2].scatter(rg, rmsd, s=4, alpha=0.3, c=t, cmap="viridis")
    ax[2].set(xlabel="raio de giro (Å)", ylabel="RMSD (Å)",
              title="mapa das duas features globais")
    fig.tight_layout(); fig.savefig(args.figs / "01_globais.png", dpi=160)
    plt.close(fig)

    fig, axs = plt.subplots(2, 3, figsize=(13, 8))
    for i, a in enumerate(axs.flat):
        a.scatter(X[:, 2 * i], X[:, 2 * i + 1], s=3, alpha=0.25)
        a.set(xlim=(-180, 180), ylim=(-180, 180), xticks=range(-180, 181, 90),
              yticks=range(-180, 181, 90), xlabel=rf"$\phi_{i+1}$",
              ylabel=rf"$\psi_{i+1}$", title=f"resíduo {i+1}")
        a.axhline(0, lw=.4, c="gray"); a.axvline(0, lw=.4, c="gray")
    fig.suptitle("Ramachandran por resíduo — hexâmero de alanina, 500 K")
    fig.tight_layout(); fig.savefig(args.figs / "02_ramachandran.png", dpi=160)
    plt.close(fig)

    # ---- 3. seleção de k ------------------------------------------------
    ks = list(range(2, args.kmax + 1))
    inertia, sil = [], []
    print(f"\n{'k':>3} | {'objetivo (corda)':>16} | {'silhueta':>9}")
    print("-" * 36)
    for k in ks:
        r = best_of_n(X, k, "corda", n_init=10)
        inertia.append(r.objective[-1])
        s = silhouette_score(U, r.labels)
        sil.append(s)
        print(f"{k:3d} | {inertia[-1]:16.1f} | {s:9.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ks, inertia, "o-"); ax[0].set(xlabel="k",
                                             ylabel="objetivo de corda",
                                             title="cotovelo")
    ax[1].plot(ks, sil, "o-", color="C2")
    ax[1].set(xlabel="k", ylabel="silhueta", title="silhueta (imersão sin/cos)")
    ax[1].axvline(ks[int(np.argmax(sil))], ls="--", c="gray")
    fig.tight_layout(); fig.savefig(args.figs / "03_selecao_k.png", dpi=160)
    plt.close(fig)

    k_best = ks[int(np.argmax(sil))]
    print(f"\nmelhor silhueta em k = {k_best}")

    # ---- 4. comparação das variantes ------------------------------------
    print(f"\ndiagnóstico das três variantes em k = {k_best} (20 sementes):")
    print(f"{'variante':<9} | {'convergiu':>9} | {'ciclou':>6} | {'obj subiu':>9}")
    print("-" * 44)
    for v in ("tutorial", "mista", "corda"):
        c = cy = nm = 0
        for s in range(20):
            r = kmeans_angular(X, k_best, variant=v, seed=s)
            c += r.converged; cy += r.cycled
            nm += monotonicity_violations(r.objective) > 0
        print(f"{v:<9} | {c:5d}/20  | {cy:4d}   | {nm:6d}/20")

    # ---- 5-7. física dos estados ---------------------------------------
    res = best_of_n(X, k_best, "corda", n_init=20)
    n, p, dg = free_energies(res.labels, k_best, args.temperature)
    med = medoids(X, res.labels, res.centers)
    T = transition_matrix(res.labels, k_best)
    dwell = np.where(np.diag(T) < 1, 10.0 / (1.0 - np.diag(T)), np.inf)

    order = np.argsort(-n)
    print(f"\nestados metaestáveis (k = {k_best}, T = {args.temperature:.0f} K):")
    print(f"{'estado':>6} | {'membros':>7} | {'pop':>6} | {'ΔG (kcal/mol)':>13} | "
          f"{'residência (ps)':>15} | {'medoide':>7}")
    print("-" * 72)
    for j in order:
        print(f"{j:6d} | {n[j]:7d} | {p[j]*100:5.1f}% | {dg[j]:13.2f} | "
              f"{dwell[j]:15.0f} | {med[j]:7d}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].bar(range(k_best), dg[order], color="C0")
    ax[0].set(xlabel="estado (ordenado por população)",
              ylabel=r"$\Delta G$ (kcal/mol)", title="energias livres relativas")
    ax[0].set_xticks(range(k_best)); ax[0].set_xticklabels(order)
    im = ax[1].imshow(T[np.ix_(order, order)], cmap="magma", vmin=0, vmax=1)
    ax[1].set(xlabel="estado final", ylabel="estado inicial",
              title="matriz de transição (lag 10 ps)")
    ax[1].set_xticks(range(k_best)); ax[1].set_xticklabels(order)
    ax[1].set_yticks(range(k_best)); ax[1].set_yticklabels(order)
    fig.colorbar(im, ax=ax[1])
    fig.tight_layout(); fig.savefig(args.figs / "04_estados.png", dpi=160)
    plt.close(fig)

    # medoides em .xyz, para inspeção no VMD/VESTA
    out = args.figs.parent.parent / "molsim" / "medoides.xyz"
    with open(out, "w") as fh:
        for j in order:
            fh.write(f"{len(names)}\nestado {j}  frame {med[j]}  "
                     f"pop {p[j]*100:.1f}%  dG {dg[j]:.2f} kcal/mol\n")
            for nm_, xyz in zip(names, coords[med[j]]):
                fh.write(f"{nm_:<5s}{xyz[0]:12.5f}{xyz[1]:12.5f}{xyz[2]:12.5f}\n")

    print(f"\nfiguras  -> {args.figs}")
    print(f"medoides -> {out}")


if __name__ == "__main__":
    main()
