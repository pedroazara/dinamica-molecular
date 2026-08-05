"""Mede a coerência das três variantes de K-means nos dados reais do exercício.

Reproduz a tabela do relatório: taxa de convergência, ciclos limite e violações
de monotonicidade do objetivo, para k = 4..12, com 20 inicializações cada.

    python -m molsim.experiment_coherence
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data import dihedrals, read_xyz
from .kmeans_variants import kmeans_angular, monotonicity_violations

DEFAULT_TRAJ = Path(r"G:/Meu Drive/ufla/ic/exercises/trajectory.xyz")
COERENCIA = {"tutorial": "não", "mista": "parcial", "corda": "SIM"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj", type=Path, default=DEFAULT_TRAJ)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--k", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    args = ap.parse_args()

    coords, _ = read_xyz(args.traj)
    X, names = dihedrals(coords)
    print(f"{args.traj.name}: {len(coords)} frames × {coords.shape[1]} sítios")
    print(f"features: {X.shape[1]} diedros ({', '.join(names)})\n")

    borda = (np.abs(np.abs(X) - 180) < 40).mean()
    print(f"amostras a menos de 40° da descontinuidade ±180°: {borda*100:.1f}%\n")

    head = (f"{'k':>3} | {'variante':<9} | {'coerente':>8} | {'convergiu':>9} | "
            f"{'ciclou':>6} | {'obj subiu':>10} | {'inércia mediana':>15}")
    print("=" * len(head)); print(head); print("=" * len(head))

    for k in args.k:
        for v in ("tutorial", "mista", "corda"):
            conv = cyc = nm = 0
            inertias = []
            for s in range(args.seeds):
                r = kmeans_angular(X, k, variant=v, seed=s)
                conv += r.converged
                cyc += r.cycled
                nm += monotonicity_violations(r.objective) > 0
                inertias.append(r.inertia)
            print(f"{k:3d} | {v:<9} | {COERENCIA[v]:>8} | {conv:5d}/{args.seeds:<3d} | "
                  f"{cyc:4d}   | {nm:6d}/{args.seeds:<3d} | {np.median(inertias):15.4e}")
        print("-" * len(head))

    print("\nExemplo mínimo do problema da média aritmética:")
    ang = np.array([-170.0, 175.0, 170.0, -175.0])
    arit = ang.mean()
    circ = np.degrees(np.arctan2(np.sin(np.radians(ang)).mean(),
                                 np.cos(np.radians(ang)).mean()))
    from .kmeans_variants import minimum_image
    print(f"   ângulos            : {ang}")
    print(f"   média aritmética   : {arit:8.2f}°   Σd² = "
          f"{(minimum_image(ang - arit) ** 2).sum():10.1f}")
    print(f"   média circular     : {circ:8.2f}°   Σd² = "
          f"{(minimum_image(ang - circ) ** 2).sum():10.1f}")


if __name__ == "__main__":
    main()
