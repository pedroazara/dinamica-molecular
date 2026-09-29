"""Leitura da trajetória e cálculo dos descritores estruturais (Fases A-C).

Sistema: Ace-(Ala)6-NH2 (hexâmero de alanina), 42 sítios, modelo de átomo
unido. Cada frame foi relaxado a T=0K por minimização antes de ser gravado
(PLANO.md §1.1).

Descritores: raio de giração, RMSD (relativo ao primeiro frame) e os 12
ângulos diedros de backbone (phi/psi de cada um dos 6 resíduos), na mesma
ordem e com os mesmos índices usados nos notebooks 01 e 02.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Índices dos 12 diedros phi/psi na lista `u.dihedrals` que o `to_guess` do
# MDAnalysis constrói para esta topologia — mesma ordem do PLANO.md §1.1:
# (phi1, psi1, phi2, psi2, ..., phi6, psi6).
DIHEDRAL_INDICES = [3, 7, 15, 19, 27, 31, 39, 43, 51, 55, 63, 67]
DIHEDRAL_NAMES = [f"dihedral{i}" for i in DIHEDRAL_INDICES]


def load_features(traj: Path | str) -> pd.DataFrame:
    """Carrega `traj` (.xyz) e devolve um DataFrame com `time`, `radgyr`,
    `rmsd` e os 12 diedros — as mesmas colunas de `df_features` nos
    notebooks 01 e 02. Requer MDAnalysis (dependência opcional, ver
    PLANO.md §6.1)."""
    import MDAnalysis as mda
    from MDAnalysis.analysis import rms

    guess = ["bonds", "angles", "dihedrals", "masses"]
    u = mda.Universe(str(traj), to_guess=guess, dt=10)

    molecule = u.select_atoms("all")
    nframes = len(u.trajectory)
    radgyr = np.zeros(nframes)
    for ts in u.trajectory:
        radgyr[ts.frame] = molecule.radius_of_gyration()

    R = rms.RMSD(u, u, select="all", ref_frame=0)
    R.run()
    rmsd = R.results.rmsd[:, 2]

    rows = []
    for ts in u.trajectory:
        row = {"time": ts.time, "radgyr": radgyr[ts.frame], "rmsd": rmsd[ts.frame]}
        for name, idx in zip(DIHEDRAL_NAMES, DIHEDRAL_INDICES):
            row[name] = u.dihedrals[idx].value()
        rows.append(row)

    return pd.DataFrame(rows)
