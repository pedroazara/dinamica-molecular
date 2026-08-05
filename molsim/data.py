"""Leitura das trajetórias do exercício MolSim e cálculo dos descritores.

O exercício original usa MDAnalysis com `to_guess=['bonds','angles','dihedrals']`
para descobrir a conectividade a partir das distâncias. Aqui os índices são
explícitos: a ordem dos átomos nos dois arquivos `.xyz` é regular e conhecida,
o que dispensa a etapa de adivinhação e remove uma fonte silenciosa de erro
(a lista de diedros adivinhados depende de raios de van der Waals e muda de
versão para versão do MDAnalysis).

Sistema: Ace-(Ala)6-NH2, modelo de átomo unido, 42 sítios.

    índice  0: CH3   (metila da acetila)
    índice  1: C     (carbonila da acetila)
    índice  2: O
    resíduo i (i = 0..5), bloco começando em 3 + 6i:
            N, H, CA, CB, C, O
    índice 39: N      (amida C-terminal)
    índices 40, 41: H, H

Trajetórias disponíveis:
    trajectory.xyz       500 K, 2000 frames, nomes por elemento (C/H/N/O)
    trajectory_300K.xyz  300 K, 1000 frames, nomes de tipo GROMOS (CH3/CH1/NH1…)

Ambas foram gravadas a cada 10 ps e cada frame foi relaxado a T = 0 K por
minimização de energia antes de ser escrito.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

N_RESIDUES = 6
N_SITES = 42

# Massas do modelo de átomo unido (g/mol). `element_masses=True` usa massas
# atômicas puras, reproduzindo o que o MDAnalysis infere de trajectory.xyz.
UNITED_MASSES = {
    "CH3": 15.035, "CH1": 13.019, "CH2": 14.027,
    "C": 12.011, "O": 15.999, "N": 14.007,
    "NH1": 15.015, "NH2": 16.023, "H": 1.008,
}
ELEMENT_OF = {"CH3": "C", "CH1": "C", "CH2": "C", "NH1": "N", "NH2": "N"}
ATOMIC_MASSES = {"C": 12.011, "N": 14.007, "O": 15.999, "H": 1.008}


# --------------------------------------------------------------------------
# Leitura
# --------------------------------------------------------------------------
def read_xyz(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Devolve (coords (n_frames, n_atoms, 3), nomes dos átomos)."""
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].split()[0])
    stride = n + 2
    n_frames = len(lines) // stride

    names = [lines[2 + i].split()[0] for i in range(n)]
    coords = np.empty((n_frames, n, 3), dtype=float)
    for f in range(n_frames):
        base = f * stride + 2
        for i in range(n):
            coords[f, i] = [float(v) for v in lines[base + i].split()[1:4]]
    return coords, names


def masses(names: list[str], element_masses: bool = True) -> np.ndarray:
    if element_masses:
        return np.array([ATOMIC_MASSES[ELEMENT_OF.get(n, n)] for n in names])
    return np.array([UNITED_MASSES[n] for n in names])


# --------------------------------------------------------------------------
# Índices dos diedros do esqueleto
# --------------------------------------------------------------------------
def backbone_dihedral_indices() -> tuple[list[tuple[int, int, int, int]], list[str]]:
    """Os 12 diedros (6 φ + 6 ψ) do hexâmero, por índice de átomo."""
    idx, names = [], []
    for i in range(N_RESIDUES):
        prev_c, n, ca, c = 1 + 6 * i, 3 + 6 * i, 5 + 6 * i, 7 + 6 * i
        next_n = 9 + 6 * i if i < N_RESIDUES - 1 else 39
        idx.append((prev_c, n, ca, c)); names.append(f"phi{i + 1}")
        idx.append((n, ca, c, next_n)); names.append(f"psi{i + 1}")
    return idx, names


# --------------------------------------------------------------------------
# Descritores
# --------------------------------------------------------------------------
def dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray,
             p3: np.ndarray) -> np.ndarray:
    """Diedro IUPAC em graus, vetorizado sobre o eixo dos frames."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1).sum(-1, keepdims=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdims=True) * b1
    return np.degrees(np.arctan2((np.cross(b1, v) * w).sum(-1), (v * w).sum(-1)))


def dihedrals(coords: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Matriz (n_frames, 12) com os φ/ψ em graus."""
    idx, names = backbone_dihedral_indices()
    cols = [dihedral(coords[:, a], coords[:, b], coords[:, c], coords[:, d])
            for a, b, c, d in idx]
    return np.column_stack(cols), names


def radius_of_gyration(coords: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Raio de giro ponderado pela massa, por frame."""
    com = (coords * m[None, :, None]).sum(1) / m.sum()
    d2 = ((coords - com[:, None, :]) ** 2).sum(-1)
    return np.sqrt((m[None, :] * d2).sum(1) / m.sum())


def _kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Rotação ótima que alinha P a Q (ambos já centrados)."""
    V, _, Wt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    return V @ D @ Wt


def rmsd_to_reference(coords: np.ndarray, m: np.ndarray,
                      ref_frame: int = 0) -> np.ndarray:
    """RMSD em relação a um frame de referência, após superposição ótima."""
    w = m / m.sum()
    ref = coords[ref_frame] - (coords[ref_frame] * w[:, None]).sum(0)
    out = np.empty(len(coords))
    for f, frame in enumerate(coords):
        P = frame - (frame * w[:, None]).sum(0)
        R = _kabsch(P * w[:, None], ref * w[:, None])
        out[f] = np.sqrt((w * ((P @ R - ref) ** 2).sum(-1)).sum())
    return out


def feature_table(path: str | Path, dt_ps: float = 10.0,
                  element_masses: bool = True):
    """Devolve um DataFrame com tempo, raio de giro, RMSD e os 12 diedros."""
    import pandas as pd

    coords, names = read_xyz(path)
    m = masses(names, element_masses)
    dih, dih_names = dihedrals(coords)

    df = pd.DataFrame({
        "time": np.arange(len(coords)) * dt_ps,
        "radgyr": radius_of_gyration(coords, m),
        "rmsd": rmsd_to_reference(coords, m),
    })
    for j, n in enumerate(dih_names):
        df[n] = dih[:, j]
    return df
