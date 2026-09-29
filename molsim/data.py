"""Leitura da trajetória e cálculo dos descritores estruturais (Fases A-C).

Sistema: Ace-(Ala)6-NH2 (hexâmero de alanina), 42 sítios, modelo de átomo
unido. Cada frame foi relaxado a T=0K por minimização antes de ser gravado
(PLANO.md §1.1).

Descritores: raio de giração, RMSD (relativo ao primeiro frame) e os 12
ângulos diedros de backbone (phi/psi de cada um dos 6 resíduos), na mesma
ordem e com os mesmos nomes de coluna usados nos notebooks 01 e 02.

Tudo em numpy, sem MDAnalysis. Isso não é só conveniência: o arquivo de 300 K
usa nomes de tipo GROMOS (`CH3`, `CH1`, `NH1`), e o MDAnalysis atribui massa
ZERO a esses átomos — o raio de giração ponderado pela massa sai errado sem
aviso. Aqui os diedros vêm de quádruplas de átomos explícitas e as massas do
elemento (primeira letra do nome), e a ordem dos átomos é a mesma nos dois
arquivos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Nomes de coluna herdados dos notebooks 01 e 02: o número é a posição do
# diedro na lista `u.dihedrals` que o `to_guess` do MDAnalysis constrói.
DIHEDRAL_INDICES = [3, 7, 15, 19, 27, 31, 39, 43, 51, 55, 63, 67]
DIHEDRAL_NAMES = [f"dihedral{i}" for i in DIHEDRAL_INDICES]

# Os mesmos 12 diedros como quádruplas de átomos, a partir da ordem regular do
# arquivo (PLANO.md §1.1): 1 é o C da acetila, e o resíduo i (i = 1..6) começa
# em 3 + 6(i-1) com N, H, CA, CB, C, O; 39 é o N da amida C-terminal.
#   phi_i = C(i-1) - N(i) - CA(i) - C(i)
#   psi_i = N(i)   - CA(i) - C(i) - N(i+1)
def _backbone_quadruples() -> list[tuple[int, int, int, int]]:
    quads = []
    for i in range(6):
        n = 3 + 6 * i
        ca, c = n + 2, n + 4
        c_prev = 1 if i == 0 else n - 2
        n_next = n + 6
        quads.append((c_prev, n, ca, c))    # phi
        quads.append((n, ca, c, n_next))    # psi
    return quads


DIHEDRAL_ATOMS = _backbone_quadruples()
PHI_PSI_NAMES = [f"{ang}{i}" for i in range(1, 7) for ang in ("phi", "psi")]

MASSES = {"C": 12.011, "N": 14.007, "O": 15.999, "H": 1.008}


def read_xyz(path: Path | str) -> tuple[list[str], np.ndarray]:
    """Lê um .xyz multi-frame. Devolve (nomes, coords[n_frames, n_atoms, 3])."""
    lines = Path(path).read_text().splitlines()
    n_atoms = int(lines[0])
    block = n_atoms + 2
    n_frames = len(lines) // block
    names = [lines[2 + a].split()[0] for a in range(n_atoms)]
    coords = np.empty((n_frames, n_atoms, 3))
    for f in range(n_frames):
        body = lines[f * block + 2:(f + 1) * block]
        coords[f] = [[float(v) for v in ln.split()[1:4]] for ln in body]
    return names, coords


def element_masses(names: list[str]) -> np.ndarray:
    """Massa pelo elemento, lido da primeira letra do nome (`CH3` -> C)."""
    return np.array([MASSES[name[0]] for name in names])


def dihedrals(coords: np.ndarray, quads=DIHEDRAL_ATOMS) -> np.ndarray:
    """Diedros em graus, (-180, 180], convenção IUPAC. coords: (n_frames, n_atoms, 3)."""
    q = np.asarray(quads)
    p0, p1, p2, p3 = (coords[:, q[:, j]] for j in range(4))
    b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    x = np.sum(n1 * n2, axis=-1)
    y = np.linalg.norm(b2, axis=-1) * np.sum(b1 * n2, axis=-1)
    return np.degrees(np.arctan2(y, x))


def radius_of_gyration(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Raio de giração ponderado pela massa, um valor por frame."""
    w = masses / masses.sum()
    com = np.einsum("a,fak->fk", w, coords)
    sq = np.sum((coords - com[:, None, :]) ** 2, axis=-1)
    return np.sqrt(sq @ w)


def rmsd_to_reference(coords: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """RMSD de cada frame a `ref` após superposição ótima (Kabsch, pesos iguais)."""
    x = coords - coords.mean(axis=1, keepdims=True)
    y = ref - ref.mean(axis=0)
    h = np.einsum("fai,aj->fij", x, y)
    u, s, vt = np.linalg.svd(h)
    # correção de reflexão: det(R) = +1
    sign = np.sign(np.linalg.det(u @ vt))
    s[:, -1] *= sign
    e0 = np.sum(x * x, axis=(1, 2)) + np.sum(y * y)
    msd = np.maximum(e0 - 2 * s.sum(axis=1), 0.0) / coords.shape[1]
    return np.sqrt(msd)


def load_features(traj: Path | str, dt_ps: float = 10.0) -> pd.DataFrame:
    """Carrega `traj` (.xyz) e devolve um DataFrame com `time`, `radgyr`,
    `rmsd` e os 12 diedros — as mesmas colunas de `df_features` nos
    notebooks 01 e 02.

    O RMSD é relativo ao primeiro frame do PRÓPRIO arquivo; entre temperaturas
    a referência é outra estrutura, então o RMSD não é comparável entre os
    arquivos de 500 K e 300 K. Raio de giração e diedros são.
    """
    names, coords = read_xyz(traj)
    data = {
        "time": np.arange(len(coords)) * dt_ps,
        "radgyr": radius_of_gyration(coords, element_masses(names)),
        "rmsd": rmsd_to_reference(coords, coords[0]),
    }
    data.update(zip(DIHEDRAL_NAMES, dihedrals(coords).T))
    return pd.DataFrame(data)


def ramachandran_region(phi, psi) -> np.ndarray:
    """Região de Ramachandran grosseira: 'α' (α_R), 'β', 'P' (PPII) ou 'L' (α_L).

    φ > 0 é α_L; com φ ≤ 0, a faixa -120° < ψ ≤ 50° é α_R, e o resto é a região
    estendida, dividida em β (φ < -100°) e PPII (φ ≥ -100°).
    """
    phi = np.asarray(phi)
    psi = np.asarray(psi)
    estendida = np.where(phi < -100, "β", "P")
    return np.where(phi > 0, "L", np.where((psi > -120) & (psi <= 50), "α", estendida))


def state_name(center: np.ndarray) -> str:
    """Nome estrutural de um estado a partir do seu centroide (12 diedros,
    ordem phi1, psi1, ..., phi6, psi6): uma letra por resíduo, do N ao C."""
    center = np.asarray(center)
    return "".join(ramachandran_region(center[0::2], center[1::2]))
