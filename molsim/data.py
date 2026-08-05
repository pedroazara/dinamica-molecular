"""Leitura das trajetórias do exercício MolSim.  [ESQUELETO — Fase B1]

Objetivo: ler os `.xyz` sem depender do MDAnalysis, para que a lista de diedros
não dependa de raios de van der Waals nem da versão da biblioteca.

Mantenha o MDAnalysis instalado para CONFERIR seus números contra uma
implementação independente. Se as duas concordarem, provavelmente as duas estão
certas.

Ver ROTEIRO.md, tarefa B1.  Alvo: `pytest tests/test_molsim.py -k leitura`
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

N_RESIDUES = 6
N_SITES = 42

# Massas atômicas (g/mol). Os nomes de tipo GROMOS do arquivo de 300 K
# (CH3, CH1, NH1, NH2) precisam ser mapeados para o elemento correspondente.
ATOMIC_MASSES = {"C": 12.011, "N": 14.007, "O": 15.999, "H": 1.008}


def read_xyz(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Lê um arquivo `.xyz` de múltiplos frames.

    O formato: cada frame começa com uma linha contendo o número de átomos,
    seguida de uma linha de comentário, seguida de uma linha por átomo no
    formato `nome x y z`.

    Returns
    -------
    coords : np.ndarray, shape (n_frames, n_atoms, 3)
    names  : list[str], os nomes dos átomos (iguais em todos os frames)
    """
    raise NotImplementedError("Fase B1")


def element_of(name: str) -> str:
    """Elemento correspondente a um nome de átomo.

    Precisa funcionar tanto para símbolos de elemento (`trajectory.xyz`) quanto
    para nomes de tipo GROMOS (`trajectory_300K.xyz`: CH3, CH1, NH1, NH2).
    """
    raise NotImplementedError("Fase B1")


def masses(names: list[str]) -> np.ndarray:
    """Vetor de massas, uma por átomo, na ordem de `names`."""
    raise NotImplementedError("Fase B1")


def backbone_dihedral_indices() -> tuple[list[tuple[int, int, int, int]], list[str]]:
    """Os 12 diedros do esqueleto (6 φ + 6 ψ), por índice de átomo.

    Um diedro é definido por quatro átomos ligados em cadeia:

        φ do resíduo i:  C(i−1) — N(i)  — CA(i) — C(i)
        ψ do resíduo i:  N(i)   — CA(i) — C(i)  — N(i+1)

    Para o resíduo 1, o "C anterior" é a carbonila da acetila. Para o resíduo 6,
    o "N seguinte" é o nitrogênio da amida C-terminal.

    O mapa de índices dos átomos está na referência rápida do ROTEIRO.md —
    confira contra o arquivo antes de confiar.

    Returns
    -------
    indices : lista de 12 tuplas (i, j, k, l)
    names   : lista de 12 rótulos, ex. ["phi1", "psi1", "phi2", ...]
    """
    raise NotImplementedError("Fase B1")
