"""Descritores estruturais.  [ESQUELETO — Fase B1]

Os três descritores que o plano de trabalho pede:

    raio de giro  — compactação global
    RMSD          — desvio estrutural em relação a uma referência
    diedros φ/ψ   — conformação local do esqueleto

Ver ROTEIRO.md, tarefas A2, A3 e B1.
Alvo: `pytest tests/test_molsim.py -k descritor`
"""

from __future__ import annotations

import numpy as np


def dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray,
             p3: np.ndarray) -> np.ndarray:
    """Ângulo diedro p0–p1–p2–p3, em graus, na convenção IUPAC.

    Deve aceitar entradas de shape (3,) ou (n_frames, 3) e devolver escalar ou
    (n_frames,) conforme o caso.

    O valor está em (−180, 180]. **O sinal importa**: ele codifica a
    quiralidade. Uma implementação com o sinal invertido produz um mapa de
    Ramachandran espelhado — todas as distâncias e ângulos de valência
    continuam corretos, e o erro passa despercebido até alguém comparar as
    bacias com a literatura.

    O teste `test_dihedral_convencao_iupac` fixa a convenção com um caso
    calculável à mão. Faça ele passar antes de seguir.
    """
    raise NotImplementedError("Fase A3 / B1")


def dihedrals(coords: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Os 12 diedros do esqueleto para todos os frames.

    Returns
    -------
    X     : np.ndarray, shape (n_frames, 12), em graus
    names : list[str], os rótulos das colunas
    """
    raise NotImplementedError("Fase A3 / B1")


def radius_of_gyration(coords: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Raio de giro ponderado pela massa, por frame. Shape (n_frames,).

    Mede a dispersão da massa em torno do centro de massa. É invariante a
    translação e a rotação.
    """
    raise NotImplementedError("Fase A2 / B1")


def rmsd_to_reference(coords: np.ndarray, m: np.ndarray,
                      ref_frame: int = 0) -> np.ndarray:
    """RMSD de cada frame contra um frame de referência, após superposição.

    A superposição ótima (remoção de translação e rotação de corpo rígido) é
    obrigatória: sem ela o RMSD mede o movimento da molécula pela caixa, não a
    mudança de conformação.

    O algoritmo padrão para a rotação ótima é o de Kabsch, via decomposição em
    valores singulares. Atenção ao caso de determinante negativo, que
    corresponde a uma reflexão — rotação própria não reflete a molécula.

    Alternativa: use `MDAnalysis.analysis.rms.RMSD` na Fase A e implemente na
    mão só na Fase B, comparando os dois.

    Returns
    -------
    np.ndarray, shape (n_frames,).  O valor em `ref_frame` deve ser exatamente 0.
    """
    raise NotImplementedError("Fase A2 / B1")


def feature_table(path, dt_ps: float = 10.0):
    """DataFrame com `time`, `radgyr`, `rmsd` e os 12 diedros.

    É a tabela que a §7.4 do tutorial pede.
    """
    raise NotImplementedError("Fase A3 / B1")
