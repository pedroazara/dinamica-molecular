"""Testes de `molsim.data.load_features` contra a trajetória real.

MDAnalysis é dependência opcional para a análise (PLANO.md §6.1) — os testes
aqui pulam, em vez de falhar, quando ela não está instalada no interpretador
usado para rodar pytest.

Executar com:  pytest -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("MDAnalysis")

from molsim.data import DIHEDRAL_NAMES, load_features
from molsim.kmeans_variants import discontinuity_fraction

TRAJ_500K = Path(__file__).resolve().parents[1] / "notebooks" / "trajectory.xyz"


@pytest.fixture(scope="module")
def features_500k():
    return load_features(TRAJ_500K)


def test_load_features_tem_as_colunas_esperadas(features_500k):
    assert list(features_500k.columns) == ["time", "radgyr", "rmsd", *DIHEDRAL_NAMES]


def test_load_features_cobre_a_trajetoria_inteira(features_500k):
    """20 ns a 10 ps/frame = 2000 frames (PLANO.md §1.1)."""
    assert len(features_500k) == 2000
    assert features_500k["time"].iloc[0] == 0
    assert features_500k["time"].iloc[-1] == pytest.approx(19990.0)


def test_diedros_estao_no_intervalo_convencional(features_500k):
    valores = features_500k[DIHEDRAL_NAMES].values
    assert valores.min() >= -180.0
    assert valores.max() <= 180.0


def test_fracao_perto_da_descontinuidade_bate_com_o_plano(features_500k):
    """PLANO.md §3.1: ~11,6% das amostras a menos de 40° de ±180°."""
    frac = discontinuity_fraction(features_500k[DIHEDRAL_NAMES].values, threshold_deg=40.0)
    assert frac == pytest.approx(0.1165, abs=0.001)
