"""Testes de `molsim.data` contra as trajetórias reais.

A leitura é numpy puro; o MDAnalysis só entra no teste de conferência
(`test_bate_com_o_mdanalysis_em_500k`), que pula quando ele não está instalado.

Executar com:  pytest -q
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molsim.data import (
    DIHEDRAL_ATOMS, DIHEDRAL_INDICES, DIHEDRAL_NAMES, dihedrals, element_masses,
    load_features, ramachandran_region, read_xyz, state_name,
)
from molsim.kmeans_variants import discontinuity_fraction

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
TRAJ_500K = NOTEBOOKS / "trajectory.xyz"
TRAJ_300K = NOTEBOOKS / "trajectory_300K.xyz"


@pytest.fixture(scope="module")
def features_500k():
    return load_features(TRAJ_500K)


@pytest.fixture(scope="module")
def features_300k():
    return load_features(TRAJ_300K)


def test_load_features_tem_as_colunas_esperadas(features_500k):
    assert list(features_500k.columns) == ["time", "radgyr", "rmsd", *DIHEDRAL_NAMES]


def test_load_features_cobre_a_trajetoria_inteira(features_500k, features_300k):
    """20 ns e 10 ns a 10 ps/frame (PLANO.md §1.1)."""
    assert len(features_500k) == 2000
    assert features_500k["time"].iloc[-1] == pytest.approx(19990.0)
    assert len(features_300k) == 1000
    assert features_300k["time"].iloc[-1] == pytest.approx(9990.0)


def test_diedros_estao_no_intervalo_convencional(features_500k):
    valores = features_500k[DIHEDRAL_NAMES].values
    assert valores.min() >= -180.0
    assert valores.max() <= 180.0


def test_fracao_perto_da_descontinuidade_bate_com_o_plano(features_500k):
    """PLANO.md §3.1: ~11,6% das amostras a menos de 40° de ±180°."""
    frac = discontinuity_fraction(features_500k[DIHEDRAL_NAMES].values, threshold_deg=40.0)
    assert frac == pytest.approx(0.1165, abs=0.001)


# --------------------------------------------------------------------------
# Quádruplas e massas: o que torna o arquivo de 300 K legível
# --------------------------------------------------------------------------
def test_quadruplas_sao_phi_e_psi_do_backbone():
    """phi = C-N-CA-C e psi = N-CA-C-N, lidos dos nomes do arquivo de 500 K."""
    names, _ = read_xyz(TRAJ_500K)
    for j, quad in enumerate(DIHEDRAL_ATOMS):
        elementos = "".join(names[a] for a in quad)
        assert elementos == ("CNCC" if j % 2 == 0 else "NCCN")


def test_ordem_dos_atomos_e_a_mesma_nas_duas_temperaturas():
    n500, _ = read_xyz(TRAJ_500K)
    n300, _ = read_xyz(TRAJ_300K)
    assert [n[0] for n in n300] == n500


def test_massas_nao_zeram_com_nomes_gromos():
    """O MDAnalysis dá massa 0 a `CH3`, `CH1`, `NH1`; aqui as massas das duas
    temperaturas saem idênticas, porque a ordem dos átomos é a mesma."""
    n500, _ = read_xyz(TRAJ_500K)
    n300, _ = read_xyz(TRAJ_300K)
    np.testing.assert_array_equal(element_masses(n300), element_masses(n500))
    assert element_masses(n300).min() > 0


def test_sinal_do_diedro_segue_a_convencao_iupac():
    """Quádrupla sintética: olhando do átomo 1 para o 2 (ao longo de +z), o
    átomo 3 está 60° no sentido horário em relação ao 0 — diedro +60°. A
    imagem especular dá -60°, e é esse sinal que separa α_R de α_L."""
    ang = np.radians(60.0)
    p = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1], [np.cos(ang), np.sin(ang), 1]],
                 dtype=float)
    espelho = p * [1, -1, 1]
    quad = [(0, 1, 2, 3)]
    assert dihedrals(p[None], quad)[0, 0] == pytest.approx(60.0)
    assert dihedrals(espelho[None], quad)[0, 0] == pytest.approx(-60.0)


def test_bate_com_o_mdanalysis_em_500k(features_500k):
    """Conferência contra uma implementação independente: os diedros de
    `u.dihedrals[DIHEDRAL_INDICES]`, o raio de giração e o RMSD do MDAnalysis."""
    mda = pytest.importorskip("MDAnalysis")
    from MDAnalysis.analysis import rms

    u = mda.Universe(str(TRAJ_500K), to_guess=["bonds", "angles", "dihedrals", "masses"],
                     dt=10)
    rg = np.array([u.atoms.radius_of_gyration() for _ in u.trajectory])
    rmsd = rms.RMSD(u, u, select="all", ref_frame=0).run().results.rmsd[:, 2]
    dih = np.array([[u.dihedrals[i].value() for i in DIHEDRAL_INDICES]
                    for _ in u.trajectory])

    np.testing.assert_allclose(features_500k["radgyr"], rg, atol=1e-5)
    np.testing.assert_allclose(features_500k["rmsd"], rmsd, atol=1e-5)
    delta = (features_500k[DIHEDRAL_NAMES].values - dih + 180) % 360 - 180
    assert np.abs(delta).max() < 1e-3


# --------------------------------------------------------------------------
# Nomeação estrutural
# --------------------------------------------------------------------------
def test_regioes_de_ramachandran():
    phi = [-57, -120, -75, 60, -160]
    psi = [-47, 130, 145, 45, -170]  # o último é β do outro lado de ψ = ±180°
    assert list(ramachandran_region(phi, psi)) == ["α", "β", "P", "L", "β"]


def test_nome_do_estado_le_um_residuo_por_vez():
    helice = np.tile([-57.0, -47.0], 6)
    assert state_name(helice) == "αααααα"
    misto = helice.copy()
    misto[10:12] = [-120.0, 130.0]
    assert state_name(misto) == "αααααβ"
