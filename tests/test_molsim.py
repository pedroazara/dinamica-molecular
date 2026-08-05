"""Especificação executável dos módulos de `molsim/`.

Estes testes definem o ALVO. Eles falham enquanto os esqueletos não estiverem
implementados — é assim que devem começar.

    pytest tests/test_molsim.py -q                 # tudo
    pytest tests/test_molsim.py -k leitura -q      # só a tarefa B1
    pytest tests/test_molsim.py -k descritor -q
    pytest tests/test_molsim.py -k kmeans -q

Os testes verificam PROPRIEDADES, não fórmulas: invariâncias, convenções de
sinal, identidades que qualquer implementação correta satisfaz. Eles não dizem
COMO calcular — dizem o que precisa ser verdade no final.

O helper `assert_is_minimizer` no fim do arquivo é para você usar nos seus
próprios testes da Fase B4.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

TRAJ_500 = Path(r"G:/Meu Drive/ufla/ic/exercises/trajectory.xyz")
TRAJ_300 = Path(r"G:/Meu Drive/ufla/ic/exercises/trajectory_300K.xyz")

precisa_traj = pytest.mark.skipif(
    not TRAJ_500.exists(), reason="trajetória do exercício não encontrada")


# ==========================================================================
# B1 — leitura
# ==========================================================================
@precisa_traj
def test_leitura_shapes():
    from molsim.data import read_xyz
    coords, names = read_xyz(TRAJ_500)
    assert coords.shape == (2000, 42, 3)
    assert len(names) == 42
    assert coords.dtype.kind == "f"


@precisa_traj
def test_leitura_300K():
    from molsim.data import read_xyz
    coords, names = read_xyz(TRAJ_300)
    assert coords.shape == (1000, 42, 3)


@precisa_traj
def test_leitura_ordem_de_atomos_e_a_mesma_nos_dois_arquivos():
    """Mesmo sistema, mesma ordem — só muda a nomenclatura dos átomos."""
    from molsim.data import element_of, read_xyz
    _, n500 = read_xyz(TRAJ_500)
    _, n300 = read_xyz(TRAJ_300)
    assert [element_of(a) for a in n500] == [element_of(b) for b in n300]


@pytest.mark.parametrize("nome, elemento", [
    ("C", "C"), ("N", "N"), ("O", "O"), ("H", "H"),
    ("CH3", "C"), ("CH1", "C"), ("NH1", "N"), ("NH2", "N"),
])
def test_leitura_mapeamento_de_elementos(nome, elemento):
    from molsim.data import element_of
    assert element_of(nome) == elemento


def test_leitura_indices_dos_diedros():
    from molsim.data import backbone_dihedral_indices
    idx, names = backbone_dihedral_indices()
    assert len(idx) == 12 and len(names) == 12
    assert all(len(t) == 4 for t in idx)
    assert all(0 <= a < 42 for t in idx for a in t)
    # φ e ψ do mesmo resíduo compartilham três átomos consecutivos
    for i in range(6):
        phi, psi = idx[2 * i], idx[2 * i + 1]
        assert phi[1:] == psi[:3], f"resíduo {i+1}: φ e ψ não se encaixam"


# ==========================================================================
# A2/A3/B1 — descritores
# ==========================================================================
@pytest.mark.parametrize("d, esperado", [
    ((1.0, 1.0, 0.0), 0.0),
    ((1.0, -1.0, 0.0), 180.0),
    ((1.0, 0.0, 1.0), 90.0),
    ((1.0, 0.0, -1.0), -90.0),
])
def test_descritor_dihedral_convencao_iupac(d, esperado):
    """Fixa a convenção de sinal com um caso calculável à mão.

    Geometria: b na origem, c no eixo +x, a no eixo +y. Olhando de b para c,
    `a` projeta à direita. Um `d` no eixo +z está a 90° no sentido horário
    dessa vista, o que é positivo na convenção IUPAC.

    Se este teste falhar só no SINAL, sua fórmula está espelhada — e um mapa de
    Ramachandran espelhado passa despercebido em qualquer outra verificação.
    """
    from molsim.features import dihedral
    a = np.array([0.0, 1.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([1.0, 0.0, 0.0])
    obtido = float(dihedral(a, b, c, np.array(d)))
    assert obtido == pytest.approx(esperado, abs=1e-6) or \
        (esperado == 180.0 and abs(obtido) == pytest.approx(180.0, abs=1e-6))


def test_descritor_dihedral_e_invariante_a_rotacao():
    rng = np.random.default_rng(0)
    from molsim.features import dihedral
    pts = rng.normal(size=(4, 3))
    antes = float(dihedral(*pts))
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:          # garantir rotação própria
        Q[:, 0] *= -1
    depois = float(dihedral(*(pts @ Q.T + np.array([5.0, -3.0, 2.0]))))
    assert depois == pytest.approx(antes, abs=1e-8)


def test_descritor_dihedral_reflete_o_sinal_sob_espelhamento():
    """Refletir a molécula troca o sinal do diedro. É isso que detecta quiralidade."""
    from molsim.features import dihedral
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(4, 3))
    espelhado = pts * np.array([1.0, 1.0, -1.0])
    assert float(dihedral(*espelhado)) == pytest.approx(-float(dihedral(*pts)),
                                                        abs=1e-8)


def test_descritor_rg_invariante_a_translacao_e_rotacao():
    from molsim.features import radius_of_gyration
    rng = np.random.default_rng(2)
    coords = rng.normal(size=(1, 20, 3)) * 3.0
    m = rng.uniform(1.0, 16.0, size=20)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    movido = coords @ Q.T + np.array([10.0, -7.0, 4.0])
    assert float(radius_of_gyration(movido, m)[0]) == pytest.approx(
        float(radius_of_gyration(coords, m)[0]), rel=1e-9)


def test_descritor_rmsd_contra_si_mesmo_e_zero():
    from molsim.features import rmsd_to_reference
    rng = np.random.default_rng(3)
    coords = rng.normal(size=(5, 20, 3))
    m = np.ones(20)
    assert float(rmsd_to_reference(coords, m, ref_frame=2)[2]) == pytest.approx(
        0.0, abs=1e-9)


def test_descritor_rmsd_ignora_movimento_de_corpo_rigido():
    """Se o RMSD subir quando a molécula só transladou, falta a superposição."""
    from molsim.features import rmsd_to_reference
    rng = np.random.default_rng(4)
    base = rng.normal(size=(20, 3))
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    coords = np.stack([base, base @ Q.T + np.array([50.0, 20.0, -30.0])])
    assert float(rmsd_to_reference(coords, np.ones(20))[1]) == pytest.approx(
        0.0, abs=1e-6)


@precisa_traj
def test_descritor_diedros_na_faixa_e_com_quiralidade_L():
    """Alanina é L-aminoácido: a densidade tem que ficar majoritariamente em φ<0.

    Nuvem espelhada para φ>0 significa índices na ordem errada ou sinal
    invertido em `dihedral`.
    """
    from molsim.data import read_xyz
    from molsim.features import dihedrals
    coords, _ = read_xyz(TRAJ_500)
    X, names = dihedrals(coords)
    assert X.shape == (2000, 12)
    assert X.min() >= -180.0 and X.max() <= 180.0
    phi = X[:, [i for i, n in enumerate(names) if n.startswith("phi")]]
    assert (phi < 0).mean() > 0.75


# ==========================================================================
# B2/B3/B4 — k-means
# ==========================================================================
def test_kmeans_distancia_periodica_caso_do_tutorial():
    """O caso de teste que a §7.8 fornece."""
    from molsim.kmeans_variants import distance
    assert distance(np.array([10.0, 170.0]),
                    np.array([30.0, -170.0]), 360.0) == pytest.approx(28.284,
                                                                      abs=1e-3)


def test_kmeans_distancia_e_simetrica_e_periodica():
    from molsim.kmeans_variants import distance
    rng = np.random.default_rng(5)
    a = rng.uniform(-180, 180, size=12)
    b = rng.uniform(-180, 180, size=12)
    assert distance(a, b, 360.0) == pytest.approx(distance(b, a, 360.0))
    assert distance(a, b + 360.0, 360.0) == pytest.approx(distance(a, b, 360.0))
    assert distance(a, a, 360.0) == pytest.approx(0.0, abs=1e-12)


def test_kmeans_monotonicity_violations():
    from molsim.kmeans_variants import monotonicity_violations
    assert monotonicity_violations(np.array([10.0, 8.0, 5.0, 5.0])) == 0
    assert monotonicity_violations(np.array([10.0, 12.0, 5.0])) == 1
    assert monotonicity_violations(np.array([1.0, 2.0, 3.0])) == 2
    assert monotonicity_violations(np.array([7.0])) == 0


@precisa_traj
def test_kmeans_resultado_bem_formado():
    from molsim.data import read_xyz
    from molsim.features import dihedrals
    from molsim.kmeans_variants import kmeans_angular
    coords, _ = read_xyz(TRAJ_500)
    X, _ = dihedrals(coords)
    r = kmeans_angular(X, k=6, variant="tutorial", seed=0)
    assert r.labels.shape == (len(X),)
    assert set(np.unique(r.labels)) <= set(range(6))
    assert r.centers.shape == (6, 12)
    assert r.converged or r.cycled or r.n_iter > 0
    assert len(r.objective) == r.n_iter or len(r.objective) > 0


# ==========================================================================
# Helper para os SEUS testes da Fase B4
# ==========================================================================
def assert_is_minimizer(candidato, membros, metrica, n_grid: int = 721):
    """Verifica que `candidato` minimiza Σ metrica(m, candidato) sobre `membros`.

    Esta é a propriedade que faz o método de Lloyd convergir. Use nos seus
    testes da B4 para checar cada par (métrica de atribuição, regra de
    atualização) que você implementar:

        assert_is_minimizer(centroid(membros, "sua_regra"),
                            membros,
                            lambda a, b: sua_metrica(a, b))

    A verificação é por força bruta numa grade fina de candidatos, dimensão a
    dimensão — lenta, mas independente da sua fórmula, que é o ponto.
    """
    membros = np.atleast_2d(membros)
    candidato = np.atleast_1d(candidato)
    grade = np.linspace(-180.0, 180.0, n_grid)

    for j in range(membros.shape[1]):
        col = membros[:, j]
        custo_cand = sum(metrica(float(v), float(candidato[j])) for v in col)
        custo_grade = min(sum(metrica(float(v), float(g)) for v in col)
                          for g in grade)
        assert custo_cand <= custo_grade + 1e-6 * max(1.0, abs(custo_grade)), (
            f"dimensão {j}: o centroide proposto custa {custo_cand:.6g}, "
            f"mas existe candidato com {custo_grade:.6g} — "
            f"a regra de atualização não minimiza esta métrica"
        )


def test_helper_assert_is_minimizer_funciona():
    """Sanidade do helper: a média aritmética minimiza a distância euclidiana."""
    membros = np.array([[10.0], [20.0], [30.0]])
    assert_is_minimizer(np.array([20.0]), membros, lambda a, b: (a - b) ** 2)
    with pytest.raises(AssertionError):
        assert_is_minimizer(np.array([5.0]), membros, lambda a, b: (a - b) ** 2)
