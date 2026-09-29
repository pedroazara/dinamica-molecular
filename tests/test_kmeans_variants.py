"""Testes do K-means periódico e do programa de medida da §3.3 do PLANO.md.

O teste central (`test_tutorial_cicla_mista_e_corda_convergem`) existe para
operacionalizar em código a hipótese da §3: que a incoerência entre a
atribuição por imagem mínima e a atualização por média aritmética (variante
`tutorial`, a do material de referência) produz ciclo limite e viola a
monotonicidade do objetivo, enquanto as duas correções (`mista`, `corda`) não.
Sem esse teste, a alegação central do projeto fica só em prosa no PLANO.md.

Executar com:  pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest

from molsim.kmeans_variants import (
    VARIANTS, arithmetic_mean, chord_sq, circular_mean, convergence_program,
    discontinuity_fraction, kmeans_periodic, min_image_delta, min_image_sq,
)


# --------------------------------------------------------------------------
# Distâncias e médias — casos numéricos tirados literalmente do PLANO.md
# --------------------------------------------------------------------------
def test_min_image_distance_bate_com_o_caso_do_tutorial():
    """§7.8 do MolSim: distance((10, 170), (30, -170), lbox=360) = 28.284."""
    x1 = np.array([10.0, 170.0])
    x2 = np.array([30.0, -170.0])
    assert np.sqrt(min_image_sq(x1, x2)) == pytest.approx(28.284, abs=1e-3)


def test_media_aritmetica_e_circular_reproduzem_o_exemplo_da_secao_3_1():
    """PLANO.md §3.1: os quatro ângulos [-170, 175, 170, -175] têm média
    aritmética 0° (Σd² = 119050) e média circular 180° (Σd² = 250) —
    a média aritmética erra para o lado *oposto* do círculo."""
    angulos = np.array([-170.0, 175.0, 170.0, -175.0])

    media_ari = arithmetic_mean(angulos)
    media_circ = circular_mean(angulos)
    assert media_ari == pytest.approx(0.0, abs=1e-9)
    assert abs(media_circ) == pytest.approx(180.0, abs=1e-6)

    soma_d2_ari = min_image_sq(angulos, media_ari).sum()
    soma_d2_circ = min_image_sq(angulos, 180.0).sum()
    assert soma_d2_ari == pytest.approx(119050.0, abs=1.0)
    assert soma_d2_circ == pytest.approx(250.0, abs=1.0)
    assert soma_d2_circ < soma_d2_ari


@pytest.mark.parametrize("theta_deg, esperado", [
    (0.0, 0.0),
    (90.0, 1.0),
    (180.0, 2.0),
    (-90.0, 1.0),
])
def test_chord_sq_e_a_metrica_de_corda(theta_deg, esperado):
    """chord_sq(x, x + theta) = 1 - cos(theta); é a métrica cujo minimizador
    exato é a média circular (diferente de min_image_sq, que não é)."""
    x1 = np.array([0.0])
    x2 = np.array([theta_deg])
    assert chord_sq(x1, x2) == pytest.approx(esperado, abs=1e-9)


def test_min_image_delta_lida_com_multiplas_voltas():
    d = min_image_delta(np.array([370.0]), np.array([-10.0]))
    assert d[0] == pytest.approx(20.0, abs=1e-9)


# --------------------------------------------------------------------------
# kmeans_periodic — mecânica básica
# --------------------------------------------------------------------------
def test_variante_desconhecida_levanta_erro():
    with pytest.raises(ValueError):
        kmeans_periodic(np.zeros((10, 1)), k=2, variant="inexistente")


@pytest.mark.parametrize("variant", VARIANTS)
def test_kmeans_periodic_separa_clusters_obvios(variant):
    """Dois grupos bem separados e longe de ±180°: as três variantes devem
    convergir e recuperar os dois grupos, independente de qual seja."""
    rng = np.random.default_rng(1)
    grupo_a = rng.normal(0.0, 3.0, size=(60, 1))
    grupo_b = rng.normal(90.0, 3.0, size=(60, 1))
    features = np.concatenate([grupo_a, grupo_b])

    resultado = kmeans_periodic(features, k=2, variant=variant, seed=0)
    assert resultado.converged
    # os 60 primeiros pontos devem cair todos no mesmo cluster, e os 60
    # últimos no outro (a ordem do rótulo 0/1 não é determinada)
    assert len(set(resultado.labels[:60])) == 1
    assert len(set(resultado.labels[60:])) == 1
    assert resultado.labels[0] != resultado.labels[-1]


def test_objective_history_tem_um_valor_por_iteracao():
    rng = np.random.default_rng(2)
    features = rng.uniform(-180, 180, size=(40, 1))
    resultado = kmeans_periodic(features, k=3, variant="corda", seed=0)
    assert len(resultado.objective_history) == resultado.n_iter


# --------------------------------------------------------------------------
# O resultado central: tutorial cicla e viola monotonicidade; mista e corda não
# --------------------------------------------------------------------------
@pytest.fixture
def angulos_na_descontinuidade():
    """Dois grupos de ângulos centrados perto de ±180°, cada um cruzando a
    descontinuidade — o caso que a §3.1 do PLANO.md argumenta ser típico
    (11,6% dos dados na trajetória de 500 K), não patológico."""
    rng = np.random.default_rng(0)
    grupo_a = rng.normal(178.0, 6.0, 150)
    grupo_a = (grupo_a + 180) % 360 - 180
    grupo_b = rng.normal(-178.0, 6.0, 150)
    grupo_b = (grupo_b + 180) % 360 - 180
    return np.concatenate([grupo_a, grupo_b]).reshape(-1, 1)


def test_tutorial_cicla_mista_e_corda_convergem(angulos_na_descontinuidade):
    stats = {
        variant: convergence_program(angulos_na_descontinuidade, k=2, variant=variant,
                                      n_init=40, seed0=0)
        for variant in VARIANTS
    }

    # tutorial: atribuição por imagem mínima + atualização por média
    # aritmética são incoerentes -> ciclo limite numa fração substancial das
    # inicializações, e o objetivo sobe em algum momento em praticamente
    # todas elas (a média aritmética empurra o centro para o lado errado do
    # círculo, como no exemplo numérico da §3.1)
    assert stats["tutorial"].rate_cycled > 0.3
    assert stats["tutorial"].monotonic_violations > 0

    # mista e corda: atualização por média circular -> sempre convergem e
    # nunca violam monotonicidade nesta configuração
    for variant in ("mista", "corda"):
        assert stats[variant].rate_converged == 1.0
        assert stats[variant].rate_cycled == 0.0
        assert stats[variant].monotonic_violations == 0


def test_corda_e_o_unico_minimizador_exato_e_por_isso_sempre_monotonico():
    """Só em `corda` a atualização é o minimizador exato da métrica de
    atribuição, então o objetivo NUNCA pode subir entre iterações, por
    construção — testar em várias sementes e configurações de k."""
    rng = np.random.default_rng(3)
    features = rng.uniform(-180, 180, size=(80, 2))
    for seed in range(10):
        for k in (2, 3, 5):
            resultado = kmeans_periodic(features, k=k, variant="corda", seed=seed)
            assert resultado.monotonic, f"seed={seed} k={k}"


# --------------------------------------------------------------------------
# Programa de medida — fração de amostras perto da descontinuidade
# --------------------------------------------------------------------------
@pytest.mark.parametrize("angulo, threshold, esperado", [
    (179.0, 40.0, True),
    (-179.0, 40.0, True),
    (150.0, 40.0, True),   # dist. até a borda = 30° < 40°
    (139.0, 40.0, False),  # dist. até a borda = 41° > 40°
    (0.0, 40.0, False),
])
def test_discontinuity_fraction_classifica_um_unico_angulo(angulo, threshold, esperado):
    frac = discontinuity_fraction(np.array([angulo]), threshold_deg=threshold)
    assert frac == (1.0 if esperado else 0.0)


def test_discontinuity_fraction_com_mistura_conhecida():
    perto = np.full(116, 170.0)   # a 10° da borda, dentro do limiar de 40°
    longe = np.full(884, 0.0)     # a 180° da borda, fora do limiar
    angulos = np.concatenate([perto, longe])
    assert discontinuity_fraction(angulos, threshold_deg=40.0) == pytest.approx(0.116, abs=1e-9)
