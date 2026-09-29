"""Testes de `molsim.kinetics` e dos estados canônicos de `molsim.kmeans_variants`.

O teste de referência é uma cadeia de Markov de dois estados, cuja escala de
tempo implícita tem forma fechada: com P(0→1) = a e P(1→0) = b, o segundo
autovalor é λ = 1 - a - b e t = -1/ln λ frames, para qualquer τ.

Executar com:  pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest

from molsim.kinetics import (
    KB_KCAL, block_bootstrap, bootstrap_timescales, count_matrix, dwell_times,
    free_energy, implied_timescales, populations, spectral_split, transition_matrix,
)
from molsim.kmeans_variants import (
    best_of, canonical_states, kmeans_periodic, match_states, sort_by_population,
)


def markov_chain(tmat, n_steps, seed=0):
    rng = np.random.default_rng(seed)
    cum = np.cumsum(tmat, axis=1)
    u = rng.random(n_steps)
    s = np.empty(n_steps, dtype=int)
    s[0] = 0
    for t in range(1, n_steps):
        s[t] = np.searchsorted(cum[s[t - 1]], u[t])
    return s


A, B = 0.02, 0.05
T2 = np.array([[1 - A, A], [B, 1 - B]])
T_EXATO = -1.0 / np.log(1 - A - B)  # ~13.8 frames


@pytest.fixture(scope="module")
def cadeia():
    return markov_chain(T2, 200_000, seed=1)


# --------------------------------------------------------------------------
# Contagem e matriz de transição
# --------------------------------------------------------------------------
def test_contagem_a_mao():
    labels = [0, 0, 1, 1, 1, 0]
    np.testing.assert_array_equal(count_matrix(labels, lag=1), [[1, 1], [1, 2]])
    np.testing.assert_array_equal(count_matrix(labels, lag=2), [[0, 2], [1, 1]])


def test_matriz_de_transicao_recupera_a_cadeia(cadeia):
    t = transition_matrix(count_matrix(cadeia, lag=1))
    np.testing.assert_allclose(t, T2, atol=3e-3)
    np.testing.assert_allclose(t.sum(axis=1), 1.0)


def test_estado_nao_visitado_nao_gera_nan():
    t = transition_matrix(count_matrix([0, 1, 0, 1], lag=1, n_states=3))
    assert np.isfinite(t).all()
    np.testing.assert_array_equal(t[2], 0.0)


# --------------------------------------------------------------------------
# Escalas de tempo implícitas
# --------------------------------------------------------------------------
def test_escala_de_tempo_bate_com_a_forma_fechada(cadeia):
    its = implied_timescales(cadeia, lags=[1])
    assert its[0, 0] == pytest.approx(T_EXATO, rel=0.05)


def test_escala_de_tempo_nao_depende_do_lag_numa_cadeia_markoviana(cadeia):
    """O patamar em τ é a assinatura de estados markovianos."""
    its = implied_timescales(cadeia, lags=[1, 3, 6, 10])[:, 0]
    np.testing.assert_allclose(its, T_EXATO, rtol=0.08)


def test_so_a_fronteira_lenta_da_patamar():
    """Três microestados: 0 e 1 trocam rápido, 1 e 2 devagar (t ≈ 68 frames).
    Agrupar em dois macroestados pela fronteira LENTA ({0,1} | {2}) preserva a
    propriedade de Markov: patamar no valor exato. Agrupar pela fronteira
    rápida ({0} | {1,2}) a quebra: a escala sobe com τ e não chega a um valor.
    É o critério que decide, na Fase C, se uma partição geométrica é cinética."""
    t3 = np.array([[0.90, 0.10, 0.00],
                   [0.10, 0.89, 0.01],
                   [0.00, 0.01, 0.99]])
    t_lento = -1.0 / np.log(np.sort(np.linalg.eigvals(t3).real)[-2])
    s = markov_chain(t3, 200_000, seed=2)
    lags = [1, 5, 20, 50]

    boa = implied_timescales(np.where(s == 2, 1, 0), lags)[:, 0]
    np.testing.assert_allclose(boa, t_lento, rtol=0.05)

    ruim = implied_timescales(np.where(s == 0, 0, 1), lags)[:, 0]
    assert np.all(np.diff(ruim) > 0)
    assert ruim[-1] > 3 * ruim[0]
    assert ruim[-1] < 0.5 * t_lento


@pytest.fixture(scope="module")
def cadeia3():
    """0 e 1 trocam rápido, 1 e 2 devagar."""
    t3 = np.array([[0.90, 0.10, 0.00],
                   [0.10, 0.89, 0.01],
                   [0.00, 0.01, 0.99]])
    t_lento = -1.0 / np.log(np.sort(np.linalg.eigvals(t3).real)[-2])
    return markov_chain(t3, 20_000, seed=3), t_lento


def test_bootstrap_das_escalas_cobre_o_valor_exato(cadeia3):
    s, t_lento = cadeia3
    amostras = bootstrap_timescales(s, lag=5, n_blocks=20, n_boot=300)[:, 0]
    lo, hi = np.percentile(amostras, [2.5, 97.5])
    assert lo < t_lento < hi
    assert hi - lo < t_lento  # intervalo informativo, não trivial


def test_bootstrap_das_escalas_nao_inventa_transicoes_nas_emendas(cadeia3):
    """Sem reamostrar (todos os blocos uma vez), a soma das contagens por bloco
    perde só os pares que atravessam as emendas — e a escala não cai."""
    s, _ = cadeia3
    inteira = implied_timescales(s, lags=[10])[0, 0]
    por_bloco = bootstrap_timescales(s, lag=10, n_blocks=10, n_boot=300)[:, 0]
    assert np.median(por_bloco) == pytest.approx(inteira, rel=0.1)


def test_divisao_espectral_acha_a_fronteira_lenta(cadeia3):
    s, _ = cadeia3
    np.testing.assert_array_equal(spectral_split(s, lag=5), [0, 0, 1])


def test_lambda_negativo_vira_nan():
    alternando = np.tile([0, 1], 100)
    its = implied_timescales(alternando, lags=[1])
    assert np.isnan(its[0, 0])


# --------------------------------------------------------------------------
# Residência, populações, energia livre, bootstrap
# --------------------------------------------------------------------------
def test_tempos_de_residencia():
    labels = [0, 0, 1, 1, 1, 0, 2, 2, 0]
    todos = dwell_times(labels, drop_censored=False)
    assert [list(d) for d in todos] == [[2, 1, 1], [3], [2]]
    sem_bordas = dwell_times(labels)
    assert [list(d) for d in sem_bordas] == [[1], [3], [2]]


def test_residencia_media_na_cadeia_e_geometrica(cadeia):
    """Numa cadeia de Markov, a visita a 0 dura em média 1/a frames."""
    d0 = dwell_times(cadeia)[0]
    assert d0.mean() == pytest.approx(1 / A, rel=0.05)


def test_energia_livre():
    dg = free_energy([0.5, 0.25, 0.25], temperature=300.0)
    np.testing.assert_allclose(dg, [0, KB_KCAL * 300 * np.log(2), KB_KCAL * 300 * np.log(2)])


def test_bootstrap_centra_na_estimativa(cadeia):
    amostras = block_bootstrap(cadeia[:20_000], lambda l: populations(l, 2)[1],
                               n_blocks=20, n_boot=300)
    assert amostras.mean() == pytest.approx(A / (A + B), abs=0.03)
    assert amostras.std() > 0


# --------------------------------------------------------------------------
# Estados canônicos
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tres_bacias():
    rng = np.random.default_rng(0)
    centros = np.array([[-60.0, -45.0], [-120.0, 130.0], [60.0, 40.0]])
    tamanhos = [300, 150, 50]
    return np.concatenate([c + rng.normal(0, 12, (n, 2)) for c, n in zip(centros, tamanhos)])


def test_sort_by_population_numera_do_maior_para_o_menor(tres_bacias):
    r = sort_by_population(kmeans_periodic(tres_bacias, 3, "corda", seed=5))
    assert list(r.n_members) == sorted(r.n_members, reverse=True)
    assert np.bincount(r.labels).tolist() == list(r.n_members)


def test_best_of_e_deterministico_e_independe_da_ordem_das_sementes(tres_bacias):
    r1 = best_of(tres_bacias, 3, n_init=20, seed0=0)
    r2 = best_of(tres_bacias, 3, n_init=20, seed0=0)
    np.testing.assert_array_equal(r1.labels, r2.labels)
    assert list(r1.n_members) == [300, 150, 50]


def test_canonical_states_e_reprodutivel(tres_bacias):
    r1, r2 = canonical_states(tres_bacias, k=3), canonical_states(tres_bacias, k=3)
    np.testing.assert_array_equal(r1.labels, r2.labels)
    assert r1.variant == "corda"


def test_match_states_recupera_a_permutacao(tres_bacias):
    ref = best_of(tres_bacias, 3, n_init=10)
    perm_verdadeira = np.array([2, 0, 1])
    embaralhado = ref.centers[perm_verdadeira]
    perm = match_states(ref.centers, embaralhado)
    np.testing.assert_array_equal(embaralhado[perm], ref.centers)


def test_match_states_respeita_a_periodicidade():
    """-175° e 178° são vizinhos; 0° está do outro lado do círculo."""
    ref = np.array([[178.0], [0.0]])
    outro = np.array([[5.0], [-175.0]])
    np.testing.assert_array_equal(match_states(ref, outro), [1, 0])
