"""Critério cinético de estado (Fase C, PLANO.md §4).

Um agrupamento geométrico diz quais estruturas se parecem; só a dinâmica diz
quais regiões a molécula ocupa por tempo longo comparado ao tempo de trânsito.
Tudo aqui opera sobre a sequência temporal de rótulos (um por frame) e mede o
tempo em FRAMES — a conversão para ps é multiplicar pelo intervalo de gravação
(10 ps nas trajetórias do exercício).

    count_matrix         transições i -> j a um tempo de atraso τ
    transition_matrix    probabilidade condicional, com balanço detalhado
    implied_timescales   t_i(τ) = -τ / ln λ_i(τ); patamar em τ = estados markovianos
    bootstrap_timescales incerteza das escalas, sem transições falsas entre blocos
    spectral_split       dois macroestados pelo processo mais lento
    dwell_times          duração de cada visita ininterrupta a um estado
    free_energy          ΔG_i = -k_B T ln(p_i / p_max)
    block_bootstrap      incerteza respeitando a correlação temporal
"""

from __future__ import annotations

import numpy as np

KB_KCAL = 0.0019872041  # kcal/(mol K)


def count_matrix(labels, lag: int = 1, n_states: int | None = None) -> np.ndarray:
    """C[i, j] = número de pares (t, t+lag) com rótulo i em t e j em t+lag
    (janela deslizante: usa todos os pares, não só t = 0, lag, 2·lag, ...)."""
    labels = np.asarray(labels, dtype=int)
    n = int(labels.max()) + 1 if n_states is None else n_states
    c = np.zeros((n, n))
    np.add.at(c, (labels[:-lag], labels[lag:]), 1.0)
    return c


def transition_matrix(counts, reversible: bool = True) -> np.ndarray:
    """T[i, j] = P(j em t+τ | i em t). Com `reversible`, simetriza as contagens,
    (C + Cᵀ)/2, o que impõe balanço detalhado e deixa os autovalores reais.
    Linhas sem contagem (estado nunca visitado) ficam zeradas."""
    c = np.asarray(counts, dtype=float)
    if reversible:
        c = 0.5 * (c + c.T)
    rows = c.sum(axis=1, keepdims=True)
    return np.divide(c, rows, out=np.zeros_like(c), where=rows > 0)


def _reversible_eigenvalues(counts) -> np.ndarray:
    """Autovalores de T = D⁻¹ C_sim, em ordem decrescente, calculados pela forma
    simétrica D^(-1/2) C_sim D^(-1/2) (mesmo espectro, numericamente exata)."""
    c = 0.5 * (counts + counts.T)
    visited = c.sum(axis=1) > 0
    c = c[np.ix_(visited, visited)]
    d = 1.0 / np.sqrt(c.sum(axis=1))
    return np.sort(np.linalg.eigvalsh(d[:, None] * c * d[None, :]))[::-1]


def implied_timescales(labels, lags, n_states: int | None = None) -> np.ndarray:
    """Escalas de tempo implícitas t_i(τ) = -τ / ln λ_i(τ), em frames.

    Devolve (len(lags), n-1): uma linha por τ, os processos do mais lento ao
    mais rápido (o autovalor 1, estacionário, é descartado). λ ≤ 0 vira NaN —
    o processo já decaiu dentro de τ e não é resolvido.

    Se a dinâmica entre os estados é markoviana, t_i não depende de τ. Curvas
    que sobem com τ indicam estados mal definidos (memória dentro do estado);
    o número de processos lentos separados dos demais por um salto é o número
    de estados metaestáveis menos um.
    """
    labels = np.asarray(labels, dtype=int)
    n = int(labels.max()) + 1 if n_states is None else n_states
    out = np.full((len(lags), n - 1), np.nan)
    for row, lag in enumerate(lags):
        ev = _reversible_eigenvalues(count_matrix(labels, lag, n))[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            ts = np.where(ev > 0, -lag / np.log(ev), np.nan)
        out[row, :len(ts)] = ts
    return out


def bootstrap_timescales(labels, lag: int, n_states: int | None = None,
                         n_blocks: int = 10, n_boot: int = 500,
                         seed: int = 0) -> np.ndarray:
    """Distribuição bootstrap das escalas de tempo implícitas a um τ fixo.

    Divide a trajetória em `n_blocks` blocos contíguos, conta as transições
    DENTRO de cada bloco, e reamostra os blocos com reposição somando as
    matrizes de contagem. Concatenar os rótulos dos blocos, em vez disso,
    inventaria transições nas emendas (lag x n_blocks pares falsos) e
    puxaria as escalas para baixo. Devolve (n_boot, n-1), em frames.
    """
    labels = np.asarray(labels, dtype=int)
    n = int(labels.max()) + 1 if n_states is None else n_states
    per_block = np.stack([count_matrix(b, lag, n) for b in np.array_split(labels, n_blocks)])
    rng = np.random.default_rng(seed)
    out = np.full((n_boot, n - 1), np.nan)
    for b in range(n_boot):
        c = per_block[rng.integers(0, n_blocks, n_blocks)].sum(axis=0)
        ev = _reversible_eigenvalues(c)[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            ts = np.where(ev > 0, -lag / np.log(ev), np.nan)
        out[b, :len(ts)] = ts
    return out


def spectral_split(labels, lag: int, n_states: int | None = None) -> np.ndarray:
    """Divide os microestados em dois macroestados pelo sinal do segundo
    autovetor à direita de T(τ) — o processo mais lento. Devolve, para cada
    microestado, 0 ou 1 (o macroestado 0 é o mais populado).

    É a versão de dois estados do PCCA: a fronteira cai onde a dinâmica é mais
    lenta, não onde a geometria muda mais.
    """
    labels = np.asarray(labels, dtype=int)
    n = int(labels.max()) + 1 if n_states is None else n_states
    c = count_matrix(labels, lag, n)
    c = 0.5 * (c + c.T)
    visited = c.sum(axis=1) > 0
    cv = c[np.ix_(visited, visited)]
    d = 1.0 / np.sqrt(cv.sum(axis=1))
    w, u = np.linalg.eigh(d[:, None] * cv * d[None, :])
    right = d * u[:, np.argsort(w)[-2]]  # autovetor à direita de T = D⁻¹ C
    macro = np.zeros(n, dtype=int)
    macro[np.flatnonzero(visited)] = (right > 0).astype(int)
    pops = np.bincount(macro[labels], minlength=2)
    return macro if pops[0] >= pops[1] else 1 - macro


def dwell_times(labels, n_states: int | None = None,
                drop_censored: bool = True) -> list[np.ndarray]:
    """Duração, em frames, de cada visita ininterrupta a cada estado.

    Com `drop_censored`, descarta a primeira e a última visita da trajetória:
    elas foram cortadas pelo início/fim da simulação e subestimam a duração.
    """
    labels = np.asarray(labels, dtype=int)
    n = int(labels.max()) + 1 if n_states is None else n_states
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate([[0], change])
    lengths = np.diff(np.concatenate([starts, [len(labels)]]))
    states = labels[starts]
    if drop_censored:
        starts, lengths, states = starts[1:-1], lengths[1:-1], states[1:-1]
    return [lengths[states == s] for s in range(n)]


def populations(labels, n_states: int | None = None) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    n = int(labels.max()) + 1 if n_states is None else n_states
    return np.bincount(labels, minlength=n) / len(labels)


def free_energy(pops, temperature: float) -> np.ndarray:
    """ΔG_i = -k_B T ln(p_i / p_max), em kcal/mol; o estado mais populado é 0."""
    pops = np.asarray(pops, dtype=float)
    with np.errstate(divide="ignore"):
        return -KB_KCAL * temperature * np.log(pops / pops.max())


def block_bootstrap(labels, statistic, n_blocks: int = 10, n_boot: int = 1000,
                    seed: int = 0) -> np.ndarray:
    """Reamostra a trajetória em `n_blocks` blocos contíguos, com reposição, e
    devolve `statistic(labels_reamostrados)` para cada uma das `n_boot` réplicas.

    Frames vizinhos são correlacionados; reamostrar frames individuais
    subestimaria a incerteza. Blocos longos comparados ao tempo de residência
    preservam essa correlação. Serve para estatísticas de frame único
    (populações, ΔG); para escalas de tempo use `bootstrap_timescales`.
    """
    labels = np.asarray(labels, dtype=int)
    blocks = np.array_split(labels, n_blocks)
    rng = np.random.default_rng(seed)
    return np.array([
        statistic(np.concatenate([blocks[i] for i in rng.integers(0, n_blocks, n_blocks)]))
        for _ in range(n_boot)
    ])
