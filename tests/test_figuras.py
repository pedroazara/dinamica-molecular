"""Testes de `molsim.figuras`: o gerador roda, e a renderização não espelha
a molécula.

Executar com:  pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest

from molsim import figuras
from molsim.data import read_xyz
from molsim.parametros import TRAJ_500K


def test_nomes_das_figuras_sao_unicos_e_tem_prefixo_de_secao():
    nomes = [f.nome for f in figuras.FIGURAS]
    assert len(nomes) == len(set(nomes))
    assert all(n.split("_")[0] in figuras.ORDEM_SECOES for n in nomes)


def test_virgula_decimal():
    assert figuras.br(0.3456, 2) == "0,35"
    assert figuras.pct(0.965, 1) == "96,5%"


def test_vista_e_uma_base_direita():
    """Uma base esquerda projetaria a imagem especular: L-alanina viraria D."""
    _, coords = read_xyz(TRAJ_500K)
    for f in (0, 500, 1500):
        base = figuras.eixos_de_vista(coords[f])
        assert np.linalg.det(base) == pytest.approx(1.0)


def test_kabsch_recupera_rotacao_conhecida():
    _, coords = read_xyz(TRAJ_500K)
    ref = coords[0]
    ang = np.radians(40)
    rot = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    movel = ref @ rot.T + [3.0, -1.0, 2.0]
    np.testing.assert_allclose(figuras.kabsch(movel, ref, figuras.BACKBONE), ref, atol=1e-8)


def test_ligacoes_formam_uma_arvore():
    """42 sítios sem anel: uma molécula conectada tem exatamente 41 ligações."""
    names, coords = read_xyz(TRAJ_500K)
    assert len(figuras.ligacoes(coords[0], names)) == len(names) - 1


def test_gera_pdf_e_png(tmp_path, monkeypatch):
    monkeypatch.setattr(figuras, "SAIDA", tmp_path)
    fig = next(f for f in figuras.FIGURAS if f.nome == "B_media_circular")
    legenda = fig.gerar(figuras.Dados())
    assert (tmp_path / "B_media_circular.pdf").stat().st_size > 0
    assert (tmp_path / "B_media_circular.png").stat().st_size > 0
    assert "119 050" in legenda
