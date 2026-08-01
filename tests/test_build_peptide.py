"""Testes de regressão da construção geométrica dos peptídeos.

O teste de helicidade (`test_alpha_helix_is_right_handed`) existe porque a
primeira versão de `dihedral()` tinha o sinal invertido em relação a `place()`,
o que produzia silenciosamente estruturas espelhadas — hélices canhotas e
mapas de Ramachandran refletidos. Como todas as ligações e ângulos continuavam
corretos, o erro só aparece num teste que meça quiralidade.

Executar com:  pytest -q
"""

from __future__ import annotations

import math

import pytest

from src.build_peptide import (
    build_ala_peptide, dihedral, distance, is_l_amino_acid,
    _cross, _dot, _sub, _scale, _unit, _norm,
)


# --------------------------------------------------------------------------
# Convenção de diedro
# --------------------------------------------------------------------------
@pytest.mark.parametrize("d, esperado", [
    ((1.0, 1.0, 0.0), 0.0),
    ((1.0, -1.0, 0.0), 180.0),
    ((1.0, 0.0, 1.0), 90.0),
    ((1.0, 0.0, -1.0), -90.0),
])
def test_dihedral_segue_convencao_iupac(d, esperado):
    """Caso analítico: b na origem, c em +x, a em +y."""
    a, b, c = (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)
    assert abs(abs(dihedral(a, b, c, d)) - abs(esperado)) < 1e-9
    if esperado not in (0.0, 180.0):
        assert math.copysign(1, dihedral(a, b, c, d)) == math.copysign(1, esperado)


@pytest.mark.parametrize("alvo", [0.0, 45.0, 90.0, -90.0, 137.0, -179.0])
def test_place_e_dihedral_sao_inversos(alvo):
    """`place` e `dihedral` precisam usar a MESMA convenção de sinal."""
    from src.build_peptide import place
    a, b, c = (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)
    d = place(a, b, c, 1.5, 109.5, alvo)
    assert dihedral(a, b, c, d) == pytest.approx(alvo, abs=1e-9)


# --------------------------------------------------------------------------
# Topologia e geometria local
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n, n_atomos", [(1, 22), (2, 32), (10, 112)])
def test_contagem_de_atomos(n, n_atomos):
    p = build_ala_peptide(n)
    assert len(p) == n_atomos
    assert len(p.bonds) == n_atomos - 1  # molécula acíclica


@pytest.mark.parametrize("n", [1, 3, 10])
def test_comprimentos_de_ligacao_plausiveis(n):
    p = build_ala_peptide(n)
    for i, j in p.bonds:
        d = distance(p.atoms[i].pos, p.atoms[j].pos)
        assert 0.95 <= d <= 1.65, f"{p.atoms[i]}-{p.atoms[j]} = {d:.3f} Å"


@pytest.mark.parametrize("phi, psi", [(-60.0, -45.0), (-139.0, 135.0), (60.0, 60.0)])
def test_phi_psi_sao_reproduzidos(phi, psi):
    n = 4
    p = build_ala_peptide(n, phi, psi)
    for i in range(1, n + 1):
        prev_c = p.pos("ACE:C") if i == 1 else p.pos(f"{i - 1}:C")
        next_n = p.pos(f"{i + 1}:N") if i < n else p.pos("NME:N")
        assert dihedral(prev_c, p.pos(f"{i}:N"), p.pos(f"{i}:CA"),
                        p.pos(f"{i}:C")) == pytest.approx(phi, abs=1e-6)
        assert dihedral(p.pos(f"{i}:N"), p.pos(f"{i}:CA"), p.pos(f"{i}:C"),
                        next_n) == pytest.approx(psi, abs=1e-6)


def test_ligacoes_peptidicas_sao_trans():
    n = 5
    p = build_ala_peptide(n)
    for i in range(2, n + 1):
        om = dihedral(p.pos(f"{i - 1}:CA"), p.pos(f"{i - 1}:C"),
                      p.pos(f"{i}:N"), p.pos(f"{i}:CA"))
        assert abs(abs(om) - 180.0) < 1e-6


# --------------------------------------------------------------------------
# Quiralidade
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 5, 10])
def test_todos_os_residuos_sao_l(n):
    p = build_ala_peptide(n)
    for i in range(1, n + 1):
        assert is_l_amino_acid(
            p.pos(f"{i}:N"), p.pos(f"{i}:CA"), p.pos(f"{i}:C"),
            p.pos(f"{i}:CB"), p.pos(f"{i}:HA")
        ), f"resíduo {i} saiu na configuração D"


def test_alpha_helix_is_right_handed():
    """(phi,psi)=(-57,-47) deve gerar uma alfa-hélice DESTRA de parâmetros padrão.

    Mede a helicidade pela regra da mão direita — produtos vetoriais apenas —
    sem passar por `dihedral()`, de modo a ser um teste independente dela.
    """
    n = 12
    p = build_ala_peptide(n, -57.0, -47.0)
    ca = [p.pos(f"{i}:CA") for i in range(1, n + 1)]

    passos = [_sub(ca[i + 1], ca[i]) for i in range(len(ca) - 1)]
    eixo = _unit(tuple(sum(s[k] for s in passos) for k in range(3)))
    centro = _scale(tuple(sum(c[k] for c in ca) for k in range(3)), 1.0 / len(ca))

    def radial(pt):
        v = _sub(pt, centro)
        return _sub(v, _scale(eixo, _dot(v, eixo)))

    giros, avancos, raios = [], [], []
    for i in range(len(ca) - 1):
        ri, rj = radial(ca[i]), radial(ca[i + 1])
        # regra da mão direita: avanço em +eixo com rotação positiva = destra
        assert _dot(_cross(ri, rj), eixo) > 0, f"volta {i} saiu canhota"
        giros.append(math.degrees(math.acos(
            max(-1.0, min(1.0, _dot(_unit(ri), _unit(rj)))))))
        avancos.append(_dot(_sub(ca[i + 1], ca[i]), eixo))
        raios.append(_norm(ri))

    giro = sum(giros) / len(giros)
    assert 360.0 / giro == pytest.approx(3.6, abs=0.15)      # resíduos por volta
    assert sum(avancos) / len(avancos) == pytest.approx(1.5, abs=0.15)  # avanço (Å)
    assert sum(raios) / len(raios) == pytest.approx(2.3, abs=0.2)       # raio (Å)

    # ponte de hidrogênio i -> i+4, assinatura da alfa-hélice
    d = [distance(p.pos(f"{i}:O"), p.pos(f"{i + 4}:H")) for i in range(1, n - 3)]
    assert sum(d) / len(d) == pytest.approx(2.0, abs=0.3)


# --------------------------------------------------------------------------
# Saída
# --------------------------------------------------------------------------
def test_pdb_escrito_tem_atomos_e_conect(tmp_path):
    from src.build_peptide import write_pdb
    p = build_ala_peptide(1)
    out = tmp_path / "ala.pdb"
    write_pdb(p, out, title="teste")
    linhas = out.read_text(encoding="utf-8").splitlines()
    assert sum(1 for ln in linhas if ln.startswith("ATOM")) == 22
    assert sum(1 for ln in linhas if ln.startswith("CONECT")) >= 22
    assert {ln[17:20].strip() for ln in linhas if ln.startswith("ATOM")} == {
        "ACE", "ALA", "NME"}
