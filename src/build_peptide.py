"""Construção de peptídeos ACE-(ALA)n-NME a partir de coordenadas internas.

Gera as estruturas iniciais dos dois sistemas do projeto sem depender de
download nem de ferramentas externas (tleap, PDBFixer, CHARMM-GUI). As posições
são construídas por NeRF (Natural Extension Reference Frame) a partir de um
conjunto de comprimentos de ligação, ângulos e diedros padrão de peptídeos.

    Sistema A: n = 1   -> dipeptídeo de alanina (ACE-ALA-NME), 22 átomos
    Sistema B: n = 10  -> deca-alanina capeada (ACE-ALA10-NME), 109 átomos

Os nomes de átomo e resíduo seguem a convenção Amber, de modo que o PDB gerado
casa diretamente com os templates de `amber14-all.xml` do OpenMM. Registros
CONECT são escritos para garantir a conectividade correta na leitura.

Uso:
    python -m src.build_peptide --n 1  --out structures/alanine_dipeptide.pdb
    python -m src.build_peptide --n 10 --out structures/ala10.pdb
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

Vec = tuple[float, float, float]

# --------------------------------------------------------------------------
# Coordenadas internas padrão (Å / graus)
# --------------------------------------------------------------------------
B_C_N = 1.335    # ligação peptídica C-N
B_N_CA = 1.449
B_CA_C = 1.522
B_C_O = 1.229
B_N_H = 1.010
B_CA_CB = 1.525
B_CA_HA = 1.090
B_C_H = 1.090    # C-H em metila
B_C_CT = 1.522   # C(carbonila)-CH3 na ACE

A_CT_C_N = 116.6
A_C_N_CA = 121.9
A_N_CA_C = 110.1
A_CA_C_N = 116.6
A_O_C_N = 122.9  # diedro-referência coloca O anti ao CA seguinte
A_H_N_C = 119.0
A_N_CA_CB = 110.5
A_N_CA_HA = 109.5
A_TETRA = 109.5

OMEGA = 180.0  # ligação peptídica trans

# Conformação inicial: região alfa-R / C7eq
PHI_DEFAULT = -60.0
PSI_DEFAULT = -45.0


# --------------------------------------------------------------------------
# Álgebra vetorial mínima (sem numpy, para o script rodar em qualquer Python)
# --------------------------------------------------------------------------
def _sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a: Vec, k: float) -> Vec:
    return (a[0] * k, a[1] * k, a[2] * k)


def _dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec, b: Vec) -> Vec:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a: Vec) -> float:
    return math.sqrt(_dot(a, a))


def _unit(a: Vec) -> Vec:
    n = _norm(a)
    if n < 1e-12:
        raise ValueError("vetor nulo não pode ser normalizado")
    return _scale(a, 1.0 / n)


def distance(a: Vec, b: Vec) -> float:
    return _norm(_sub(a, b))


def angle(a: Vec, b: Vec, c: Vec) -> float:
    """Ângulo a-b-c em graus."""
    u, v = _unit(_sub(a, b)), _unit(_sub(c, b))
    return math.degrees(math.acos(max(-1.0, min(1.0, _dot(u, v)))))


def dihedral(a: Vec, b: Vec, c: Vec, d: Vec) -> float:
    """Diedro a-b-c-d em graus, convenção IUPAC (-180, 180]."""
    b0, b1, b2 = _sub(a, b), _sub(c, b), _sub(d, c)
    b1u = _unit(b1)
    # componentes de b0 e b2 perpendiculares a b1
    v = _sub(b0, _scale(b1u, _dot(b0, b1u)))
    w = _sub(b2, _scale(b1u, _dot(b2, b1u)))
    x = _dot(v, w)
    y = _dot(_cross(b1u, v), w)
    return math.degrees(math.atan2(y, x))


def place(a: Vec, b: Vec, c: Vec, bond: float, ang: float, dih: float) -> Vec:
    """Posiciona um átomo D ligado a C, com ângulo B-C-D e diedro A-B-C-D."""
    theta, phi = math.radians(ang), math.radians(dih)
    bc = _unit(_sub(c, b))
    n = _unit(_cross(_sub(b, a), bc))
    m = _cross(n, bc)
    d_local = (
        -bond * math.cos(theta),
        bond * math.sin(theta) * math.cos(phi),
        bond * math.sin(theta) * math.sin(phi),
    )
    offset = (
        bc[0] * d_local[0] + m[0] * d_local[1] + n[0] * d_local[2],
        bc[1] * d_local[0] + m[1] * d_local[1] + n[1] * d_local[2],
        bc[2] * d_local[0] + m[2] * d_local[1] + n[2] * d_local[2],
    )
    return _add(c, offset)


# --------------------------------------------------------------------------
# Estrutura de dados
# --------------------------------------------------------------------------
class Atom:
    __slots__ = ("name", "resname", "resseq", "pos", "element")

    def __init__(self, name: str, resname: str, resseq: int, pos: Vec):
        self.name = name
        self.resname = resname
        self.resseq = resseq
        self.pos = pos
        self.element = "H" if name.startswith("H") else name[0]

    def __repr__(self) -> str:  # pragma: no cover - conveniência de debug
        return f"<Atom {self.resname}{self.resseq}:{self.name}>"


class Peptide:
    """Coleção ordenada de átomos com a lista de ligações."""

    def __init__(self) -> None:
        self.atoms: list[Atom] = []
        self.bonds: list[tuple[int, int]] = []
        self._index: dict[str, int] = {}

    def add(self, key: str, name: str, resname: str, resseq: int, pos: Vec) -> int:
        idx = len(self.atoms)
        self.atoms.append(Atom(name, resname, resseq, pos))
        self._index[key] = idx
        return idx

    def bond(self, key_a: str, key_b: str) -> None:
        self.bonds.append((self._index[key_a], self._index[key_b]))

    def pos(self, key: str) -> Vec:
        return self.atoms[self._index[key]].pos

    def idx(self, key: str) -> int:
        return self._index[key]

    def __len__(self) -> int:
        return len(self.atoms)


# --------------------------------------------------------------------------
# Quiralidade
# --------------------------------------------------------------------------
def is_l_amino_acid(n: Vec, ca: Vec, c: Vec, cb: Vec, ha: Vec) -> bool:
    """Regra CORN: com HA apontando para o observador, C'->CB->N é horário em L.

    Um observador em HA olhando para CA enxerga na direção d = CA - HA. Para
    esse observador, a rotação de `a` para `b` é horária quando (a x b)·d > 0.
    """
    a = _sub(c, ca)
    b = _sub(cb, ca)
    d = _sub(ca, ha)
    return _dot(_cross(a, b), d) > 0.0


# --------------------------------------------------------------------------
# Construção
# --------------------------------------------------------------------------
def build_ala_peptide(
    n_residues: int = 1,
    phi: float = PHI_DEFAULT,
    psi: float = PSI_DEFAULT,
) -> Peptide:
    """Constrói ACE-(ALA)n-NME com todos os resíduos no mesmo (phi, psi)."""
    if n_residues < 1:
        raise ValueError("n_residues deve ser >= 1")

    p = Peptide()
    ace_res, nme_res = 1, n_residues + 2

    # --- esqueleto -------------------------------------------------------
    # Os três primeiros átomos definem o referencial; os demais vêm do NeRF.
    p.add("ACE:CH3", "CH3", "ACE", ace_res, (0.0, 0.0, 0.0))
    p.add("ACE:C", "C", "ACE", ace_res, (B_C_CT, 0.0, 0.0))
    third = (
        B_C_CT - B_C_N * math.cos(math.radians(A_CT_C_N)),
        B_C_N * math.sin(math.radians(A_CT_C_N)),
        0.0,
    )
    p.add("1:N", "N", "ALA", 2, third)

    p.add(
        "1:CA", "CA", "ALA", 2,
        place(p.pos("ACE:CH3"), p.pos("ACE:C"), p.pos("1:N"),
              B_N_CA, A_C_N_CA, OMEGA),
    )
    p.add(
        "1:C", "C", "ALA", 2,
        place(p.pos("ACE:C"), p.pos("1:N"), p.pos("1:CA"),
              B_CA_C, A_N_CA_C, phi),
    )

    for i in range(2, n_residues + 1):
        prev, cur = i - 1, i
        p.add(
            f"{cur}:N", "N", "ALA", cur + 1,
            place(p.pos(f"{prev}:N"), p.pos(f"{prev}:CA"), p.pos(f"{prev}:C"),
                  B_C_N, A_CA_C_N, psi),
        )
        p.add(
            f"{cur}:CA", "CA", "ALA", cur + 1,
            place(p.pos(f"{prev}:CA"), p.pos(f"{prev}:C"), p.pos(f"{cur}:N"),
                  B_N_CA, A_C_N_CA, OMEGA),
        )
        p.add(
            f"{cur}:C", "C", "ALA", cur + 1,
            place(p.pos(f"{prev}:C"), p.pos(f"{cur}:N"), p.pos(f"{cur}:CA"),
                  B_CA_C, A_N_CA_C, phi),
        )

    last = n_residues
    p.add(
        "NME:N", "N", "NME", nme_res,
        place(p.pos(f"{last}:N"), p.pos(f"{last}:CA"), p.pos(f"{last}:C"),
              B_C_N, A_CA_C_N, psi),
    )
    p.add(
        "NME:CH3", "CH3", "NME", nme_res,
        place(p.pos(f"{last}:CA"), p.pos(f"{last}:C"), p.pos("NME:N"),
              B_N_CA, A_C_N_CA, OMEGA),
    )

    # --- substituintes ---------------------------------------------------
    # ACE: carbonila anti ao CA seguinte, metila escalonada em relação ao N.
    p.add(
        "ACE:O", "O", "ACE", ace_res,
        place(p.pos("1:CA"), p.pos("1:N"), p.pos("ACE:C"),
              B_C_O, A_O_C_N, 0.0),
    )
    for name, rot in (("HH31", 60.0), ("HH32", 180.0), ("HH33", -60.0)):
        p.add(
            f"ACE:{name}", name, "ACE", ace_res,
            place(p.pos("1:N"), p.pos("ACE:C"), p.pos("ACE:CH3"),
                  B_C_H, A_TETRA, rot),
        )

    # Quiralidade L: um dos dois arranjos de (CB, HA) satisfaz a regra CORN.
    # Determinamos qual no primeiro resíduo e reutilizamos nos demais.
    cb_offset = _resolve_chirality(p, phi)

    for i in range(1, n_residues + 1):
        prev_c = "ACE:C" if i == 1 else f"{i - 1}:C"
        resseq = i + 1

        p.add(
            f"{i}:H", "H", "ALA", resseq,
            place(p.pos("ACE:CH3") if i == 1 else p.pos(f"{i - 1}:CA"),
                  p.pos(prev_c), p.pos(f"{i}:N"), B_N_H, A_H_N_C, 0.0),
        )
        p.add(
            f"{i}:CB", "CB", "ALA", resseq,
            place(p.pos(prev_c), p.pos(f"{i}:N"), p.pos(f"{i}:CA"),
                  B_CA_CB, A_N_CA_CB, phi + cb_offset),
        )
        p.add(
            f"{i}:HA", "HA", "ALA", resseq,
            place(p.pos(prev_c), p.pos(f"{i}:N"), p.pos(f"{i}:CA"),
                  B_CA_HA, A_N_CA_HA, phi - cb_offset),
        )
        for name, rot in (("HB1", 60.0), ("HB2", 180.0), ("HB3", -60.0)):
            p.add(
                f"{i}:{name}", name, "ALA", resseq,
                place(p.pos(f"{i}:N"), p.pos(f"{i}:CA"), p.pos(f"{i}:CB"),
                      B_C_H, A_TETRA, rot),
            )
        next_n = f"{i + 1}:N" if i < n_residues else "NME:N"
        next_ca = f"{i + 1}:CA" if i < n_residues else "NME:CH3"
        p.add(
            f"{i}:O", "O", "ALA", resseq,
            place(p.pos(next_ca), p.pos(next_n), p.pos(f"{i}:C"),
                  B_C_O, A_O_C_N, 0.0),
        )

    p.add(
        "NME:H", "H", "NME", nme_res,
        place(p.pos(f"{last}:CA"), p.pos(f"{last}:C"), p.pos("NME:N"),
              B_N_H, A_H_N_C, 0.0),
    )
    for name, rot in (("HH31", 60.0), ("HH32", 180.0), ("HH33", -60.0)):
        p.add(
            f"NME:{name}", name, "NME", nme_res,
            place(p.pos(f"{last}:C"), p.pos("NME:N"), p.pos("NME:CH3"),
                  B_C_H, A_TETRA, rot),
        )

    _add_bonds(p, n_residues)
    return _reorder_amber(p, n_residues)


def _resolve_chirality(p: Peptide, phi: float) -> float:
    """Retorna o deslocamento de diedro (+120 ou -120) que produz um L-aminoácido."""
    for offset in (120.0, -120.0):
        cb = place(p.pos("ACE:C"), p.pos("1:N"), p.pos("1:CA"),
                   B_CA_CB, A_N_CA_CB, phi + offset)
        ha = place(p.pos("ACE:C"), p.pos("1:N"), p.pos("1:CA"),
                   B_CA_HA, A_N_CA_HA, phi - offset)
        if is_l_amino_acid(p.pos("1:N"), p.pos("1:CA"), p.pos("1:C"), cb, ha):
            return offset
    raise RuntimeError("nenhum dos arranjos produziu configuração L")


def _add_bonds(p: Peptide, n: int) -> None:
    p.bond("ACE:CH3", "ACE:C")
    p.bond("ACE:C", "ACE:O")
    for h in ("HH31", "HH32", "HH33"):
        p.bond("ACE:CH3", f"ACE:{h}")
    p.bond("ACE:C", "1:N")

    for i in range(1, n + 1):
        p.bond(f"{i}:N", f"{i}:H")
        p.bond(f"{i}:N", f"{i}:CA")
        p.bond(f"{i}:CA", f"{i}:HA")
        p.bond(f"{i}:CA", f"{i}:CB")
        p.bond(f"{i}:CA", f"{i}:C")
        p.bond(f"{i}:C", f"{i}:O")
        for h in ("HB1", "HB2", "HB3"):
            p.bond(f"{i}:CB", f"{i}:{h}")
        p.bond(f"{i}:C", f"{i + 1}:N" if i < n else "NME:N")

    p.bond("NME:N", "NME:H")
    p.bond("NME:N", "NME:CH3")
    for h in ("HH31", "HH32", "HH33"):
        p.bond("NME:CH3", f"NME:{h}")


def _reorder_amber(p: Peptide, n: int) -> Peptide:
    """Reordena os átomos na sequência dos templates Amber."""
    order = ["ACE:HH31", "ACE:CH3", "ACE:HH32", "ACE:HH33", "ACE:C", "ACE:O"]
    for i in range(1, n + 1):
        order += [f"{i}:{a}" for a in
                  ("N", "H", "CA", "HA", "CB", "HB1", "HB2", "HB3", "C", "O")]
    order += ["NME:N", "NME:H", "NME:CH3",
              "NME:HH31", "NME:HH32", "NME:HH33"]

    out = Peptide()
    for key in order:
        a = p.atoms[p.idx(key)]
        out.add(key, a.name, a.resname, a.resseq, a.pos)
    remap = {p.idx(k): out.idx(k) for k in order}
    out.bonds = [(remap[i], remap[j]) for i, j in p.bonds]
    return out


# --------------------------------------------------------------------------
# Escrita
# --------------------------------------------------------------------------
def write_pdb(p: Peptide, path: Path, title: str = "") -> None:
    lines: list[str] = []
    if title:
        lines.append(f"REMARK   1 {title}")
    for i, a in enumerate(p.atoms, start=1):
        name = a.name if len(a.name) >= 4 else f" {a.name:<3s}"
        lines.append(
            f"ATOM  {i:5d} {name}{'':1s}{a.resname:>3s} A{a.resseq:4d}    "
            f"{a.pos[0]:8.3f}{a.pos[1]:8.3f}{a.pos[2]:8.3f}"
            f"{1.00:6.2f}{0.00:6.2f}          {a.element:>2s}"
        )
    lines.append("TER")

    # CONECT garante a conectividade mesmo sem template de resíduo conhecido.
    neighbours: dict[int, list[int]] = {}
    for i, j in p.bonds:
        neighbours.setdefault(i, []).append(j)
        neighbours.setdefault(j, []).append(i)
    for i in sorted(neighbours):
        nb = sorted(neighbours[i])
        for chunk_start in range(0, len(nb), 4):
            chunk = nb[chunk_start:chunk_start + 4]
            lines.append(
                f"CONECT{i + 1:5d}" + "".join(f"{j + 1:5d}" for j in chunk)
            )
    lines.append("END")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1,
                    help="número de resíduos de alanina (1 = Sistema A, 10 = Sistema B)")
    ap.add_argument("--phi", type=float, default=PHI_DEFAULT)
    ap.add_argument("--psi", type=float, default=PSI_DEFAULT)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pep = build_ala_peptide(args.n, args.phi, args.psi)
    write_pdb(pep, args.out,
              title=f"ACE-(ALA){args.n}-NME  phi={args.phi:.1f} psi={args.psi:.1f}")
    print(f"{len(pep)} átomos, {len(pep.bonds)} ligações -> {args.out}")


if __name__ == "__main__":
    main()
