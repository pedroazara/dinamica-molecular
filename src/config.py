"""Caminhos e parâmetros globais do projeto.

As trajetórias ficam FORA da pasta sincronizada do OneDrive: um microssegundo
de Ala10 gravado a cada 10 ps ocupa alguns GB, e deixar isso sob sincronização
trava a máquina durante a produção. O destino padrão é `C:/md-data/...` e pode
ser trocado pela variável de ambiente `MD_DATA_ROOT`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# Caminhos
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
STRUCTURES = REPO_ROOT / "structures"
FIGS = REPO_ROOT / "figs"
NOTEBOOKS = REPO_ROOT / "notebooks"

DATA_ROOT = Path(
    os.environ.get("MD_DATA_ROOT", r"C:/md-data/3-dinamica-molecular")
).resolve()
RAW = DATA_ROOT / "raw"
PROCESSED = DATA_ROOT / "processed"


def ensure_dirs() -> None:
    for d in (STRUCTURES, FIGS, RAW, PROCESSED):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Parâmetros de simulação (§1.3 do PLANO.md)
# --------------------------------------------------------------------------
TEMPERATURE_K = 300.0
FRICTION_PER_PS = 1.0
TIMESTEP_FS = 2.0        # 2 fs com constraints=HBonds
REPORT_PS = 10.0         # intervalo de gravação de frames
EQUIL_NS = 1.0           # equilibração descartada
NONBONDED_CUTOFF_NM = 1.0
IMPLICIT_CUTOFF_NM = 2.0
SOLVENT_PADDING_NM = 1.0


@dataclass(frozen=True)
class SystemSpec:
    """Definição de um dos dois sistemas moleculares do projeto."""

    key: str
    label: str
    pdb: Path
    solvent: str          # "explicit" | "implicit"
    production_ns: float
    n_replicas: int = 3

    @property
    def forcefield_files(self) -> tuple[str, ...]:
        if self.solvent == "explicit":
            return ("amber14-all.xml", "amber14/tip3p.xml")
        return ("amber14-all.xml", "implicit/gbn2.xml")


SYSTEMS: dict[str, SystemSpec] = {
    "A": SystemSpec(
        key="A",
        label="Dipeptídeo de alanina (ACE-ALA-NME), TIP3P explícito",
        pdb=STRUCTURES / "alanine_dipeptide.pdb",
        solvent="explicit",
        production_ns=500.0,
    ),
    "B": SystemSpec(
        key="B",
        label="Deca-alanina capeada (ACE-ALA10-NME), GBn2 implícito",
        pdb=STRUCTURES / "ala10.pdb",
        solvent="implicit",
        production_ns=1000.0,
    ),
}


def run_dir(system: str, replica: int) -> Path:
    """Diretório de saída de uma réplica: data/raw/sysA/rep1/."""
    return RAW / f"sys{system.upper()}" / f"rep{replica}"


def replica_seed(system: str, replica: int) -> int:
    """Semente determinística e distinta por (sistema, réplica)."""
    return 1000 * (ord(system.upper()) - ord("A") + 1) + replica
