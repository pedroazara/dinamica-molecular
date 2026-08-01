"""Produção de dinâmica molecular em OpenMM.

Implementa o protocolo da §1.3 do PLANO.md: ensemble NVT, termostato de
Langevin a 300 K, passo de 2 fs com `constraints=HBonds`, minimização seguida
de 1 ns de equilibração descartada, gravação a cada 10 ps.

Apenas os átomos do soluto são gravados na trajetória. No Sistema A a água
representa ~99,5% dos átomos e nenhum dos descritores do projeto (diedros,
raio de giro, RMSD) a utiliza — gravá-la multiplicaria o tamanho da trajetória
por ~200 sem retorno.

Uso:
    python -m src.simulate --system A --replica 1 --ns 500
    python -m src.simulate --system A --replica 1 --ns 1 --equil-ns 0.02   # teste
"""

from __future__ import annotations

import argparse
import json
import platform as _platform
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import openmm as mm
from openmm import app, unit

from . import config as cfg


# --------------------------------------------------------------------------
# Plataforma
# --------------------------------------------------------------------------
def available_platforms() -> list[str]:
    return [mm.Platform.getPlatform(i).getName()
            for i in range(mm.Platform.getNumPlatforms())]


def select_platform(preferred: str | None = None):
    """Escolhe a plataforma mais rápida disponível, com fallback para CPU."""
    names = available_platforms()
    order = [preferred] if preferred else ["CUDA", "OpenCL", "CPU", "Reference"]
    for name in order:
        if name in names:
            p = mm.Platform.getPlatformByName(name)
            props = {"Precision": "mixed"} if name in ("CUDA", "OpenCL") else {}
            return p, props
    raise RuntimeError(f"nenhuma plataforma utilizável entre {names}")


# --------------------------------------------------------------------------
# Montagem do sistema
# --------------------------------------------------------------------------
def build_simulation(spec: cfg.SystemSpec, seed: int, platform_name: str | None,
                     out_dir: Path):
    """Monta topologia, sistema e `Simulation`; devolve também os índices do soluto."""
    if not spec.pdb.exists():
        raise FileNotFoundError(
            f"{spec.pdb} não existe. Gere com:\n"
            f"    python -m src.build_peptide --n "
            f"{1 if spec.key == 'A' else 10} --out {spec.pdb}"
        )

    pdb = app.PDBFile(str(spec.pdb))
    forcefield = app.ForceField(*spec.forcefield_files)
    n_solute = pdb.topology.getNumAtoms()

    if spec.solvent == "explicit":
        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addSolvent(
            forcefield,
            model="tip3p",
            padding=cfg.SOLVENT_PADDING_NM * unit.nanometer,
            neutralize=True,
        )
        topology, positions = modeller.topology, modeller.positions
        system = forcefield.createSystem(
            topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=cfg.NONBONDED_CUTOFF_NM * unit.nanometer,
            constraints=app.HBonds,
            rigidWater=True,
        )
    else:
        topology, positions = pdb.topology, pdb.positions
        system = forcefield.createSystem(
            topology,
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=cfg.IMPLICIT_CUTOFF_NM * unit.nanometer,
            constraints=app.HBonds,
            soluteDielectric=1.0,
            solventDielectric=78.5,
        )

    # Ensemble NVT: termostato de Langevin, sem barostato.
    integrator = mm.LangevinMiddleIntegrator(
        cfg.TEMPERATURE_K * unit.kelvin,
        cfg.FRICTION_PER_PS / unit.picosecond,
        cfg.TIMESTEP_FS * unit.femtoseconds,
    )
    integrator.setRandomNumberSeed(seed)

    plat, props = select_platform(platform_name)
    simulation = app.Simulation(topology, system, integrator, plat, props)
    simulation.context.setPositions(positions)

    # A topologia de referência da análise contém apenas o soluto, para casar
    # com os frames gravados.
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "topology.pdb", "w") as fh:
        app.PDBFile.writeFile(pdb.topology, pdb.positions, fh)

    return simulation, list(range(n_solute)), plat.getName()


# --------------------------------------------------------------------------
# Execução
# --------------------------------------------------------------------------
def run(system: str, replica: int, ns: float | None = None,
        equil_ns: float | None = None, platform_name: str | None = None,
        report_ps: float | None = None, out_dir: Path | None = None) -> Path:
    spec = cfg.SYSTEMS[system.upper()]
    ns = spec.production_ns if ns is None else ns
    equil_ns = cfg.EQUIL_NS if equil_ns is None else equil_ns
    report_ps = cfg.REPORT_PS if report_ps is None else report_ps
    out_dir = cfg.run_dir(system, replica) if out_dir is None else out_dir
    seed = cfg.replica_seed(system, replica)

    cfg.ensure_dirs()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{spec.key}/rep{replica}] {spec.label}")
    print(f"[{spec.key}/rep{replica}] plataformas disponíveis: "
          f"{', '.join(available_platforms())}")

    t0 = time.time()
    simulation, solute, plat_name = build_simulation(
        spec, seed, platform_name, out_dir)
    n_total = simulation.topology.getNumAtoms()
    print(f"[{spec.key}/rep{replica}] plataforma: {plat_name} | "
          f"{n_total} átomos ({len(solute)} de soluto) | seed {seed}")

    steps_per_report = int(round(report_ps * 1000.0 / cfg.TIMESTEP_FS))
    equil_steps = int(round(equil_ns * 1e6 / cfg.TIMESTEP_FS))
    prod_steps = int(round(ns * 1e6 / cfg.TIMESTEP_FS))

    print(f"[{spec.key}/rep{replica}] minimizando…")
    simulation.minimizeEnergy()

    simulation.context.setVelocitiesToTemperature(
        cfg.TEMPERATURE_K * unit.kelvin, seed)
    print(f"[{spec.key}/rep{replica}] equilibrando {equil_ns} ns (descartado)…")
    simulation.step(equil_steps)

    # Zera o relógio para que a trajetória de produção comece em t = 0.
    simulation.context.setTime(0.0)
    simulation.currentStep = 0

    # O reporter do MDTraj aceita subconjunto de átomos; o do OpenMM não.
    from mdtraj.reporters import DCDReporter as MDTrajDCDReporter

    traj_path = out_dir / "traj.dcd"
    simulation.reporters.append(
        MDTrajDCDReporter(str(traj_path), steps_per_report, atomSubset=solute))
    simulation.reporters.append(app.StateDataReporter(
        str(out_dir / "log.csv"), steps_per_report, step=True, time=True,
        potentialEnergy=True, kineticEnergy=True, temperature=True,
        volume=(spec.solvent == "explicit"), speed=True))
    simulation.reporters.append(app.StateDataReporter(
        sys.stdout, steps_per_report * 100, step=True, time=True,
        temperature=True, speed=True, progress=True, remainingTime=True,
        totalSteps=prod_steps))
    simulation.reporters.append(app.CheckpointReporter(
        str(out_dir / "state.chk"), steps_per_report * 100))

    print(f"[{spec.key}/rep{replica}] produção: {ns} ns "
          f"({prod_steps:,} passos, {prod_steps // steps_per_report:,} frames)")
    simulation.step(prod_steps)

    elapsed = time.time() - t0
    meta = {
        "sistema": spec.key,
        "descricao": spec.label,
        "replica": replica,
        "seed": seed,
        "estrutura_inicial": str(spec.pdb),
        "campo_de_forca": list(spec.forcefield_files),
        "solvente": spec.solvent,
        "ensemble": "NVT",
        "temperatura_K": cfg.TEMPERATURE_K,
        "friccao_por_ps": cfg.FRICTION_PER_PS,
        "passo_fs": cfg.TIMESTEP_FS,
        "constraints": "HBonds",
        "equilibracao_ns": equil_ns,
        "producao_ns": ns,
        "intervalo_gravacao_ps": report_ps,
        "n_frames": prod_steps // steps_per_report,
        "n_atomos_total": n_total,
        "n_atomos_soluto": len(solute),
        "plataforma": plat_name,
        "openmm": mm.version.version,
        "python": sys.version.split()[0],
        "host": _platform.node(),
        "inicio": datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
        "duracao_s": round(elapsed, 1),
        "ns_por_dia": round(ns / (elapsed / 86400.0), 1) if elapsed > 0 else None,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[{spec.key}/rep{replica}] concluído em {timedelta(seconds=int(elapsed))} "
          f"({meta['ns_por_dia']} ns/dia) -> {traj_path}")
    return traj_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", choices=["A", "B"], required=True)
    ap.add_argument("--replica", type=int, default=1)
    ap.add_argument("--ns", type=float, default=None,
                    help="duração da produção (padrão: valor do config)")
    ap.add_argument("--equil-ns", type=float, default=None)
    ap.add_argument("--report-ps", type=float, default=None)
    ap.add_argument("--platform", default=None,
                    choices=["CUDA", "OpenCL", "CPU", "Reference"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run(args.system, args.replica, args.ns, args.equil_ns,
        args.platform, args.report_ps, args.out)


if __name__ == "__main__":
    main()
