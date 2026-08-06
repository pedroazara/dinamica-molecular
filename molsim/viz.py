"""Animação da trajetória.

Gera um GIF da molécula ao longo do tempo, com a trajetória previamente
alinhada para remover a rotação de corpo rígido.

Duas observações sobre estes dados, que explicam o resultado:

1. Cada frame foi relaxado a T = 0 K por minimização antes de ser gravado, e
   os frames estão a 10 ps um do outro. A animação NÃO mostra vibração
   molecular — mostra uma sequência de mínimos locais. O movimento aos
   trancos é o comportamento correto, não um defeito.

2. O centro de massa praticamente não se move (~0,04 Å em 200 frames), então
   não há translação a remover. A orientação, essa sim, muda de frame para
   frame, porque cada um foi minimizado de forma independente. Sem alinhar, a
   molécula parece girar aleatoriamente e não se enxerga mudança de forma
   nenhuma.

Uso:
    python -m molsim.viz --frames 200
    python -m molsim.viz --traj notebooks/trajectory_300K.xyz --frames 300 --fps 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Cores CPK e tamanhos por elemento.
CPK = {"C": "#909090", "N": "#3050F8", "O": "#FF0D0D", "H": "#EEEEEE"}
SIZES = {"C": 110, "N": 115, "O": 115, "H": 40}


def load_aligned(traj: Path, stop: int | None = None, step: int = 1):
    """Carrega a trajetória e remove a rotação de corpo rígido.

    Devolve (coords, names, bonds, times) com coords em (n_frames, n_atoms, 3).
    """
    import MDAnalysis as mda
    from MDAnalysis.analysis import align

    guess = ["bonds", "angles", "dihedrals", "masses"]
    u = mda.Universe(str(traj), to_guess=guess, dt=10)
    ref = mda.Universe(str(traj), to_guess=guess, dt=10)
    ref.trajectory[0]

    # in_memory=True escreve as coordenadas alinhadas de volta na Universe
    align.AlignTraj(u, ref, select="all", in_memory=True).run(stop=stop)

    names = list(u.atoms.names)
    bonds = [(b.atoms[0].index, b.atoms[1].index) for b in u.bonds]

    coords, times = [], []
    for ts in u.trajectory[:stop:step]:
        coords.append(u.atoms.positions.copy())
        times.append(ts.time)
    return np.array(coords), names, bonds, np.array(times)


def animate(coords, names, bonds, out: Path, fps: int = 15,
            spin: float = 0.25, title: str = "") -> Path:
    """Escreve o GIF. `spin` é o giro da câmera em graus por frame."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    cores = [CPK.get(n[0], "#B0B0B0") for n in names]
    tam = [SIZES.get(n[0], 90) for n in names]

    # Limites fixos e cúbicos: sem isso os eixos se reescalam a cada frame e a
    # molécula parece pulsar.
    centro = coords.reshape(-1, 3).mean(0)
    raio = np.linalg.norm(coords - centro, axis=-1).max() * 1.02

    fig = plt.figure(figsize=(6, 6), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    # zoom compensa o encolhimento da projeção 3D dentro do quadro
    ax.set_box_aspect((1, 1, 1), zoom=1.45)
    ax.set_axis_off()

    # precisa nascer com segmentos: add_collection3d rejeita coleção vazia
    linhas = Line3DCollection([[coords[0][a], coords[0][b]] for a, b in bonds],
                              colors="#555555", linewidths=2.2, zorder=1)
    ax.add_collection3d(linhas)
    pontos = ax.scatter(*coords[0].T, s=tam, c=cores,
                        edgecolors="#303030", linewidths=0.6, depthshade=True)
    rotulo = ax.text2D(0.02, 0.96, "", transform=ax.transAxes,
                       fontsize=10, family="monospace")
    if title:
        ax.set_title(title, fontsize=11)

    def desenhar(i):
        xyz = coords[i]
        pontos._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
        linhas.set_segments([[xyz[a], xyz[b]] for a, b in bonds])
        rotulo.set_text(f"frame {i:4d}")
        ax.view_init(elev=18, azim=(i * spin) % 360)
        ax.set_xlim(centro[0] - raio, centro[0] + raio)
        ax.set_ylim(centro[1] - raio, centro[1] + raio)
        ax.set_zlim(centro[2] - raio, centro[2] + raio)
        return pontos, linhas, rotulo

    anim = FuncAnimation(fig, desenhar, frames=len(coords), interval=1000 / fps)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    raiz = Path(__file__).resolve().parents[1]
    ap.add_argument("--traj", type=Path, default=raiz / "notebooks" / "trajectory.xyz")
    ap.add_argument("--frames", type=int, default=200, help="quantos frames animar")
    ap.add_argument("--step", type=int, default=1, help="pular de N em N frames")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--out", type=Path, default=raiz / "figs" / "trajetoria.gif")
    args = ap.parse_args()

    coords, names, bonds, times = load_aligned(args.traj, args.frames, args.step)
    print(f"{len(coords)} frames alinhados, {coords.shape[1]} átomos, "
          f"{len(bonds)} ligações  ({times[0]:.0f}–{times[-1]:.0f} ps)")

    out = animate(coords, names, bonds, args.out, fps=args.fps,
                  title=f"{args.traj.stem} — alinhada")
    print(f"-> {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
