"""Parâmetros de análise compartilhados entre o notebook 05 e `molsim.figuras`.

Mudar um valor aqui muda os dois ao mesmo tempo — o relatório e o notebook
que imprime os números citados nele não podem divergir.
"""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
TRAJ_500K = RAIZ / "notebooks" / "trajectory.xyz"
TRAJ_300K = RAIZ / "notebooks" / "trajectory_300K.xyz"

DT_PS = 10.0               # intervalo de gravação: 1 frame = 10 ps
K_ESTADOS = 4              # partição de referência (canonical_states)

LAGS = (1, 2, 3, 5, 8, 10, 15, 20, 30, 40)   # tempos de atraso, em frames
LAG_REF = 10               # τ das tabelas e do bootstrap das escalas de tempo
K_MICRO = (20, 40)         # microestados para o espectro de T(τ)
N_INIT_MICRO = 20          # inicializações do best_of dos microestados
BOOT_BLOCOS = 10           # blocos contíguos do bootstrap
BOOT_N = 500               # réplicas do bootstrap das escalas de tempo
BOOT_N_POP = 1000          # réplicas do bootstrap de populações e ΔG
N_SEMENTES = 50            # sementes do teste de reprodutibilidade de K=4
