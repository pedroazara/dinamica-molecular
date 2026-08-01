# Aplicação de Métodos de Aprendizado de Máquina na Análise de Simulações de Dinâmica Molecular

Iniciação Científica PIBIC/CNPq — Departamento de Física, UFLA
Discente: Pedro Henrique Ázara de Almeida · Orientador: Prof. Igor Saulo Santos de Oliveira

Identificação automática de **estados metaestáveis** em trajetórias de dinâmica
molecular, por agrupamento não supervisionado de descritores estruturais.
O plano completo está em [PLANO.md](PLANO.md).

## Sistemas

| | Sistema | Átomos | Solvente | Papel |
|---|---|---|---|---|
| **A** | ACE-ALA-NME (dipeptídeo de alanina) | 22 | TIP3P explícito | validação — bacias conhecidas na literatura, espaço (φ,ψ) bidimensional |
| **B** | ACE-(ALA)₁₀-NME (deca-alanina) | 112 | GBn2 implícito | aplicação — transições hélice ↔ novelo, espaço alto-dimensional |

## Instalação

O Python do sistema não serve: OpenMM, MDTraj e MDAnalysis são distribuídos via
conda-forge e não publicam builds para as versões mais recentes do interpretador.

```bash
conda env create -f environment.yml
conda activate md-ml
python -m openmm.testInstallation
```

O último comando precisa listar `CUDA` entre as plataformas.

### Onde ficam os dados

As trajetórias **não** ficam no repositório nem na pasta sincronizada do
OneDrive — um microssegundo de Ala10 ocupa alguns GB e a sincronização
travaria a máquina durante a produção. O destino padrão é
`C:/md-data/3-dinamica-molecular/`, configurável pela variável de ambiente
`MD_DATA_ROOT` (ver [src/config.py](src/config.py)).

## Reprodução

```bash
# 1. estruturas iniciais (determinísticas, sem download)
python -m src.build_peptide --n 1  --out structures/alanine_dipeptide.pdb
python -m src.build_peptide --n 10 --phi -57 --psi -47 --out structures/ala10.pdb

# 2. verificação da geometria
pytest -q

# 3. teste de fumaça ponta-a-ponta (poucos minutos)
python -m src.simulate --system A --replica 0 --ns 1 --equil-ns 0.02

# 4. produção
powershell -File scripts/run_replicas.ps1 -System A
powershell -File scripts/run_replicas.ps1 -System B
```

Ou, de forma guiada, [notebooks/00_setup.ipynb](notebooks/00_setup.ipynb).

## Estrutura

```
src/build_peptide.py    construção de ACE-(ALA)n-NME por coordenadas internas
src/config.py           caminhos e parâmetros de simulação
src/simulate.py         produção NVT em OpenMM
tests/                  regressão geométrica (25 testes)
notebooks/00_setup.ipynb  Fase 0 — validação ponta-a-ponta
structures/             PDBs iniciais (versionados)
figs/                   figuras geradas
```

## Protocolo de simulação

Ensemble NVT · termostato de Langevin (300 K, γ = 1 ps⁻¹) · passo de 2 fs com
`constraints=HBonds` · campo de força `amber14-all` · minimização seguida de
1 ns de equilibração descartada · gravação a cada 10 ps · 3 réplicas
independentes por sistema, com sementes distintas.

Apenas os átomos do soluto entram na trajetória: no Sistema A a água é ~99,5%
dos átomos e nenhum descritor do projeto a utiliza.

## Nota sobre a construção das estruturas

As coordenadas iniciais são geradas por NeRF a partir de comprimentos de
ligação, ângulos e diedros padrão. A suíte de testes verifica quiralidade **L**
em todos os resíduos e reproduz os parâmetros da α-hélice destra
(3,6 resíduos/volta, avanço de 1,5 Å, raio de 2,3 Å, ponte O(i)···H(i+4) de
2,0 Å) a partir de (φ,ψ) = (−57°, −47°).

Esse teste não é decorativo: uma inversão de sinal entre a colocação de átomos
e a medida de diedros produz estruturas **espelhadas** — hélices canhotas e
mapas de Ramachandran refletidos — sem alterar nenhum comprimento de ligação
nem ângulo de valência. O erro é invisível a qualquer verificação que não meça
quiralidade explicitamente.
