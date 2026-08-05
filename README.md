# Aplicação de Métodos de Aprendizado de Máquina na Análise de Simulações de Dinâmica Molecular

Iniciação Científica PIBIC/CNPq — Departamento de Física, UFLA
Discente: Pedro Henrique Ázara de Almeida · Orientador: Prof. Igor Saulo Santos de Oliveira

Identificação de **estados metaestáveis** em trajetórias de dinâmica molecular
por agrupamento não supervisionado de descritores estruturais.

O projeto segue o roteiro do capítulo 7 do **Lab Course MolSim 2025**
(*Machine Learning for Molecular Simulation*, Universidade de Amsterdã), ao qual
o plano de trabalho registrado no SIGAA corresponde item a item. O planejamento
completo está em [PLANO.md](PLANO.md).

## Sistema

**Ace-(Ala)₆-NH₂** — hexâmero de alanina capeado, modelo de átomo unido, 42 sítios.

| trajetória | T | frames | duração | gravação |
|---|---|---|---|---|
| `trajectory.xyz` | 500 K | 2000 | 20 ns | 10 ps |
| `trajectory_300K.xyz` | 300 K | 1000 | 10 ns | 10 ps |

Fornecidas com o exercício, em `G:/Meu Drive/ufla/ic/exercises/`. Cada frame foi
relaxado a T = 0 K por minimização de energia antes de ser gravado.

A partir da Fase D o projeto gera trajetórias próprias do mesmo peptídeo em
campo de força Amber all-atom, para obter estatística maior e mais de uma
temperatura.

## Descritores

- **raio de giro** — compactação global, ponderado pela massa
- **RMSD** ao primeiro frame, após superposição ótima (Kabsch)
- **12 ângulos diedros** — φ e ψ dos 6 resíduos

## Contribuição

Implementação de K-means adequado a features **periódicas**. O algoritmo de
Lloyd só converge se a regra de atualização for o minimizador da métrica usada
na atribuição; combinar distância periódica com média aritmética quebra essa
garantia e leva a ciclos limite. Ver §3 do [PLANO.md](PLANO.md) para o
resultado medido.

| variante | atribuição | atualização | coerente |
|---|---|---|---|
| `tutorial` | imagem mínima (L=360) | média aritmética | não |
| `mista` | imagem mínima (L=360) | média circular | parcialmente |
| `corda` | Σ(1 − cos Δθ) | média circular | **sim** |

## Instalação

**As Fases A–C não exigem instalação.** numpy, scipy, scikit-learn, pandas e
matplotlib bastam, e [molsim/data.py](molsim/data.py) lê os `.xyz` sem
MDAnalysis.

O ambiente conda só é necessário na Fase D, para as simulações em OpenMM:

```bash
conda env create -f environment.yml
conda activate md-ml
python -m openmm.testInstallation
```

## Uso

```bash
python -m molsim.analyze
```

```bash
python -m molsim.experiment_coherence --k 6 8 --seeds 20
```

```bash
pytest -q
```

## Estrutura

```
molsim/                   análise do exercício MolSim (Fases A–C)
  data.py                 leitura .xyz, diedros, raio de giro, RMSD
  kmeans_variants.py      as três variantes + diagnóstico de coerência
  analyze.py              pipeline completo ponta-a-ponta
  experiment_coherence.py experimento de convergência
src/                      simulação própria em OpenMM (Fase D)
  build_peptide.py        construção de peptídeos por coordenadas internas
  config.py               caminhos e parâmetros de simulação
  simulate.py             produção NVT
notebooks/                00 (setup OpenMM), 01–04 (análise)
tests/                    regressão geométrica e numérica
structures/               PDBs iniciais (versionados)
figs/molsim/              figuras geradas
```

## Notas técnicas

**Índices dos diedros.** A ordem dos átomos nos `.xyz` é regular
(`0:CH3 1:C 2:O`, depois `N,H,CA,CB,C,O` por resíduo, e `39:N 40:H 41:H`), o que
permite indexar os diedros diretamente. Isso dispensa o `to_guess` do
MDAnalysis, cuja lista de diedros inferidos depende de raios de van der Waals e
varia entre versões.

**Dois erros no material original.** O PDF lista 11 diedros na dica da §7.4
(falta o índice 31); o notebook usa os 12 corretos. E `trajectory_300K.xyz` usa
nomes de tipo GROMOS (`CH3`, `CH1`, `NH1`) em vez de símbolos de elemento, o que
impede a inferência de elementos e quebra a tarefa opcional de KNN sem
renomeação prévia.

**Construção de estruturas (Fase D).** As coordenadas iniciais são geradas por
NeRF a partir de coordenadas internas padrão. A suíte de testes verifica
quiralidade **L** e reproduz os parâmetros da α-hélice destra (3,6 resíduos por
volta, avanço de 1,5 Å, raio de 2,3 Å) a partir de (φ,ψ) = (−57°, −47°). Esse
teste não é decorativo: uma inversão de sinal entre a colocação de átomos e a
medida de diedros gera estruturas **espelhadas** sem alterar nenhum comprimento
de ligação nem ângulo de valência.
