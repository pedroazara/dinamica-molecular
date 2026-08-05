# Plano de Execução — IC PIBIC/CNPq
### Aplicação de Métodos de Aprendizado de Máquina na Análise de Simulações de Dinâmica Molecular

**Discente:** Pedro Henrique Ázara de Almeida (202210612)
**Orientador:** Prof. Igor Saulo Santos de Oliveira (DFI/UFLA)
**Vigência da bolsa:** 01/09/2025 – 31/08/2026
**Horizonte deste plano:** 05/08/2026 → 30/09/2026 (9 semanas)

---

## 1. Base do trabalho

O projeto segue o roteiro do capítulo 7 do **Lab Course MolSim 2025**
(*Machine Learning for Molecular Simulation*, Universidade de Amsterdã), que é
o material adotado pela orientação. O plano de trabalho registrado no SIGAA
corresponde item a item a esse capítulo:

| Plano de trabalho (SIGAA) | Capítulo 7 do MolSim |
|---|---|
| oligopeptídeo | hexâmero de alanina, Ace-(Ala)₆-NH₂ |
| relaxação a 0 K por minimização de energia | protocolo de geração das trajetórias fornecidas |
| raio de giro, RMSD, ângulos diedros | §7.3 e §7.4 |
| K-means com scikit-learn | §7.6 |
| desenvolvimento de versão personalizada do K-means | §7.8 |
| energias livres relativas e matriz de transição | §7.11, tarefas 9 e 10 |

O material de apoio complementar é o conjunto de exercícios de Frenkel & Smit,
*Understanding Molecular Simulation* (`Exercises2024`), em particular o
capítulo 5 (dinâmica molecular básica) como fundamentação — não como
entregável.

### 1.1 Dados de partida

Fornecidos com o exercício, em `G:/Meu Drive/ufla/ic/exercises/`:

| arquivo | T | frames | duração | nomes de átomo |
|---|---|---|---|---|
| `trajectory.xyz` | 500 K | 2000 | 20 ns | símbolos de elemento |
| `trajectory_300K.xyz` | 300 K | 1000 | 10 ns | tipos GROMOS (`CH3`, `CH1`, `NH1`) |

Sistema: Ace-(Ala)₆-NH₂ em modelo de átomo unido, 42 sítios. Gravação a cada
10 ps, cada frame relaxado a T = 0 K por minimização antes de ser escrito.

Dois detalhes práticos já verificados: o PDF lista 11 diedros na dica da §7.4
(falta o índice 31) enquanto o notebook usa os 12 corretos; e os nomes de tipo
GROMOS do arquivo de 300 K impedem a inferência de elementos pelo `to_guess` do
MDAnalysis, o que quebra a tarefa opcional de KNN sem renomeação prévia.

---

## 2. Pergunta de pesquisa

> Métodos de agrupamento não supervisionado aplicados a descritores estruturais
> extraídos de trajetórias de dinâmica molecular conseguem identificar os
> estados metaestáveis de um oligopeptídeo? Sob que critério — geométrico ou
> cinético — a resposta é bem definida?

A contribuição original está em §3, e nasceu de um defeito reprodutível
encontrado no algoritmo proposto pelo material de referência.

---

## 3. Contribuição: K-means coerente para features periódicas

### 3.1 O problema

O algoritmo de Lloyd converge porque os dois passos reduzem o **mesmo**
objetivo: a atribuição reduz ao mover cada ponto para o centro mais próximo, e
a atualização reduz porque o novo centroide é o *minimizador* da soma das
distâncias dos seus membros. Se a regra de atualização não for o minimizador da
métrica usada na atribuição, essa garantia se perde.

A §7.8 do MolSim propõe corrigir a periodicidade dos ângulos diedros usando
distância de **imagem mínima** com L = 360°, mas mantém a atualização como
*"the mean of the positions of its cluster members"* — média aritmética. As
duas escolhas são incompatíveis. A média aritmética de ângulos é indefinida na
descontinuidade ±180°:

```
ângulos          [-170°, 175°, 170°, -175°]
média aritmética     0.00°   →  Σd² = 119050
média circular     180.00°   →  Σd² =    250
```

Na trajetória de 500 K, **11,6% das amostras estão a menos de 40° dessa
descontinuidade**, então o caso não é patológico — é típico.

### 3.2 As três variantes

| variante | atribuição | atualização | coerente |
|---|---|---|---|
| `tutorial` | imagem mínima (L=360) | média aritmética | não |
| `mista` | imagem mínima (L=360) | média circular | parcialmente |
| `corda` | Σ(1 − cos Δθ) | média circular | **sim** |

Só na variante `corda` a atualização é o minimizador exato da métrica de
atribuição: μ = atan2(⟨sin θ⟩, ⟨cos θ⟩) minimiza Σ(1 − cos(θ − μ)). Equivale a
K-means euclidiano sobre a imersão (cos θ, sin θ).

### 3.3 Resultado medido

Trajetória de 500 K, 12 diedros, 20 inicializações por configuração:

| k | variante | convergiu | ciclo limite | objetivo subiu |
|---|---|---|---|---|
| 8 | `tutorial` | 7/20 | **13** | 17/20 |
| 8 | `mista` | 20/20 | 0 | 7/20 |
| 8 | `corda` | 20/20 | 0 | **0/20** |

A receita do material de referência entra em **ciclo limite** — nunca converge,
fica alternando entre partições — em 13 de 20 execuções, com k = 8, que é
justamente o valor fixado no esqueleto do notebook. A variante coerente
converge sempre e o objetivo nunca sobe, em nenhuma semente, para nenhum k
testado (4 a 15).

**Ressalva registrada:** a variante `corda` apresenta inércia de imagem mínima
ligeiramente *maior*, porque otimiza um objetivo diferente. O ganho
demonstrado é de convergência e reprodutibilidade, **não** de agrupamento mais
compacto. Se os estados resultantes são fisicamente melhores é pergunta em
aberto, a ser respondida na Fase C.

Reproduzir com:

```bash
python -m molsim.experiment_coherence
```

---

## 4. Segunda questão: geometria não basta

A primeira execução do pipeline completo sobre a trajetória de 500 K produziu
dois resultados que orientam o resto do projeto:

**A silhueta não estabiliza.** Ela cresce de 0,238 (k=2) até ~0,44 em k=9 e
depois oscila entre 0,43 e 0,48 até k=15, sem patamar. Selecionar k pelo máximo
da silhueta devolve k=14, que é ajuste ao ruído da curva, não um número físico
de estados.

**Os tempos de residência estão no limite da amostragem.** Com k=14 os estados
duram de 15 a 96 ps, contra um intervalo de gravação de 10 ps — ou seja, de 1,5
a 9,6 frames. Estados que sobrevivem poucos frames não são metaestáveis em
sentido útil.

Isso é esperado: a trajetória foi gerada a 500 K justamente para deixar a
molécula flexível. A consequência metodológica é que **"quantos estados
metaestáveis existem" não tem resposta bem definida por critério puramente
geométrico**, e é o que motiva as Fases C e D.

---

## 5. Cronograma (05/08 → 30/09/2026)

Toda semana termina com um artefato versionado no Git.

### Fase A — Reprodução e consolidação · Semana 1 (05–09/ago)

- [x] leitura das trajetórias sem MDAnalysis, por índices explícitos
- [x] descritores: raio de giro, RMSD com superposição de Kabsch, 12 diedros
- [x] mapas de Ramachandran por resíduo, séries temporais, mapa Rg × RMSD
- [ ] notebook `01_exercicio_molsim.ipynb` na forma do exercício, com as
      lacunas preenchidas e as Questões 1–7 respondidas por escrito
- [ ] mesma análise aplicada à trajetória de 300 K
- [ ] baseline com `sklearn.cluster.KMeans` sobre (Rg, RMSD), como na §7.6

**Entregável:** notebook completo + `figs/molsim/`. É o que se leva à primeira
reunião com o orientador.

### Fase B — K-means coerente · Semanas 2–3 (10–23/ago)

- [x] implementação das três variantes com detecção de ciclo limite
- [x] medida de convergência, ciclos e monotonicidade
- [ ] testes unitários: média circular, imagem mínima, minimizador de cada métrica
- [ ] baselines de comparação: GMM, DBSCAN, aglomerativo (Ward)
- [ ] concordância entre variantes por ARI; estabilidade sob re-inicialização
- [ ] seleção de k por joelho, não por máximo de silhueta

**Entregável:** `molsim/kmeans_variants.py` testado + notebook `02_kmeans.ipynb`
com a tabela comparativa. É o entregável técnico principal.

### Fase C — Critério cinético de estado · Semanas 3–4 (17–30/ago)

- [ ] tempos de residência e matriz de transição em função do tempo de atraso
- [ ] teste de consistência: os estados sobrevivem ao aumento do lag?
- [ ] comparação 500 K × 300 K — os estados a 300 K devem ser mais separados
- [ ] KNN treinado nos rótulos de 500 K aplicado à trajetória de 300 K
      (tarefa opcional 11 do MolSim; exige renomear os átomos do arquivo)
- [ ] resposta fundamentada à Questão 7: quantos estados, e sob qual critério

**Entregável:** notebook `03_cinetica.ipynb`.

> **31/08 — encerramento formal da vigência da bolsa.** Confirmar com a PRP o
> que precisa estar protocolado até esta data.

### Fase D — Dados próprios · Semanas 5–6 (31/ago–13/set)

A trajetória fornecida tem 2000 frames a uma única temperatura. Gerar dados
próprios é o que permite dizer algo que o exercício não pode.

- [ ] ambiente `md-ml` (Miniforge + OpenMM/CUDA) — ver §6.1
- [ ] estender `src/build_peptide.py` para a capa amida C-terminal (`NH2`),
      hoje só produz `NME`
- [ ] produção do hexâmero em campo de força all-atom Amber, 3 réplicas,
      a 300 K e 500 K
- [ ] aplicar o protocolo de relaxação a 0 K dos frames, para casar com o
      protocolo do material de referência
- [ ] repetir as Fases B e C sobre os dados novos, com estatística muito maior

**Decisão registrada:** os dados próprios usam Amber all-atom, não o átomo
unido GROMOS do material original. A comparação entre os dois conjuntos se dá
no nível dos descritores e dos estados, não do campo de força.

**Entregável:** trajetórias próprias + `notebook 04_dados_proprios.ipynb`.

### Fase E — Análise física e figuras · Semana 7 (14–20/set)

- [ ] energias livres relativas ΔG = −k_B T ln(n_i/N), populações, residências
- [ ] estruturas medoides renderizadas (VMD / VESTA / nglview)
- [ ] nomeação estrutural dos estados (α_R, β/PPII, α_L, voltas)
- [ ] figuras finais em 300 dpi

### Fase F — Escrita · Semanas 8–9 (21/set–30/set)

- [ ] Metodologia e Resultados primeiro; Introdução e Discussão depois
- [ ] enviar ao orientador com no mínimo 5 dias de antecedência
- [ ] resumo para o congresso de IC
- [ ] `README.md` de reprodução e ensaio da apresentação

---

## 6. Infraestrutura

### 6.1 Ambiente

**Fases A a C não exigem instalação nenhuma.** numpy, scipy, scikit-learn,
pandas e matplotlib já funcionam no Python do sistema, e
[molsim/data.py](molsim/data.py) lê os `.xyz` sem MDAnalysis.

O ambiente conda só é necessário a partir da Fase D, para o OpenMM:

```bash
conda env create -f environment.yml
conda activate md-ml
python -m openmm.testInstallation      # precisa listar CUDA
```

MDAnalysis e nglview entram junto, e passam a ser opcionais para a análise
(úteis para visualização e para conferir os descritores contra uma
implementação independente).

### 6.2 Estrutura do repositório

```
molsim/                 análise do exercício MolSim (Fases A–C)
  data.py               leitura .xyz, diedros, raio de giro, RMSD
  kmeans_variants.py    as três variantes + diagnóstico de coerência
  analyze.py            pipeline completo ponta-a-ponta
  experiment_coherence.py   experimento de convergência
src/                    simulação própria em OpenMM (Fase D)
  build_peptide.py      construção de peptídeos por coordenadas internas
  config.py, simulate.py
notebooks/              00 (setup OpenMM), 01–04 (análise)
tests/                  regressão geométrica e numérica
figs/molsim/            figuras geradas
```

### 6.3 Onde ficam os dados

As trajetórias próprias da Fase D vão para `C:/md-data/3-dinamica-molecular`
(configurável por `MD_DATA_ROOT`), **fora do OneDrive** — a sincronização de
alguns GB de DCD durante a produção trava a máquina.

---

## 7. Organização e acompanhamento

**Pauta da reunião de alinhamento:** apresentar o resultado da §3.3 (o defeito
de convergência e a correção), o pipeline reproduzido da Fase A, e a
constatação da §4 sobre a insuficiência do critério geométrico. Propor as
Fases C e D como o desdobramento natural.

**Definições administrativas a fechar:**

1. data-limite do relatório final na PRP e o que precisa estar protocolado até 31/08
2. prazo de submissão de resumo do congresso de IC da UFLA
3. haverá pedido de renovação / novo plano para 2026–2027?
4. há acesso a cluster do departamento, ou só a máquina local? (o plano se
   fecha usando apenas a local)

**Cadência:** reunião semanal curta com um artefato concreto na mão e e-mail de
status de 5 linhas toda sexta.

---

## 8. Riscos

| Risco | Prob. | Mitigação |
|---|---|---|
| Nenhum k tem justificativa física clara | **Alta** | já observado (§4); a Fase C troca o critério geométrico pelo cinético, e o resultado negativo é reportável |
| Trajetória de 500 K curta demais para estatística de estados | Alta | Fase D gera dados próprios; até lá, reportar intervalos de confiança |
| Variante `corda` não melhora a qualidade dos estados | Média | o ganho de convergência já está demonstrado e é independente disso |
| OpenMM/CUDA não instala no Windows | Média | Fase D é a única que depende dele; fallback para plataforma `CPU` ou WSL2 |
| Escopo inflar | Alta | congelar em 13/09; ideias novas vão para o backlog |

**Backlog explícito (trabalho futuro no relatório):** TICA e Markov State
Models completos com PCCA+, VAMPnets, metadinâmica, peptídeos maiores.

---

## 9. Definição de "pronto"

Em 30/09/2026:

1. exercício MolSim reproduzido integralmente, com as Questões 1–7 respondidas
2. implementação própria e testada das três variantes de K-means periódico
3. demonstração quantitativa do defeito de convergência e da correção
4. comparação contra ≥3 métodos de baseline
5. análise cinética dos estados (residência, transições, dependência do lag)
6. trajetórias próprias do mesmo sistema, com estatística maior
7. interpretação física dos estados, com estruturas representativas
8. relatório final submetido e resumo de congresso

Cobre os cinco itens do cronograma registrado no SIGAA.
