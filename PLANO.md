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

A ordem dos átomos é a mesma nos dois arquivos e é regular:

```
índice  0: CH3     metila da acetila
índice  1: C       carbonila da acetila
índice  2: O
resíduo i (i = 1..6), bloco começando em 3 + 6(i−1):
        N, H, CA, CB, C, O
índice 39: N       amida C-terminal
índices 40, 41: H, H
```

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

> Esta seção registra a **hipótese** e o programa de medida; confirmá-la ou
> derrubá-la é o trabalho da Fase B.

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

### 3.3 Programa de medida

O que precisa ser medido, sobre a trajetória de 500 K, para cada variante, com
muitas inicializações e vários valores de k:

- **taxa de convergência** — frações de execuções que param por estabilização
  dos rótulos
- **ocorrência de ciclo limite** — execuções que nunca convergem, detectadas
  guardando as partições já visitadas: partição repetida significa algoritmo
  preso
- **monotonicidade do objetivo** — num par (métrica, atualização) coerente, o
  objetivo não pode subir entre iterações, por construção. Qualquer violação
  indica incoerência ou bug
- **fração dos dados perto da descontinuidade ±180°** — determina se o
  problema é típico ou patológico neste sistema

**Cuidado na comparação:** variantes que otimizam objetivos diferentes não são
comparáveis pela inércia de uma só delas. Uma variante que minimiza a distância
de corda parece pior medida em imagem mínima, e vice-versa. A tabela final
precisa deixar explícito qual objetivo está sendo reportado, e a conclusão
precisa separar **convergência** (propriedade do algoritmo) de **qualidade dos
estados** (propriedade física, que só a Fase C decide).

---

## 4. Segunda questão: geometria basta?

A Questão 7 do exercício pede o número de estados metaestáveis da molécula. Há
razão para desconfiar de que o critério puramente geométrico não determine esse
número, e verificar isso é o objetivo da Fase C:

- **O cotovelo e a silhueta podem não ter patamar.** Se a curva de silhueta
  subir sem estabilizar, escolher k pelo seu máximo é ajuste ao ruído, não
  medida física. Verificar se existe joelho, e onde.
- **"Metaestável" é propriedade cinética, não geométrica.** Um estado só é
  metaestável se a molécula permanecer nele por tempo longo comparado ao tempo
  de trânsito. Medir os tempos de residência **em unidades de frames**: se um
  estado dura poucos frames, ele não é metaestável em sentido útil, por melhor
  que seja o agrupamento geométrico.
- **A temperatura importa.** A trajetória de 500 K foi gerada justamente para
  deixar a molécula flexível. Os estados a 300 K devem estar mais separados —
  comparar as duas é o teste direto dessa hipótese.

Se a conclusão for que o número de estados não é bem definido geometricamente,
isso é **resultado**, não fracasso — e é o que justifica as Fases C e D.

---

## 5. Cronograma (05/08 → 30/09/2026)

Toda semana termina com um artefato versionado no Git.

### Fase A — Reprodução e consolidação · Semana 1 (05–09/ago)

Trabalhar dentro do notebook do exercício, com MDAnalysis e nglview.

- [ ] carregar e visualizar a trajetória (§7.2)
- [ ] raio de giro e RMSD, com os gráficos e o mapa Rg × RMSD (§7.3)
- [ ] os 12 diedros no dataframe (§7.4)
- [ ] análise exploratória, histogramas, Ramachandran por resíduo (§7.5)
- [ ] K-means do scikit-learn sobre (Rg, RMSD) e gráfico de cotovelo (§7.6, §7.7)
- [ ] mesma análise aplicada à trajetória de 300 K
- [ ] Questões 1 a 5 respondidas por escrito

**Entregável:** `notebooks/01_exercicio_molsim.ipynb` + figuras. É o que se
leva à primeira reunião com o orientador.

### Fase B — K-means coerente · Semanas 2–3 (10–23/ago)

- [ ] portar leitura e descritores para módulos em `molsim/`, com testes
- [ ] distância periódica (§7.8) — o caso de teste do tutorial dá 28.284
- [ ] K-means escrito na mão, com os quatro passos da §7.8
- [ ] instrumentação: histórico do objetivo, contagem de iterações, detecção
      de ciclo limite guardando as partições já visitadas
- [ ] as variantes que a análise da §3 sugerir, e a medida comparativa
- [ ] baselines: GMM, DBSCAN, aglomerativo (Ward); concordância por ARI
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
src/                    simulação própria em OpenMM (Fase D)
  build_peptide.py      construção de peptídeos por coordenadas internas
  config.py, simulate.py
notebooks/              notebooks de análise
tests/                  testes
figs/                   figuras geradas
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
