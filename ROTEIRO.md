# Roteiro de desenvolvimento

Este é o caminho a percorrer. O [PLANO.md](PLANO.md) diz *o que* o projeto é e
*quando* cada coisa acontece; este arquivo diz *como* avançar, tarefa por
tarefa.

**Cada tarefa tem quatro campos:**

- **Objetivo** — o que precisa existir ao final
- **Entregável** — o arquivo concreto que fica versionado
- **Como saber que está certo** — a verificação. Esta é a parte que importa:
  em análise de trajetória quase todo erro é silencioso, produz números
  plausíveis e só aparece na banca
- **Armadilha** — o erro específico que costuma acontecer nessa tarefa

Marque `[x]` conforme avançar. Commit ao fim de cada tarefa.

---

## Referência rápida do sistema

**Ace-(Ala)₆-NH₂**, átomo unido, 42 sítios. A ordem dos átomos nos dois `.xyz`
é a mesma e é regular — confira você mesmo antes de confiar:

```
índice  0: CH3     metila da acetila
índice  1: C       carbonila da acetila
índice  2: O
resíduo i (i = 1..6), bloco começando em 3 + 6(i−1):
        N, H, CA, CB, C, O
índice 39: N       amida C-terminal
índices 40, 41: H, H
```

Trajetórias em `G:/Meu Drive/ufla/ic/exercises/`:

| arquivo | T | frames | gravação |
|---|---|---|---|
| `trajectory.xyz` | 500 K | 2000 | 10 ps |
| `trajectory_300K.xyz` | 300 K | 1000 | 10 ps |

---

# FASE A — Reproduzir o exercício MolSim

> Semana 1 (05–09/ago). Meta: exercício rodando de ponta a ponta, Questões 1 a 5
> respondidas por escrito.

Trabalhe **dentro do notebook do exercício** nesta fase
(`G:/Meu Drive/ufla/ic/exercises/molsim_ML_exercise.ipynb`). Copie-o para
`notebooks/01_exercicio_molsim.ipynb` no repositório antes de começar, para
versionar o seu progresso sem mexer no original.

### A0 — Ambiente

- [ ] Instalar Miniforge
- [ ] Criar o ambiente de análise:

```bash
conda create -n molsim -c conda-forge python=3.11 mdanalysis nglview scikit-learn pandas matplotlib jupyterlab -y
```

- [ ] Ler o capítulo 7 inteiro do PDF (10 páginas) antes de escrever qualquer código

**Como saber que está certo:** `import MDAnalysis` e `import nglview` sem erro.

**Armadilha:** o ambiente pesado com OpenMM (`environment.yml`) só é necessário
na Fase D. Não gaste 2 GB de download hoje.

---

### A1 — Carregar e visualizar a trajetória

Tarefas 1 e 2 do MolSim (§7.2).

- [ ] Carregar `trajectory.xyz` numa `Universe` do MDAnalysis com
      `to_guess=['bonds','angles','dihedrals','masses']` e `dt=10`
- [ ] Inspecionar `u.atoms`, `u.angles`, `u.dihedrals`
- [ ] Exibir a molécula com nglview
- [ ] **Questão 1:** qual a desvantagem de usar as coordenadas atômicas
      diretamente como entrada do agrupamento?

**Entregável:** `notebooks/01_exercicio_molsim.ipynb` com as células rodando.

**Como saber que está certo:** `len(u.trajectory)` = 2000, `len(u.atoms)` = 42.
A molécula na tela tem que parecer um peptídeo — cadeia com ramificações
laterais, não uma nuvem de átomos soltos.

**Armadilha:** se o widget do nglview não aparecer no Jupyter Lab, use
`jupyter notebook`. O próprio tutorial avisa disso.

---

### A2 — Descritores globais: raio de giro e RMSD

Tarefa 3, parte 1 (§7.3).

- [ ] Raio de giro por frame
- [ ] RMSD em relação ao frame 0
- [ ] Gráficos de cada um versus número do frame
- [ ] Mapa Rg × RMSD
- [ ] **Questão 2:** o que você observa nos dois?

**Entregável:** as quatro figuras no notebook.

**Como saber que está certo:**
- o raio de giro de um hexapeptídeo fica na casa de poucos ångströms — se der
  0,5 Å ou 50 Å, algo está errado na unidade ou na seleção de átomos
- o RMSD do frame 0 contra ele mesmo tem que dar **exatamente 0**
- o RMSD tem que ser invariante a translação e rotação. Teste: translade a
  molécula inteira por um vetor qualquer e confira que o RMSD não muda

**Armadilha:** RMSD sem superposição prévia mede translação e rotação do corpo
inteiro, não mudança de conformação. O `rms.RMSD` do MDAnalysis já superpõe;
se você implementar na mão, precisa alinhar antes.

---

### A3 — Descritores locais: os 12 ângulos diedros

Tarefa 3, parte 2 (§7.4).

- [ ] Identificar quais dos diedros da `Universe` são os φ e ψ do esqueleto
- [ ] Calcular os 12 para todos os frames
- [ ] Montar o `DataFrame` com `time`, `radgyr`, `rmsd` e os 12 diedros

**Entregável:** `df_features` no notebook, e salvo em CSV.

**Como saber que está certo:**
- todos os valores caem em (−180, 180]
- a maior parte da densidade tem que ficar em **φ < 0**. Alanina é
  L-aminoácido; se a nuvem estiver espelhada para φ > 0, os índices dos átomos
  no diedro estão na ordem errada
- compare seu Ramachandran com o da Wikipedia para alanina — as bacias
  ocupadas têm que bater

**Armadilha registrada:** a dica da §7.4 no PDF lista **11** diedros — falta o
índice 31. O notebook (célula 23) tem os 12 corretos. Confira a sua lista.

---

### A4 — Análise exploratória

Tarefa 4 (§7.5).

- [ ] `head()`, `describe()`, `info()`, checagem de NaN
- [ ] Grade 6×2: cada diedro versus tempo
- [ ] Grade 6×2: histograma de cada diedro
- [ ] Grade 2×3: φ versus ψ por resíduo (Ramachandran)
- [ ] Escrever no notebook: os seis resíduos amostram o mesmo espaço? Por quê?

**Entregável:** as três grades de figuras.

**Como saber que está certo:** resíduos das pontas (1 e 6) devem ser mais
livres que os do meio — têm menos vizinhos restringindo. Se todos parecerem
iguais, verifique se você não plotou o mesmo diedro seis vezes.

**Conceito:** a §7.5.1 explica por que o escalonamento de features é omitido
aqui — Rg e RMSD têm faixas parecidas entre si, e os diedros também. Entenda o
argumento, porque ele volta na Fase B quando você misturar os dois conjuntos.

---

### A5 — K-means de referência (scikit-learn)

Tarefa 5 (§7.6).

- [ ] `KMeans` sobre (Rg, RMSD) com k = 4
- [ ] Inspecionar `cluster_centers_`, `labels_`, contar membros
- [ ] Gráfico Rg × RMSD colorido por cluster, com os centroides marcados
- [ ] Repetir com vários k
- [ ] **Questão 3:** o que muda ao rodar o mesmo K-means várias vezes?
- [ ] **Questão 4:** dá para dizer algo sobre um bom valor de k?

**Como saber que está certo:** rodando duas vezes sem `random_state` fixo, os
rótulos trocam de número mas as partições devem ser parecidas. Se as partições
mudarem de verdade, o k está alto demais para os dados.

---

### A6 — Cotovelo e inspeção estrutural

Tarefas 7 e 8 (§7.7).

- [ ] Gráfico de cotovelo: `score()` versus k, de 2 a 15
- [ ] Escolher k num joelho da curva
- [ ] Extrair as estruturas de cada cluster para `.xyz`, alinhar com
      `align.AlignTraj`, visualizar no nglview
- [ ] **Questão 5:** as estruturas de um mesmo cluster se parecem?

**Entregável:** figura do cotovelo + inspeção visual documentada.

**Armadilha:** o centroide **não é** uma estrutura molecular — é uma média de
valores de features. Para "a estrutura típica do cluster" você precisa do
frame real mais próximo do centroide (medoide), não do centroide.

---

# FASE B — K-means para features periódicas

> Semanas 2–3 (10–23/ago). É o entregável técnico principal da IC.

A partir daqui saia do notebook do exercício e escreva **módulos** em
`molsim/`, com testes. Os esqueletos com as assinaturas estão em
[molsim/](molsim/); a suíte em `tests/test_molsim.py` é a especificação —
implemente até ela passar.

```bash
pytest -q
```

### B1 — Portar a leitura e os descritores para módulo

- [ ] `molsim/data.py` — ler `.xyz` sem depender do MDAnalysis
- [ ] `molsim/features.py` — diedros, raio de giro, RMSD

**Por que sair do MDAnalysis:** a lista de diedros que o `to_guess` infere
depende de raios de van der Waals e muda entre versões da biblioteca. Indexar
explicitamente torna o resultado reprodutível. Mantenha o MDAnalysis instalado
para **conferir** seus números contra uma implementação independente — se as
duas concordarem, os dois estão provavelmente certos.

**Como saber que está certo:** `pytest tests/test_molsim.py -k "leitura or descritor"`.

---

### B2 — A distância periódica

- [ ] Implementar a distância com convenção de imagem mínima, L = 360
- [ ] Testar com o caso do tutorial (§7.8): deve dar **28.284**

**Conceito:** a distância entre −170° e +170° é 20°, não 340°. Esse é o
problema todo.

---

### B3 — Escrever o K-means na mão

Tarefa 6 (§7.8), com o passo a passo listado no tutorial.

- [ ] Inicialização dos centroides em pontos de dados sorteados
- [ ] Passo de atribuição
- [ ] Passo de atualização
- [ ] Critério de parada — **Questão 6:** qual medida usar para convergência?
- [ ] Comparar o resultado com o do scikit-learn sobre (Rg, RMSD)

**Como saber que está certo:** sobre (Rg, RMSD), onde a periodicidade não
importa, sua implementação e a do scikit-learn têm que dar partições
equivalentes. Se não derem, o bug está na sua e não na periodicidade.

**Instrumente desde já:** guarde o histórico do objetivo a cada iteração e o
número de iterações até parar. Você vai precisar disso em B4.

---

### B4 — A investigação central

Aqui está a contribuição original da IC. **Hipótese a testar, não resultado
dado:**

> O algoritmo de Lloyd converge porque os dois passos reduzem o mesmo
> objetivo. A atualização só reduz se o centroide for o **minimizador** da
> métrica usada na atribuição. A §7.8 combina distância periódica com média
> aritmética. A média aritmética é o minimizador da distância euclidiana, não
> da periódica. **A garantia de convergência ainda vale?**

- [ ] Formular a pergunta: qual é o minimizador de Σ(1 − cos(θ − μ))? E o da
      distância de imagem mínima ao quadrado? São o mesmo?
- [ ] Implementar as variantes que essa análise sugerir
- [ ] Medir, com muitas inicializações e vários k: taxa de convergência,
      ocorrência de ciclo limite, monotonicidade do objetivo
- [ ] Detectar ciclo limite guardando as partições já visitadas — se uma
      partição se repete, o algoritmo está preso
- [ ] Quantificar quanto dos seus dados está perto da descontinuidade ±180°,
      para saber se o problema é típico ou patológico

**Entregável:** `molsim/kmeans_variants.py` + tabela comparativa + `notebooks/02_kmeans.ipynb`.

**Como saber que está certo:** para uma variante em que atribuição e
atualização sejam coerentes, o objetivo **nunca pode subir** entre iterações.
Se subir, ou a variante é incoerente ou tem bug — e distinguir os dois casos é
parte do trabalho.

**Armadilha:** comparar variantes com objetivos diferentes exige cuidado. Uma
variante que otimiza a distância de corda vai parecer pior se você medir com a
métrica de imagem mínima, e vice-versa. Deixe explícito qual objetivo está na
tabela.

---

### B5 — Baselines

- [ ] GMM, DBSCAN, aglomerativo (Ward) sobre o mesmo conjunto de features
- [ ] Concordância entre métodos por ARI (`sklearn.metrics.adjusted_rand_score`)
- [ ] Estabilidade de cada um sob re-inicialização
- [ ] Silhueta como segundo critério de seleção de k, além do cotovelo

**Armadilha:** silhueta calculada com distância euclidiana sobre ângulos crus
reintroduz exatamente o erro de periodicidade que você corrigiu no
agrupamento. A métrica da seleção de modelo tem que ser a mesma do algoritmo.

**Armadilha 2:** escolher k pelo **máximo** da silhueta ajusta ao ruído da
curva. Procure o joelho.

---

# FASE C — Critério cinético de estado

> Semanas 3–4 (17–30/ago). Tarefas opcionais 9, 10 e 11 do MolSim.

### C1 — Energias livres e transições

- [ ] ΔG(i) = −k_B T ln(n_i / N) para cada cluster
- [ ] Matriz de transição contando quantas vezes o rótulo i é seguido de j
- [ ] Grafo 2D dos estados, nós = estados, arestas = transições
- [ ] Tempo médio de residência por estado

**Como saber que está certo:** cada linha da matriz de transição soma 1.

---

### C2 — A pergunta que fecha o projeto

- [ ] Tempo de residência versus intervalo de gravação: quantos frames dura um
      estado? Se for da ordem de poucos frames, ele é metaestável de verdade?
- [ ] Refazer a matriz de transição com tempos de atraso crescentes. Os
      estados sobrevivem?
- [ ] Comparar 500 K com 300 K. Os estados a 300 K devem estar mais separados
- [ ] **Questão 7:** qual o melhor conjunto de features, e quantos estados
      metaestáveis a molécula tem? Sob qual critério — geométrico ou cinético?

**Este é o resultado científico da IC.** A resposta pode perfeitamente ser
"o critério geométrico não determina o número de estados" — resultado negativo
bem medido vale mais que positivo mal medido.

---

### C3 — KNN entre temperaturas

Tarefa opcional 11.

- [ ] Treinar KNN com os rótulos obtidos a 500 K
- [ ] Classificar a trajetória de 300 K
- [ ] Comparar populações e energias livres entre as duas temperaturas

**Armadilha registrada:** `trajectory_300K.xyz` usa nomes de tipo GROMOS
(`CH3`, `CH1`, `NH1`) em vez de símbolos de elemento. O `to_guess` do
MDAnalysis não infere elementos a partir desses nomes. Renomeie antes, ou use
o seu próprio leitor da B1.

---

# FASE D — Dados próprios

> Semanas 5–6 (31/ago–13/set).

A trajetória fornecida tem 2000 frames a uma única temperatura. Gerar dados
próprios é o que permite dizer algo que o exercício não pode.

- [ ] Ambiente `md-ml` com OpenMM/CUDA (`environment.yml`)
- [ ] `python -m openmm.testInstallation` listando CUDA
- [ ] Estender `src/build_peptide.py` para a capa amida C-terminal — hoje
      produz `NME`, o sistema do exercício usa `NH2`
- [ ] Produção do hexâmero, 3 réplicas, a 300 K e 500 K
- [ ] Relaxar os frames a 0 K, para casar com o protocolo do material original
- [ ] Repetir Fases B e C sobre os dados novos

A infraestrutura de simulação já está em [src/](src/) e é sua para usar —
`build_peptide.py`, `config.py`, `simulate.py`, com os testes de geometria em
`tests/test_build_peptide.py`.

---

# FASE E — Análise física · Semana 7 (14–20/set)

- [ ] Estruturas medoides renderizadas
- [ ] Nomeação estrutural dos estados (α_R, β/PPII, α_L, voltas)
- [ ] Figuras finais em 300 dpi

# FASE F — Escrita · Semanas 8–9 (21–30/set)

- [ ] Metodologia e Resultados primeiro
- [ ] Introdução e Discussão depois
- [ ] Enviar ao orientador com 5 dias de antecedência
- [ ] Resumo do congresso de IC

---

## As sete questões do exercício

Responda **por escrito**, no notebook. São elas que viram a Discussão do
relatório.

1. Desvantagem de usar coordenadas atômicas diretamente no agrupamento
2. O que se observa no raio de giro e no RMSD
3. O que muda ao repetir o mesmo K-means
4. O que se pode dizer sobre um bom valor de k
5. As estruturas de um mesmo cluster se parecem?
6. Que medida usar como critério de convergência
7. Melhor conjunto de features, e quantos estados metaestáveis existem

## Quando pedir ajuda

Peça revisão, explicação de conceito, ou ajuda a achar bug. O código é seu.
