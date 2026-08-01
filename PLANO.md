# Plano de Execução — IC PIBIC/CNPq
### Aplicação de Métodos de Aprendizado de Máquina na Análise de Simulações de Dinâmica Molecular

**Discente:** Pedro Henrique Ázara de Almeida (202210612)
**Orientador:** Prof. Igor Saulo Santos de Oliveira (DFI/UFLA)
**Vigência da bolsa:** 01/09/2025 – 31/08/2026
**Horizonte deste plano:** 01/08/2026 → 30/09/2026 (9 semanas)

---

## 1. Definição científica do trabalho

### 1.1 Pergunta de pesquisa

> Métodos de agrupamento não supervisionado aplicados a descritores estruturais extraídos de trajetórias de dinâmica molecular conseguem identificar automaticamente os estados metaestáveis de um oligopeptídeo? E uma métrica de distância que respeite a geometria periódica do espaço de ângulos diedros melhora essa identificação em relação ao K-means euclidiano padrão?

A segunda pergunta é a **contribuição original** do trabalho — é ela que corresponde ao item "desenvolvimento de versão personalizada do K-means" do cronograma. Ver §1.5.

### 1.2 Sistemas moleculares (dois níveis)

**Sistema A — dipeptídeo de alanina (ACE-ALA-NME), sistema de validação.**
22 átomos. É o sistema de referência clássico da amostragem conformacional. O espaço conformacional é essencialmente 2D (φ, ψ), com bacias metaestáveis bem estabelecidas na literatura: C7eq/α_R, C5/β, α_L/C7ax e P_II.

*Por que ele é indispensável:* como o espaço é 2D, dá para **plotar o mapa de Ramachandran e verificar visualmente** se o clustering acertou. Ele funciona como controle do pipeline — qualquer erro de featurização, normalização ou métrica aparece aqui de forma barata e rápida, antes de escalar.

**Sistema B — Ala10 (deca-alanina), sistema de aplicação.**
112 átomos (ACE-(ALA)₁₀-NME), solvente implícito GBn2. Exibe transições hélice ↔ novelo em escala de ns–µs. Aqui o raio de giro e o RMSD (previstos no plano de trabalho) passam a ser descritores informativos, e o espaço conformacional é alto-dimensional o suficiente para que o agrupamento seja genuinamente necessário — não é mais possível "ver" a resposta.

*Sistema C (opcional, apenas se houver folga):* chignolin CLN025 (10 resíduos, dobramento reversível). Não deve ser planejado como dependência.

### 1.3 Protocolo de simulação

| Parâmetro | Valor | Observação |
|---|---|---|
| Motor | **OpenMM** (CUDA) | Escolhido sobre o LAMMPS: API Python nativa, integra direto com MDTraj/scikit-learn. O plano prevê "OpenMM ou LAMMPS" |
| Campo de força | `amber14-all.xml` | padrão para peptídeos |
| Solvente | A: TIP3P explícito (~1500 águas) · B: implícito `implicit/gbn2.xml` | implícito no B acelera a amostragem de transições em ~1 ordem de grandeza |
| Ensemble | NVT | conforme o plano |
| Termostato | `LangevinMiddleIntegrator`, γ = 1 ps⁻¹, T = 300 K | conforme o plano |
| Passo de tempo | 2 fs com `constraints=HBonds` | o plano cita 1 fs; 2 fs com restrição de ligações X–H é equivalente e dobra a amostragem efetiva. **Justificar no relatório** |
| Duração | A: 3 × 500 ns · B: 3 × 1 µs (réplicas independentes) | réplicas independentes ≠ uma trajetória longa; permitem teste de convergência |
| Gravação | a cada 10 ps | → 50k frames (A) e 100k frames (B) por réplica |
| Minimização | antes de cada produção; equilibração NVT de 1 ns descartada | |

**Sobre a relaxação a 0 K prevista no plano:** o plano de trabalho menciona relaxar as configurações por minimização de energia para eliminar flutuações térmicas de curto alcance. Isso custa uma minimização por frame e, para descritores baseados em diedros, é largamente redundante — a suavização já vem da média temporal. **Proposta:** aplicar apenas a um subconjunto (por exemplo, os medoides de cada cluster) e demonstrar quantitativamente que o rótulo do cluster não se altera. Isso vira uma seção de robustez no relatório, em vez de um custo computacional sem retorno.

### 1.4 Descritores (features)

Três blocos, tratados separadamente e depois combinados:

1. **Diedros** — φ, ψ de cada resíduo (e χ₁ quando existir), via `mdtraj.compute_phi/psi`.
   **Ponto crítico:** ângulos são periódicos. Um φ = −179° e um φ = +179° são conformacionalmente vizinhos, mas distam 358 unidades no espaço euclidiano. Usar o ângulo cru como feature do k-means é um erro conceitual frequente — e é exatamente aí que se insere a contribuição deste trabalho.
   *Tratamento padrão:* embedding sin/cos → cada ângulo vira duas features.
   *Tratamento proposto:* métrica angular direta (§1.5).
2. **Raio de giro** (`mdtraj.compute_rg`) — descritor global de compactação. Essencial no Sistema B.
3. **RMSD** em relação à estrutura inicial e a uma referência de hélice-α ideal (`mdtraj.rmsd`), após alinhamento de corpo rígido.
   *(O plano se refere a "deslocamento quadrático médio (MSD)". Em contexto conformacional o descritor usado é o RMSD estrutural, não o MSD difusivo de Einstein. Registrar essa distinção explicitamente no relatório.)*

Normalização: `StandardScaler`. Analisar e reportar o efeito de escalonar blocos com unidades diferentes (nm vs. adimensional) — isso altera os agrupamentos resultantes.

### 1.5 A versão personalizada do K-means (núcleo original)

Proposta concreta, em ordem crescente de ambição — **implementar (i), tentar (ii), citar (iii) como trabalho futuro**:

**(i) K-means toroidal / circular.**
Substituir a distância euclidiana por uma distância angular sobre o toro:

    d(θ, μ) = Σ_k [1 − cos(θ_k − μ_k)]

e o centroide pela **média circular**:

    μ_k = atan2( ⟨sin θ_k⟩ , ⟨cos θ_k⟩ )

É uma modificação genuína do algoritmo de Lloyd — atinge tanto o passo de atribuição quanto o de atualização — motivada pela topologia real do espaço conformacional, implementável em ~80 linhas de NumPy e testável de forma limpa contra o k-means euclidiano usando as bacias conhecidas do Sistema A como gabarito. **É o entregável técnico principal do trabalho.**

**(ii) K-means com informação cinética (time-lagged).**
"Metaestável" é uma propriedade *cinética* (tempo de residência), não puramente geométrica. Projetar as features em coordenadas TICA (*time-lagged independent component analysis*, via `deeptime`) antes de agrupar alinha os clusters aos processos lentos do sistema. Comparar com o agrupamento puramente geométrico.

**(iii) Restrição de tempo mínimo de residência.**
Pós-processamento que suprime atribuições de duração inferior a τ (recruzamentos espúrios da barreira). Melhora substancialmente a interpretabilidade da matriz de transições.

**Baselines obrigatórios para comparação:** k-means euclidiano (com e sem embedding sin/cos), GMM, DBSCAN, aglomerativo (Ward). Sem baseline não há como afirmar que a versão customizada é superior.

### 1.6 Métricas e validação

- **Seleção de k:** cotovelo (inércia), silhueta, Calinski–Harabasz, Davies–Bouldin. Os quatro, na mesma figura, sobre o mesmo intervalo de k.
- **Validação física (o critério que realmente decide):**
  - populações dos clusters → energia livre relativa ΔG = −k_B T ln(p_i / p_max)
  - tempos médios de residência por estado
  - matriz de transições entre clusters
  - estrutura medoide de cada cluster, renderizada (VMD / PyMOL / `nglview`)
  - sobreposição dos clusters no mapa de Ramachandran (Sistema A) — **a figura central do trabalho**
- **Reprodutibilidade:** seed fixa, `n_init ≥ 10`, concordância entre as 3 réplicas independentes.

---

## 2. Cronograma semanal (01/08 → 30/09/2026)

> Toda semana termina com um artefato versionado no Git.

### FASE 0 — Fundação (Semana 1: 01–07/ago)

**Meta: fechar o loop ponta-a-ponta com uma trajetória real.**

- [ ] Instalar Miniforge (o Python 3.14 do sistema não serve — OpenMM/MDTraj não têm builds para 3.14). Criar env `md-ml` com Python 3.11 (comandos em §3.1)
- [ ] Validar CUDA: `python -m openmm.testInstallation` deve listar a plataforma CUDA
- [ ] `git init` no repositório; estrutura de diretórios de §3.2
- [ ] Rodar Sistema A por **1 ns** apenas para fechar o ciclo completo (build → simular → salvar DCD → abrir no MDTraj → plotar φ/ψ)
- [ ] Disparar as 3 réplicas de 500 ns do Sistema A em background

**Entregável:** `notebooks/00_setup.ipynb` + primeiro mapa de Ramachandran salvo em PNG.

### FASE 1 — Features e exploração (Semana 2: 08–14/ago)

- [ ] Módulo `src/features.py`: diedros, Rg, RMSD → matriz `(n_frames, n_features)`
- [ ] Análise exploratória: histogramas, séries temporais, matriz de correlação, PCA
- [ ] Verificação de convergência entre réplicas
- [ ] Iniciar as réplicas de 1 µs do Sistema B (rodam em background pelas duas semanas seguintes)

**Entregável:** `notebooks/01_features.ipynb` + `data/features_sysA.npz`

### FASE 2 — Clustering baseline (Semana 3: 15–21/ago)

- [ ] k-means padrão (scikit-learn) sobre as features do Sistema A
- [ ] Varredura k = 2..12 com as quatro métricas de §1.6
- [ ] Sobreposição dos clusters no mapa de Ramachandran → comparação com as bacias da literatura
- [ ] GMM, DBSCAN e Ward como baselines

**Entregável:** `notebooks/02_clustering_baseline.ipynb` — primeiro conjunto de resultados apresentável.

### FASE 3 — K-means customizado (Semanas 4–5: 22/ago – 04/set)

- [ ] `src/circular_kmeans.py` — implementação do algoritmo de §1.5(i), com testes unitários (dados sintéticos no toro com resposta conhecida)
- [ ] Benchmark quantitativo customizado vs. euclidiano: silhueta com métrica angular, ARI contra os rótulos de bacia da literatura, estabilidade sob re-inicialização
- [ ] Aplicação ao Sistema B
- [ ] (se houver folga) TICA + comparação — §1.5(ii)

**Entregável:** `src/circular_kmeans.py` testado + `notebooks/03_custom_kmeans.ipynb` com a tabela comparativa.

> **Marco 31/08 — encerramento formal da vigência da bolsa.** Confirmar com a PRP o que precisa estar protocolado até esta data.

### FASE 4 — Análise física (Semana 6: 05–11/set)

- [ ] Populações, energias livres relativas, tempos de residência, matriz de transições
- [ ] Extração e renderização dos medoides
- [ ] Interpretação estrutural: nomear cada estado metaestável (α_R, β, α_L, hélice, novelo…)
- [ ] Figuras finais em qualidade de publicação (300 dpi, vetorial quando possível)

**Entregável:** `notebooks/04_analise_fisica.ipynb` + pasta `figs/` fechada

### FASE 5 — Escrita (Semanas 7–8: 12–25/set)

- [ ] Semana 7: esqueleto do relatório + Metodologia e Resultados
- [ ] Semana 8: Introdução, Discussão, Conclusão, referências (Zotero/BibTeX). Enviar ao orientador com no mínimo 5 dias de antecedência
- [ ] Resumo para o congresso de IC

**Entregável:** relatório final em PDF, submetido

### FASE 6 — Divulgação (Semana 9: 26–30/set)

- [ ] Pôster/slides
- [ ] `README.md` do repositório com instruções de reprodução
- [ ] Ensaio da apresentação com o orientador
- [ ] Registro do que ficou fora do escopo → semente para renovação/TCC

---

## 3. Infraestrutura

### 3.1 Ambiente (Semana 1)

O Python 3.14 instalado em `C:\Python314` não atende — OpenMM, MDTraj e MDAnalysis não distribuem wheels para 3.14, e o OpenMM é distribuído primariamente via conda-forge. Instalar o Miniforge e criar um ambiente isolado:

```bash
conda create -n md-ml python=3.11 -y
conda activate md-ml
conda install -c conda-forge openmm cudatoolkit mdtraj mdanalysis scikit-learn numpy scipy matplotlib seaborn pandas jupyterlab nglview deeptime -y
python -m openmm.testInstallation
```

O último comando precisa listar `CUDA` entre as plataformas e mostrar erro de energia da ordem de 1e-6 entre elas. Resolver isso no primeiro dia — todo o cronograma de produção depende dele.

### 3.2 Estrutura do repositório

```
3-dinamica-molecular/
├── README.md
├── environment.yml
├── data/                  # .gitignore — trajetórias são grandes
│   ├── raw/               # DCD/XTC brutos
│   └── processed/         # features .npz
├── src/
│   ├── simulate.py        # construção do sistema + produção (OpenMM)
│   ├── features.py        # diedros, Rg, RMSD
│   ├── circular_kmeans.py # ★ contribuição original
│   ├── metrics.py         # silhueta, CH, DB, ARI
│   └── analysis.py        # populações, ΔG, tempos de residência
├── notebooks/             # 00 a 04, na ordem do cronograma
├── figs/
├── tests/
└── relatorio/             # LaTeX
```

Commits diários. `data/raw/` fora do versionamento (1 µs de Ala10 gravado a cada 10 ps ocupa alguns GB). Manter as trajetórias **fora da pasta sincronizada do OneDrive** — sincronizar GB de DCD durante as simulações trava a máquina.

### 3.3 Orçamento computacional

Hardware disponível: RTX 4060 Ti, Ryzen 5 5600G, 32 GB RAM. Estimativas conservadoras em OpenMM/CUDA:

| Sistema | Tamanho | Taxa estimada | 1 réplica | 3 réplicas |
|---|---|---|---|---|
| A — Ala dipeptídeo (TIP3P) | ~5k átomos | ~300–500 ns/dia | 500 ns → ~1,5 dia | ~4 dias (sequencial) |
| B — Ala10 (implícito) | ~109 átomos | ~1–3 µs/dia | 1 µs → ~1 dia | ~3 dias |

Toda a produção cabe em cerca de uma semana de máquina dedicada. Por isso as simulações são disparadas nas Semanas 1–2 e ficam em background enquanto o pipeline de análise é desenvolvido — o caminho crítico do projeto é o código e a escrita, não a GPU.

---

## 4. Organização e acompanhamento

**Reunião de alinhamento (esta semana).** Levar uma página com as decisões metodológicas propostas, para validação: (a) escolha do OpenMM sobre o LAMMPS; (b) os dois sistemas moleculares e a função de cada um; (c) o k-means circular como contribuição original; (d) este cronograma. A divisão natural de papéis é: o pipeline computacional é proposto aqui, a interpretação física é validada pelo orientador.

**Definições administrativas a fechar na mesma reunião** — elas ancoram as datas reais do cronograma:

1. Data-limite exata do relatório final na PRP, e o que precisa estar protocolado até 31/08
2. Prazo de submissão de resumo do congresso de IC da UFLA
3. Haverá pedido de renovação / novo plano para 2026–2027? (define o que vale deixar em aberto no backlog de §5)
4. Há acesso a cluster do departamento, ou o trabalho roda apenas na máquina local? (o plano acima se fecha usando somente a local)

**Cadência:** reunião semanal curta com um artefato concreto (figura, notebook, tabela) e e-mail de status de 5 linhas toda sexta.

---

## 5. Riscos e contingências

| Risco | Probabilidade | Mitigação |
|---|---|---|
| OpenMM/CUDA não instala no Windows | Média | Fallback para plataforma `CPU` (Ryzen 5600G, 6 núcleos) — Sistema A ainda roda em ~1 dia. Alternativa: WSL2 + conda |
| Amostragem insuficiente (poucas transições observadas) | **Alta** | Aumentar T para 350 K (acelera transições sem deslocar as bacias); usar solvente implícito; mais réplicas curtas em vez de uma longa |
| K-means circular não supera o euclidiano | Média | **Continua sendo resultado válido.** Um resultado negativo bem quantificado vale mais que um positivo mal medido — o relatório deve ser estruturado desde o início para acomodar as duas conclusões |
| Escopo inflar durante a execução | Alta | Congelar escopo em 04/09. Ideias novas vão para o backlog abaixo, não para o cronograma |
| Trajetórias estourarem o OneDrive | Alta | `data/` fora da pasta sincronizada desde o primeiro dia |

**Backlog deliberadamente fora do escopo** (mencionar como trabalho futuro no relatório): VAMPnets/redes neurais para coordenadas de reação, Markov State Models completos com PCCA+ e teste de Chapman–Kolmogorov, metadinâmica/umbrella sampling, sistemas maiores (Trp-cage, villin).

---

## 6. Definição de "pronto"

Em 30/09/2026 devem existir:

1. Repositório Git reprodutível, com `environment.yml` e README
2. Trajetórias de DM de dois sistemas, com controle de convergência
3. Pipeline de featurização documentado (diedros, Rg, RMSD)
4. Implementação própria e testada do k-means circular
5. Comparação quantitativa contra ≥3 métodos de baseline
6. Interpretação física dos estados metaestáveis, com estruturas representativas
7. Relatório final submetido
8. Resumo/pôster de congresso

Cobre os cinco itens do cronograma de atividades registrado no SIGAA.
