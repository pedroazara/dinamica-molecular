"""K-means para features periódicas.  [ESQUELETO — Fase B]

É aqui que está a contribuição original da IC. Ver ROTEIRO.md, tarefas B2 a B4.

O ponto de partida é a §7.8 do tutorial: escrever o K-means na mão para poder
trocar a métrica de distância, já que ângulos diedros são periódicos.

A investigação da B4 parte de uma pergunta sobre esse algoritmo:

    O método de Lloyd converge porque os dois passos reduzem o MESMO objetivo.
    O passo de atribuição reduz ao mover cada ponto para o centro mais próximo.
    O passo de atualização reduz porque o novo centroide é o MINIMIZADOR da
    soma das distâncias dos seus membros.

    A §7.8 usa distância periódica na atribuição e média aritmética na
    atualização. A média aritmética minimiza a distância euclidiana.
    Ela minimiza a distância periódica?

    Se não minimizar, a garantia de convergência ainda vale?

Responder isso — com medida, não com argumento — é o entregável da Fase B.

Alvo: `pytest tests/test_molsim.py -k kmeans`
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BOX = 360.0


# --------------------------------------------------------------------------
# B2 — distância
# --------------------------------------------------------------------------
def distance(x1: np.ndarray, x2: np.ndarray, lbox: float = BOX) -> float:
    """Distância euclidiana entre dois vetores, com condições periódicas.

    A receita está na §7.8 do tutorial:

        d = x1 − x2
        d = d − L · round(d / L)
        r = |d|

    O teste do tutorial: `distance([10, 170], [30, -170], 360)` deve dar
    **28.284**. Sem a correção periódica daria 340,6 — a diferença entre
    entender 170° e −170° como vizinhos ou como opostos.
    """
    raise NotImplementedError("Fase B2")


# --------------------------------------------------------------------------
# B4 — as variantes
# --------------------------------------------------------------------------
def centroid(members: np.ndarray, rule: str) -> np.ndarray:
    """Centroide de um conjunto de ângulos, segundo a regra escolhida.

    `rule` identifica qual regra de atualização usar. A da §7.8 é a média
    aritmética; a investigação da B4 é sobre quais outras fazem sentido e por
    quê.

    O teste `test_centroide_e_minimizador` não fixa a fórmula — ele verifica a
    PROPRIEDADE que um centroide precisa ter para o Lloyd convergir: ser o
    minimizador da soma das distâncias aos membros, na métrica correspondente.
    """
    raise NotImplementedError("Fase B4")


@dataclass
class KMeansResult:
    """Resultado de uma execução, com o que é preciso para diagnosticar.

    Guarde o histórico do objetivo e o número de iterações desde a B3 — sem
    isso não dá para responder a pergunta da B4.
    """
    centers: np.ndarray
    labels: np.ndarray
    n_iter: int
    converged: bool
    cycled: bool          # entrou em ciclo limite: nunca vai convergir
    objective: np.ndarray  # valor do objetivo a cada iteração
    inertia: float


def kmeans_angular(X_deg: np.ndarray, k: int, variant: str,
                   seed: int | None = 0, max_iter: int = 500) -> KMeansResult:
    """K-means sobre features angulares em graus.

    Os quatro passos, conforme a §7.8:

        1. inicializar os centroides em pontos de dados sorteados
        2. atribuir cada ponto ao centroide mais próximo
        3. recalcular os centroides a partir dos membros
        4. repetir 2–3 até convergir

    **Detecção de ciclo limite:** um K-means incoerente pode nunca convergir,
    alternando indefinidamente entre partições. Para distinguir "ainda não
    convergiu" de "nunca vai convergir", guarde as partições já visitadas — se
    uma se repete, o algoritmo está preso num ciclo. Sem isso você só vê
    "bateu no max_iter" e não sabe o motivo.

    `variant` seleciona a combinação (métrica de atribuição, regra de
    atualização) a ser testada.
    """
    raise NotImplementedError("Fase B3 / B4")


def monotonicity_violations(history: np.ndarray) -> int:
    """Quantas vezes o objetivo SUBIU ao longo das iterações.

    Num Lloyd coerente esse número é exatamente 0, por construção. Qualquer
    valor maior que zero significa que a atualização não está minimizando a
    métrica da atribuição — ou que existe um bug.
    """
    raise NotImplementedError("Fase B4")
