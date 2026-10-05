"""
Representación de soluciones, operadores y evaluación.

Genotipo COMÚN a las tres metaheurísticas:

    genotype = (perm, orient)

      perm   : permutación de los índices de rueda (0 .. TOTAL_WHEELS-1).  El
               ORDEN define la secuencia de colocación en el decodificador
               (decisión de posicionamiento).
      orient : lista de longitud TOTAL_WHEELS.  orient[i] ∈ {0, 1, 2} es la
               orientación de la rueda i, o data.SKIP (-1) si NO se carga
               (decisión de selección).

El posicionamiento exacto lo fija el decodificador de puntos extremos, de modo
que las tres decisiones del enunciado quedan cubiertas y TODA solución evaluada
es factible por construcción.

Las tres metaheurísticas resuelven exactamente el mismo problema, con la misma
función objetivo y las mismas restricciones; sólo cambia el mecanismo de
búsqueda.

NOTA: este módulo sólo usa la biblioteca estándar de Python.
"""

import random

from . import data
from .geometry import Decoder, empty_space, utilisation


def initial_solution(rng, noise=0.25, p_skip=0.10):
    """Generador de soluciones iniciales, COMÚN a las tres metaheurísticas.

    El orden en que el decodificador intenta colocar las ruedas es la decisión
    que más influye en el aprovechamiento: colocar primero las ruedas grandes
    evita que el espacio quede fragmentado antes de que lleguen las de mayor
    volumen. Se genera por tanto un orden *casi decreciente* por volumen de
    cilindro, perturbado con un factor aleatorio para que las distintas
    ejecuciones (y los distintos individuos) no sean idénticos.

    `noise` controla la intensidad de la perturbación:
      - noise = 0   -> orden determinista por volumen decreciente;
      - noise = 1   -> orden casi aleatorio.

    Las tres metaheurísticas usan EXACTAMENTE este mismo generador, de modo que
    el punto de partida es comparable y la inicialización no favorece a ninguna.
    """
    n = data.TOTAL_WHEELS
    types = data.wheel_types()
    keys = []
    for i in range(n):
        v = types[i // data.AVAILABLE_PER_TYPE]["volume_m3"]
        keys.append((v * (1.0 + noise * rng.uniform(-1.0, 1.0)), i))
    keys.sort(key=lambda kv: -kv[0])
    perm = [i for _, i in keys]
    orient = []
    for _ in range(n):
        if rng.random() < p_skip:
            orient.append(data.SKIP)
        else:
            orient.append(rng.randrange(data.N_ORIENT))
    return (perm, orient)


def evaluate(genotype, decoder, want_placements=False):
    """Decodifica y devuelve un dict con la información de la solución.

    Claves: objective (espacio vacío, a minimizar), volume, utilisation,
    used, n_wheels, placements, feasible.
    """
    volume, used, placements = decoder.decode(genotype, want_placements)
    return {
        "objective": empty_space(volume),     # menor = mejor
        "volume": volume,
        "utilisation": utilisation(volume),
        "used": list(used),
        "n_wheels": sum(used),
        "placements": placements,
        "feasible": True,                     # garantizado por el decodificador
    }


# --------------------------------------------------------------------------- #
# Operadores del Algoritmo Genético
# --------------------------------------------------------------------------- #
def crossover_ox(parent1, parent2, rng):
    """Order Crossover (OX) para la permutación + cruce uniforme para las
    orientaciones (éstas se indexan por rueda, no por posición)."""
    perm1, orient1 = parent1
    perm2, orient2 = parent2
    n = len(perm1)
    a = rng.randrange(n)
    b = rng.randrange(n)
    if a > b:
        a, b = b, a
    child = [None] * n
    taken = set()
    for i in range(a, b + 1):
        child[i] = perm1[i]
        taken.add(perm1[i])
    fill = [g for g in perm2 if g not in taken]
    k = 0
    for i in range(n):
        if child[i] is None:
            child[i] = fill[k]
            k += 1
    orient = [orient1[i] if rng.random() < 0.5 else orient2[i] for i in range(n)]
    return (child, orient)


def mutate(genotype, rng, p_perm=0.15, p_orient=0.25, p_skip=0.10):
    """Mutación: intercambios en la permutación, cambio de orientación y
    conmutación cargar / no cargar."""
    perm = list(genotype[0])
    orient = list(genotype[1])
    n = len(perm)
    for i in range(n):
        if rng.random() < p_perm:
            j = rng.randrange(n)
            perm[i], perm[j] = perm[j], perm[i]
    for i in range(n):
        r = rng.random()
        if r < p_orient:
            orient[i] = rng.randrange(data.N_ORIENT)
        elif r < p_orient + p_skip:
            orient[i] = data.SKIP if orient[i] != data.SKIP else rng.randrange(data.N_ORIENT)
    return (perm, orient)


# --------------------------------------------------------------------------- #
# Vecindario (búsqueda local, usada por IG)
# --------------------------------------------------------------------------- #
def neighbor(genotype, rng):
    """Aplica UN movimiento aleatorio:
       - intercambiar dos posiciones de la permutación;
       - cambiar la orientación de una rueda;
       - activar / desactivar (SKIP) una rueda;
       - invertir un bloque de la permutación.
    """
    perm = list(genotype[0])
    orient = list(genotype[1])
    n = len(perm)
    move = rng.randrange(4)
    if move == 0:
        i = rng.randrange(n)
        j = rng.randrange(n)
        perm[i], perm[j] = perm[j], perm[i]
    elif move == 1:
        i = rng.randrange(n)
        orient[i] = rng.randrange(data.N_ORIENT)
    elif move == 2:
        i = rng.randrange(n)
        orient[i] = data.SKIP if orient[i] != data.SKIP else rng.randrange(data.N_ORIENT)
    else:
        a = rng.randrange(n)
        b = rng.randrange(n)
        if a > b:
            a, b = b, a
        perm[a:b + 1] = perm[a:b + 1][::-1]
    return (perm, orient)


def local_search(genotype, decoder, rng, max_no_improve=120, eval_fn=None):
    """Búsqueda local de PRIMER-MEJOR (first improvement).

    eval_fn permite inyectar el contador de evaluaciones para que la búsqueda
    local consuma presupuesto de forma contabilizada.
    """
    if eval_fn is None:
        def eval_fn(g):
            return evaluate(g, decoder)
    cur = genotype
    cur_eval = eval_fn(cur)
    cur_obj = cur_eval["objective"]
    no_improve = 0
    while no_improve < max_no_improve:
        cand = neighbor(cur, rng)
        cand_eval = eval_fn(cand)
        if cand_eval["objective"] < cur_obj - 1e-9:
            cur, cur_eval, cur_obj = cand, cand_eval, cand_eval["objective"]
            no_improve = 0
        else:
            no_improve += 1
    return cur, cur_eval
