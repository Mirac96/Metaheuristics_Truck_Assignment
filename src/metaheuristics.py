"""
Las tres metaheurísticas de la práctica, implementadas DESDE CERO
(sin bibliotecas de optimización ni paquetes de terceros).

  1. Genetic Algorithm   (GA) : búsqueda POBLACIONAL.  Selección por torneo,
                                cruce OX, mutación y elitismo.
  2. Harmony Search      (HS) : búsqueda basada en MEMORIA ARMÓNICA.  Cada
                                improvisación combina consideración de memoria
                                (HMCR), selección aleatoria y ajuste de tono
                                (PAR / ancho de banda bw); la memoria se
                                actualiza sustituyendo la peor armonía.
  3. Iterated Greedy     (IG) : búsqueda por DESTRUCCIÓN–CONSTRUCCIÓN iterada.
                                Elimina d ruedas de la solución y las
                                reinserta de forma voraz; aceptación por
                                mejora y reinicio desde la mejor solución.

Las tres resuelven EXACTAMENTE el mismo problema, con la misma representación,
la misma función objetivo (espacio vacío), el mismo decodificador y el MISMO
presupuesto de evaluaciones (max_evals) para garantizar una comparación justa.
"""

import time

from . import data, solution


class _Counter:
    """Contador de evaluaciones + registro de la curva de convergencia.

    Cada llamada decodifica UNA solución y cuenta como una evaluación.  Todas
    las metaheurísticas consumen presupuesto exclusivamente a través de este
    contador, de modo que el esfuerzo computacional es comparable.
    """

    __slots__ = ("decoder", "n_evals", "conv", "best_so_far")

    def __init__(self, decoder):
        self.decoder = decoder
        self.n_evals = 0
        self.conv = []
        self.best_so_far = float("inf")

    def __call__(self, genotype):
        ev = solution.evaluate(genotype, self.decoder)
        self.n_evals += 1
        if ev["objective"] < self.best_so_far:
            self.best_so_far = ev["objective"]
        self.conv.append(self.best_so_far)
        return ev

    def budget_left(self, max_evals):
        return self.n_evals < max_evals


# =========================================================================== #
# 1. ALGORITMO GENÉTICO  (GA)
# =========================================================================== #
def run_ga(cfg, rng, decoder, max_evals):
    """Algoritmo genético generacional con elitismo.

    Mecanismo de EXPLORACIÓN: mutación (intercambios, orientaciones, skip) y
    cruce OX entre individuos distintos.
    Mecanismo de EXPLOTACIÓN: selección por torneo (presión selectiva) y
    conservación de los mejores individuos (elitismo).
    """
    t0 = time.time()
    pop_size = cfg.get("pop", 40)
    p_cross = cfg.get("p_cross", 0.9)
    p_perm = cfg.get("p_perm", 0.15)
    p_orient = cfg.get("p_orient", 0.25)
    p_skip = cfg.get("p_skip", 0.10)
    tour_k = cfg.get("tournament", 3)
    elite = cfg.get("elite", 2)

    counter = _Counter(decoder)
    population = [solution.initial_solution(rng) for _ in range(pop_size)]
    evals = [counter(g) for g in population]
    objs = [e["objective"] for e in evals]

    while counter.budget_left(max_evals):
        order = sorted(range(pop_size), key=lambda i: objs[i])
        new_pop = []
        new_evals = []
        # ---- elitismo ----
        for k in range(min(elite, pop_size)):
            new_pop.append(population[order[k]])
            new_evals.append(evals[order[k]])

        def tournament():
            best_i = rng.randrange(pop_size)
            for _ in range(tour_k - 1):
                j = rng.randrange(pop_size)
                if objs[j] < objs[best_i]:
                    best_i = j
            return best_i

        while len(new_pop) < pop_size and counter.budget_left(max_evals):
            pa = tournament()
            pb = tournament()
            if rng.random() < p_cross:
                child = solution.crossover_ox(population[pa], population[pb], rng)
            else:
                child = (list(population[pa][0]), list(population[pa][1]))
            child = solution.mutate(child, rng, p_perm, p_orient, p_skip)
            ce = counter(child)
            new_pop.append(child)
            new_evals.append(ce)

        if len(new_pop) < pop_size:      # presupuesto agotado a medias
            for k in range(pop_size - len(new_pop)):
                new_pop.append(population[order[k]])
                new_evals.append(evals[order[k]])
        population = new_pop
        evals = new_evals
        objs = [e["objective"] for e in evals]

    best_idx = min(range(pop_size), key=lambda i: objs[i])
    return {
        "name": "GA",
        "best_genotype": population[best_idx],
        "best_eval": evals[best_idx],
        "n_evals": counter.n_evals,
        "time": time.time() - t0,
        "convergence": counter.conv,
        "config": dict(cfg),
    }


# =========================================================================== #
# 2. HARMONY SEARCH  (HS)
# =========================================================================== #
def _hs_improvise(harmony, rng, hmcr, par, bw, par_o, p_skip_rand):
    """Improvisa una nueva armonía a partir de una armonía de referencia.

    Para la PERMUTACIÓN:
      - con probabilidad (1 - HMCR) se sustituye el contenido de la posición i
        por una rueda elegida al azar (selección aleatoria).  Se implementa
        mediante un intercambio, lo que preserva la validez de la permutación.
      - con probabilidad PAR se aplica el AJUSTE DE TONO: intercambio con una
        posición situada a lo sumo `bw` posiciones (ancho de banda).

    Para las ORIENTACIONES (variable discreta indexada por rueda):
      - con probabilidad (1 - HMCR) se elige un valor aleatorio;
      - con probabilidad PAR_O se "afina" el valor: se pasa a una orientación
        vecina, se activa una rueda no cargada o se desactiva una cargada.
    """
    perm = list(harmony[0])
    orient = list(harmony[1])
    n = len(perm)
    n_or = data.N_ORIENT

    # --- consideración de memoria / selección aleatoria (permutación) ---
    for i in range(n):
        if rng.random() >= hmcr:
            j = rng.randrange(n)
            perm[i], perm[j] = perm[j], perm[i]

    # --- ajuste de tono sobre la permutación (ancho de banda bw) ---
    if bw > 0:
        for i in range(n):
            if rng.random() < par:
                j = i + rng.randint(1, bw)
                if j >= n:
                    j = n - 1
                perm[i], perm[j] = perm[j], perm[i]

    # --- orientaciones ---
    for w in range(n):
        if rng.random() >= hmcr:
            orient[w] = data.SKIP if rng.random() < p_skip_rand else rng.randrange(n_or)
        elif rng.random() < par_o:
            cur = orient[w]
            if cur == data.SKIP:
                orient[w] = rng.randrange(n_or)
            elif rng.random() < 0.15:
                orient[w] = data.SKIP
            else:
                orient[w] = (cur + 1 + rng.randrange(2)) % n_or
    return (perm, orient)


def run_hs(cfg, rng, decoder, max_evals):
    """Harmony Search adaptado a permutaciones + variables discretas.

    Mecanismo de EXPLORACIÓN: selección aleatoria (1 - HMCR) y renovación
    continua de la memoria.
    Mecanismo de EXPLOTACIÓN: consideración de memoria (HMCR alto) y ajuste de
    tono (PAR) alrededor de las armonías almacenadas, más la actualización
    elitista de la memoria de armonías.
    """
    t0 = time.time()
    hms = cfg.get("hms", 20)
    hmcr = cfg.get("hmcr", 0.90)
    par = cfg.get("par", 0.30)
    bw = cfg.get("bw", 5)
    par_o = cfg.get("par_o", par)
    p_skip_rand = cfg.get("p_skip_rand", 0.15)

    counter = _Counter(decoder)

    # --- inicialización de la memoria de armonías ---
    HM = []
    for _ in range(hms):
        if not counter.budget_left(max_evals):
            break
        g = solution.initial_solution(rng)
        HM.append([g, counter(g)])
    if not HM:
        g = solution.initial_solution(rng)
        HM.append([g, counter(g)])

    while counter.budget_left(max_evals):
        # --- improvisación ---
        base = HM[rng.randrange(len(HM))]
        new = _hs_improvise(base[0], rng, hmcr, par, bw, par_o, p_skip_rand)
        new_eval = counter(new)
        # --- actualización de la memoria: sustituye a la peor si mejora ---
        worst = 0
        for i in range(1, len(HM)):
            if HM[i][1]["objective"] > HM[worst][1]["objective"]:
                worst = i
        if new_eval["objective"] < HM[worst][1]["objective"]:
            HM[worst] = [new, new_eval]

    best = min(HM, key=lambda h: h[1]["objective"])
    return {
        "name": "HS",
        "best_genotype": best[0],
        "best_eval": best[1],
        "n_evals": counter.n_evals,
        "time": time.time() - t0,
        "convergence": counter.conv,
        "config": dict(cfg),
    }


# =========================================================================== #
# 3. ITERATED GREEDY  (IG)
# =========================================================================== #
def run_ig(cfg, rng, decoder, max_evals):
    """Iterated Greedy: destrucción - construcción voraz - aceptación.

    Mecanismo de EXPLORACIÓN: la destrucción aleatoria de d ruedas (y el
    reinicio desde la mejor solución tras `patience` iteraciones sin mejora)
    diversifica la búsqueda.
    Mecanismo de EXPLOTACIÓN: la reinserción voraz de cada rueda eliminada en
    la mejor posición/orientación encontrada, más la búsqueda local opcional.

    Criterio de aceptación: se acepta la solución reconstruida si no empeora a
    la actual; en caso contrario se descarta y se continúa desde la anterior.
    """
    t0 = time.time()
    d = cfg.get("d", 8)
    n_try = cfg.get("n_try", 3)
    ls_iters = cfg.get("ls", 10)
    patience = cfg.get("patience", 25)

    counter = _Counter(decoder)
    n = data.TOTAL_WHEELS

    perm, orient = solution.initial_solution(rng)
    perm = list(perm)
    orient = list(orient)
    pos = [0] * n                      # pos[instancia] = índice en perm
    for i, w in enumerate(perm):
        pos[w] = i

    cur_eval = counter((perm, orient))
    cur_obj = cur_eval["objective"]
    best = (list(perm), list(orient))
    best_eval = cur_eval
    best_obj = cur_obj
    no_improve = 0

    while counter.budget_left(max_evals):
        # ---- reinicio desde la mejor solución si se estanca ----
        if no_improve >= patience:
            perm = list(best[0])
            orient = list(best[1])
            for i, w in enumerate(perm):
                pos[w] = i
            cur_eval = counter((perm, orient))
            cur_obj = cur_eval["objective"]
            no_improve = 0
            if not counter.budget_left(max_evals):
                break

        prev_perm = list(perm)
        prev_orient = list(orient)
        prev_pos = list(pos)
        prev_obj = cur_obj

        # ---------------------- DESTRUCCIÓN ---------------------- #
        active = [w for w in range(n) if orient[w] != data.SKIP]
        if len(active) > d:
            removed = rng.sample(active, d)
        else:
            removed = list(active)
        for w in removed:
            orient[w] = data.SKIP

        # ------------- CONSTRUCCIÓN (reinserción voraz) ---------- #
        rng.shuffle(removed)
        for w in removed:
            if not counter.budget_left(max_evals):
                break
            i = pos[w]
            best_cand = None
            best_j = i
            best_val = float("inf")
            for _ in range(n_try):
                j = rng.randrange(n)
                o = rng.randrange(data.N_ORIENT)
                cperm = list(perm)
                cperm[i], cperm[j] = cperm[j], cperm[i]
                corient = list(orient)
                corient[w] = o
                e = counter((cperm, corient))
                if e["objective"] < best_val:
                    best_val = e["objective"]
                    best_cand = (cperm, corient)
                    best_j = j
            if best_cand is not None:
                perm, orient = best_cand
                other = perm[i]
                pos[w] = best_j
                pos[other] = i

        if not counter.budget_left(max_evals):
            break

        # evaluar la solución reconstruida completa
        cur_eval = counter((perm, orient))
        cur_obj = cur_eval["objective"]

        # ------------------- BÚSQUEDA LOCAL (opcional) ----------- #
        for _ in range(ls_iters):
            if not counter.budget_left(max_evals):
                break
            cand = solution.neighbor((perm, orient), rng)
            e = counter(cand)
            if e["objective"] < cur_obj - 1e-9:
                perm = list(cand[0])
                orient = list(cand[1])
                cur_obj = e["objective"]
                cur_eval = e
                for i, w in enumerate(perm):
                    pos[w] = i

        # ---------------------- ACEPTACIÓN ----------------------- #
        if cur_obj <= prev_obj:
            if cur_obj < best_obj:
                best = (list(perm), list(orient))
                best_eval = cur_eval
                best_obj = cur_obj
                no_improve = 0
            else:
                no_improve += 1
        else:
            perm = prev_perm
            orient = prev_orient
            pos = prev_pos
            cur_obj = prev_obj
            no_improve += 1

    return {
        "name": "IG",
        "best_genotype": best,
        "best_eval": best_eval,
        "n_evals": counter.n_evals,
        "time": time.time() - t0,
        "convergence": counter.conv,
        "config": dict(cfg),
    }


METAHEURISTICS = {
    "GA": run_ga,
    "HS": run_hs,
    "IG": run_ig,
}
