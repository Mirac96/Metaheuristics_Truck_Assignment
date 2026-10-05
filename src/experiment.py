"""
Ejecución experimental: ajuste de parámetros, 30 ejecuciones independientes por
metaheurística, registro de resultados y exportación de las mejores soluciones
a JSON (formato del apartado 13 del enunciado).

Sólo se usa la biblioteca estándar de Python.
"""

import csv
import json
import os
import random

from . import data, metaheuristics
from .geometry import Decoder

# Semillas COMUNES a las tres metaheurísticas (reproducibilidad, apartado 9).
SEEDS = list(range(1000, 1030))

# Configuraciones por defecto (se refinan con tune_parameters).
DEFAULT_CONFIGS = {
    "GA": {"pop": 40, "p_cross": 0.90, "p_perm": 0.010, "p_orient": 0.02,
           "p_skip": 0.010, "tournament": 3, "elite": 2},
    "HS": {"hms": 20, "hmcr": 0.98, "par": 0.02, "bw": 3,
           "par_o": 0.02, "p_skip_rand": 0.15},
    "IG": {"d": 8, "n_try": 3, "ls": 10, "patience": 25},
}

# Rejillas de ajuste.  Las TRES tienen exactamente 6 configuraciones y se
# evalúan con las mismas semillas y el mismo presupuesto reducido, de modo que
# ninguna metaheurística recibe más esfuerzo de calibración que las demás
# (apartado 7 del enunciado).
TUNING_GRIDS = {
    "GA": [
        {"pop": 20, "p_cross": 0.90, "p_perm": 0.010, "p_orient": 0.02, "p_skip": 0.010, "tournament": 3, "elite": 2},
        {"pop": 20, "p_cross": 0.90, "p_perm": 0.020, "p_orient": 0.05, "p_skip": 0.020, "tournament": 5, "elite": 1},
        {"pop": 40, "p_cross": 0.90, "p_perm": 0.010, "p_orient": 0.02, "p_skip": 0.010, "tournament": 3, "elite": 2},
        {"pop": 40, "p_cross": 0.85, "p_perm": 0.005, "p_orient": 0.01, "p_skip": 0.005, "tournament": 3, "elite": 4},
        {"pop": 40, "p_cross": 0.90, "p_perm": 0.020, "p_orient": 0.05, "p_skip": 0.020, "tournament": 5, "elite": 2},
        {"pop": 60, "p_cross": 0.90, "p_perm": 0.010, "p_orient": 0.02, "p_skip": 0.010, "tournament": 3, "elite": 3},
    ],
    "HS": [
        {"hms": 10, "hmcr": 0.98, "par": 0.02, "bw": 3, "par_o": 0.02, "p_skip_rand": 0.15},
        {"hms": 10, "hmcr": 0.95, "par": 0.05, "bw": 5, "par_o": 0.05, "p_skip_rand": 0.15},
        {"hms": 20, "hmcr": 0.98, "par": 0.02, "bw": 3, "par_o": 0.02, "p_skip_rand": 0.15},
        {"hms": 20, "hmcr": 0.98, "par": 0.05, "bw": 5, "par_o": 0.05, "p_skip_rand": 0.15},
        {"hms": 30, "hmcr": 0.98, "par": 0.01, "bw": 3, "par_o": 0.01, "p_skip_rand": 0.15},
        {"hms": 30, "hmcr": 0.95, "par": 0.02, "bw": 5, "par_o": 0.02, "p_skip_rand": 0.15},
    ],
    "IG": [
        {"d": 4, "n_try": 3, "ls": 10, "patience": 20},
        {"d": 4, "n_try": 5, "ls": 20, "patience": 25},
        {"d": 8, "n_try": 3, "ls": 10, "patience": 25},
        {"d": 8, "n_try": 5, "ls": 20, "patience": 30},
        {"d": 15, "n_try": 3, "ls": 20, "patience": 25},
        {"d": 15, "n_try": 5, "ls": 30, "patience": 30},
    ],
}


def run_one(meta_name, cfg, seed, max_evals, decoder):
    """Una ejecución independiente: misma semilla, mismo presupuesto."""
    rng = random.Random(seed)
    runner = metaheuristics.METAHEURISTICS[meta_name]
    return runner(cfg, rng, decoder, max_evals)


def _row(meta_name, seed, res):
    """Fila de resultados de una ejecución (apartado 10 del enunciado)."""
    be = res["best_eval"]
    used = be["used"]
    row = {
        "metaheuristic": meta_name,
        "seed": seed,
        "objective_empty_m3": round(be["objective"], 6),
        "utilisation": round(be["utilisation"], 6),
        "n_wheels": be["n_wheels"],
        "n_evals": res["n_evals"],
        "time_s": round(res["time"], 4),
        "feasible": 1 if be["feasible"] else 0,
    }
    for ti, t in enumerate(data.wheel_types()):
        row["n_" + t["code"]] = int(used[ti])
    return row


def tune_parameters(meta_names, seeds, tune_evals, decoder, log=print):
    """Ajuste de parámetros: prueba las configuraciones de la rejilla con unas
    pocas semillas y un presupuesto reducido, y elige la de menor espacio vacío
    MEDIO (se evita seleccionar por el mejor resultado puntual)."""
    best_cfg = {}
    for m in meta_names:
        best_mean = float("inf")
        best_c = None
        for c in TUNING_GRIDS.get(m, [DEFAULT_CONFIGS[m]]):
            objs = []
            for s in seeds:
                res = run_one(m, c, s, tune_evals, decoder)
                objs.append(res["best_eval"]["objective"])
            mean_obj = sum(objs) / len(objs)
            log("  [tune] %s %s -> vacio medio %.4f m3" % (m, c, mean_obj))
            if mean_obj < best_mean:
                best_mean = mean_obj
                best_c = c
        best_cfg[m] = dict(best_c)
        log("  [tune] %s ELEGIDA: %s (vacio medio %.4f m3)" % (m, best_c, best_mean))
    return best_cfg


def save_best_json(meta_name, eval_info, out_dir):
    """Escribe best_<META>.json con el formato del apartado 13 del enunciado."""
    out = {"metaheuristic": meta_name, "placements": []}
    for p in eval_info["placements"]:
        out["placements"].append({
            "coord_x": round(p["cx"], 4),
            "coord_y": round(p["cy"], 4),
            "coord_z": round(p["cz"], 4),
            "type": data.WHEEL_SPECS[p["type_idx"]][0],
            "orientation": data.ORIENTATIONS[p["orient_idx"]],
        })
    path = os.path.join(out_dir, "best_%s.json" % meta_name)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return path


def run_full(meta_names, seeds, max_evals, configs, decoder, out_dir, log=print):
    """Ejecuta las 30 semillas de cada metaheurística y guarda todo."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    conv_data = {m: [] for m in meta_names}
    best_per_meta = {}          # meta -> (objetivo, resultado)
    csv_path = os.path.join(out_dir, "results.csv")
    fields = None

    with open(csv_path, "w", newline=""):
        pass                    # se trunca; la cabecera se escribe abajo

    for m in meta_names:
        cfg = configs[m]
        log("Ejecutando %s (%d semillas, max_evals=%d)..." % (m, len(seeds), max_evals))
        for si, s in enumerate(seeds):
            res = run_one(m, cfg, s, max_evals, decoder)
            row = _row(m, s, res)
            rows.append(row)
            conv_data[m].append(res["convergence"])
            if fields is None:
                fields = list(row.keys())
                with open(csv_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fields)
                    w.writeheader()
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)
            if m not in best_per_meta or res["best_eval"]["objective"] < best_per_meta[m][0]:
                best_per_meta[m] = (res["best_eval"]["objective"], res)
            log("  [%s] semilla %d (%d/%d): vacio=%.3f m3  ruedas=%d  t=%.1fs"
                % (m, s, si + 1, len(seeds), row["objective_empty_m3"],
                   row["n_wheels"], row["time_s"]))
        log("  %s terminado. Mejor vacio = %.4f m3" % (m, best_per_meta[m][0]))

    # Reconstruir por completo la mejor solución de cada algoritmo y exportarla.
    json_paths = {}
    for m, (obj, res) in best_per_meta.items():
        full = None
        from . import solution
        full = solution.evaluate(res["best_genotype"], decoder, want_placements=True)
        best_per_meta[m] = (obj, res, full)
        json_paths[m] = save_best_json(m, full, out_dir)
    return rows, best_per_meta, conv_data, json_paths
