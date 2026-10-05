"""
Punto de entrada de la práctica.

Flujo completo:
  1. (Opcional) Ajuste de parámetros de las tres metaheurísticas.
  2. 30 ejecuciones independientes de cada una, con las MISMAS semillas.
  3. Análisis estadístico y tablas.
  4. Visualización (empaquetado 3D + convergencia) y ficheros JSON.

Uso:
  python main.py                      -> experimento completo (30 semillas + ajuste)
  python main.py --no-tune            -> usa las configuraciones por defecto
  python main.py --max-evals 800 --seeds 30
  python main.py --quick              -> 5 semillas, presupuesto reducido

Metaheurísticas implementadas desde cero: Genetic Algorithm (GA),
Harmony Search (HS) e Iterated Greedy (IG).
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import analysis, data, experiment, visualize
from src.geometry import Decoder, check_feasibility

META_NAMES = ["GA", "HS", "IG"]


def _print_summary(sumry):
    header = ("%-6s %8s %8s %8s %8s %8s %8s %10s %8s"
              % ("Alg.", "mejor", "peor", "media", "mediana", "D.tip.", "IQR",
                 "utiliz.%", "t(s)"))
    print(header)
    print("-" * len(header))
    for r in sumry:
        print("%-6s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %9.2f%% %8.2f"
              % (r["metaheuristic"], r["best"], r["worst"], r["mean"],
                 r["median"], r["std"], r["iqr"], 100 * r["mean_utilisation"],
                 r["mean_time_s"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--max-evals", type=int, default=1200,
                    help="presupuesto de evaluaciones por ejecución")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--no-tune", action="store_true",
                    help="no ajustar parámetros (usa los valores por defecto)")
    ap.add_argument("--quick", action="store_true",
                    help="validación rápida (5 semillas, 200 evaluaciones)")
    args = ap.parse_args()

    if args.quick:
        seeds = experiment.SEEDS[:5]
        max_evals = 200
        do_tune = False
    else:
        seeds = experiment.SEEDS[:args.seeds]
        max_evals = args.max_evals
        do_tune = not args.no_tune

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    decoder = Decoder()

    t0 = time.time()

    if do_tune:
        print("== Ajuste de parámetros (misma rejilla y mismo esfuerzo para las 3) ==")
        configs = experiment.tune_parameters(META_NAMES, experiment.SEEDS[:3],
                                             400, decoder)
    else:
        configs = {m: dict(experiment.DEFAULT_CONFIGS[m]) for m in META_NAMES}

    print("\n== Configuraciones finales ==")
    for m in META_NAMES:
        print("  %s: %s" % (m, configs[m]))

    print("\n== Ejecución experimental ==")
    rows, best_per_meta, conv_data, json_paths = experiment.run_full(
        META_NAMES, seeds, max_evals, configs, decoder, out_dir)

    print("\n== Análisis ==")
    sumry = analysis.summary_table(rows)
    stest = analysis.statistical_tests(rows)
    report_path = analysis.write_report(rows, sumry, stest, out_dir, conv_data,
                                        best_per_meta, configs, max_evals, seeds)

    conv_means = analysis.convergence_means(conv_data)
    conv_png = visualize.plot_convergence(conv_means, out_dir)
    pack_pngs = {}
    for m in META_NAMES:
        pack_pngs[m] = visualize.plot_solution(best_per_meta[m][2], m, out_dir)

    print("\n== Comprobación de factibilidad de las soluciones entregadas ==")
    for m in META_NAMES:
        chk = check_feasibility(best_per_meta[m][2]["placements"])
        print("  %-3s %s  ruedas=%3d  fuera_de_limites=%d  solapes=%d  exceso_inventario=%d"
              % (m, "OK   " if chk["ok"] else "FALLO", chk["n_wheels"],
                 chk["out_of_bounds"], chk["overlaps"], chk["over_stock"]))

    print("\n== RESUMEN (espacio vacío en m³, menor es mejor) ==")
    _print_summary(sumry)

    print("\nNormalidad (Jarque-Bera): " + ", ".join(
        "%s p=%.4f (%s)" % (x["metaheuristic"], x["p"], x["verdict"])
        for x in stest["normality"]))
    print("Contraste global: %s (%s=%.4f, p=%.3e) -> %s"
          % (stest["global_test"], stest["global_stat_name"],
             stest["global_stat"], stest["global_p"],
             "diferencias significativas" if stest["global_significant"]
             else "sin diferencias significativas"))
    for pp in stest["pairwise"]:
        print("  %-14s p=%.3e  p_Holm=%.3e  %s"
              % (pp["pair"], pp["p"], pp["p_holm"],
                 "SIGNIFICATIVO" if pp["significant_holm"] else "no significativo"))

    speeds = analysis.convergence_speed(conv_data)
    if speeds:
        print("\nVelocidad de convergencia (evaluaciones medias para el 95% de la mejora):")
        for m in META_NAMES:
            if m in speeds:
                print("  %-4s %.0f" % (m, speeds[m]))

    print("\nMejores soluciones JSON: %s" % json_paths)
    print("PNG empaquetado: %s" % pack_pngs)
    print("PNG convergencia: %s" % conv_png)
    print("Informe: %s" % report_path)
    print("CSV: %s" % os.path.join(out_dir, "results.csv"))
    print("\nTiempo total: %.1f min" % ((time.time() - t0) / 60.0))


if __name__ == "__main__":
    main()
