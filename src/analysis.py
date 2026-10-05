"""
Análisis experimental y estadístico de los resultados (apartados 11 y 12).

Genera:
  - Tabla de estadísticos por metaheurística (mejor, peor, media, mediana,
    desviación típica, rango intercuartílico, tiempo medio...).
  - Medidas de "calidad sistemática" frente a "aciertos puntuales".
  - Curvas de convergencia medias y velocidad de convergencia.
  - Análisis estadístico formal: normalidad (Jarque-Bera), contraste global
    (ANOVA de un factor / Kruskal-Wallis) y comparaciones por pares
    (t de Welch / Mann-Whitney U) con corrección de Holm-Bonferroni.
  - Informe en Markdown con los resultados y las respuestas a las cuestiones
    del enunciado.

Todo el cálculo estadístico se hace con el módulo propio `stats_tools`
(sin scipy, sin statsmodels, sin pandas).
"""

import json
import math
import os

from . import data
from . import stats_tools as st


# --------------------------------------------------------------------------- #
# Agrupación y estadísticos descriptivos
# --------------------------------------------------------------------------- #
def _groups(rows):
    """Agrupa las filas por metaheurística conservando el orden de llegada."""
    order = []
    g = {}
    for r in rows:
        m = r["metaheuristic"]
        if m not in g:
            g[m] = []
            order.append(m)
        g[m].append(r)
    return order, g


def pearson(xs, ys):
    """Coeficiente de correlación de Pearson."""
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def summary_table(rows):
    """Estadísticos por metaheurística (función objetivo = espacio vacío, m³)."""
    order, g = _groups(rows)
    all_obj = [r["objective_empty_m3"] for r in rows]
    global_median = st.median(all_obj)
    global_best = min(all_obj)

    out = []
    for m in order:
        objs = [r["objective_empty_m3"] for r in g[m]]
        times = [r["time_s"] for r in g[m]]
        wheels = [r["n_wheels"] for r in g[m]]
        utils = [r["utilisation"] for r in g[m]]
        # "calidad sistemática": proporción de ejecuciones que alcanzan al menos
        # el 98 % del mejor valor obtenido en todo el estudio.
        thr = global_best * 1.02 if global_best > 0 else global_best
        systematic = sum(1 for v in objs if v <= thr) / float(len(objs))
        out.append({
            "metaheuristic": m,
            "n_runs": len(objs),
            "best": min(objs),
            "worst": max(objs),
            "mean": st.mean(objs),
            "median": st.median(objs),
            "std": st.std(objs),
            "iqr": st.iqr(objs),
            "range": max(objs) - min(objs),
            "mean_time_s": st.mean(times),
            "mean_wheels": st.mean(wheels),
            "mean_utilisation": st.mean(utils),
            "pct_close_to_global_best": 100.0 * systematic,
            "pct_below_global_median": 100.0 * sum(1 for v in objs if v <= global_median) / len(objs),
        })
    return out


# --------------------------------------------------------------------------- #
# Contrastes estadísticos
# --------------------------------------------------------------------------- #
def statistical_tests(rows, alpha=0.05):
    """Comparación formal de las tres metaheurísticas."""
    order, g = _groups(rows)
    groups = [[r["objective_empty_m3"] for r in g[m]] for m in order]
    names = list(order)
    out = {"names": names, "alpha": alpha}

    # 1) Normalidad (Jarque-Bera) por grupo
    normality = []
    for n_, grp in zip(names, groups):
        jb, p = st.jarque_bera(grp)
        normality.append({
            "metaheuristic": n_, "jb": jb, "p": p,
            "verdict": "normal" if (p == p and p > alpha) else "no normal",
        })
    out["normality"] = normality
    all_normal = all(x["verdict"] == "normal" for x in normality)

    # 2) Rango medio (descriptivo, base de Kruskal-Wallis)
    merged = [v for grp in groups for v in grp]
    ranks = st._ranks(merged)
    idx = 0
    mean_ranks = {}
    for n_, grp in zip(names, groups):
        mean_ranks[n_] = st.mean(ranks[idx:idx + len(grp)])
        idx += len(grp)
    out["mean_ranks"] = mean_ranks

    # 3) Contraste global
    if all_normal:
        f_stat, p_global = st.anova_oneway(groups)
        out["global_test"] = "ANOVA de un factor"
        out["global_stat"] = f_stat
        out["global_stat_name"] = "F"
    else:
        h_stat, p_global = st.kruskal_wallis(groups)
        out["global_test"] = "Kruskal-Wallis"
        out["global_stat"] = h_stat
        out["global_stat_name"] = "H"
    out["global_p"] = p_global
    out["global_significant"] = bool(p_global < alpha)

    # 4) Comparaciones por pares con corrección de Holm-Bonferroni
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if all_normal:
                stat, p = st.welch_ttest(groups[i], groups[j])
                stat_name = "t (Welch)"
            else:
                stat, p = st.mann_whitney_u(groups[i], groups[j])
                stat_name = "U (Mann-Whitney)"
            pairs.append({
                "pair": "%s vs %s" % (names[i], names[j]),
                "test": stat_name,
                "stat": stat,
                "p": p,
                "mean_a": st.mean(groups[i]),
                "mean_b": st.mean(groups[j]),
            })
    raw_p = [pp["p"] for pp in pairs]
    adj = st.holm_adjust(raw_p)
    for pp, a in zip(pairs, adj):
        pp["p_holm"] = a
        pp["significant_holm"] = bool(a < alpha)
    out["pairwise"] = pairs
    out["bonferroni_alpha"] = st.bonferroni_alpha(alpha, len(pairs))
    out["all_normal"] = all_normal
    return out


# --------------------------------------------------------------------------- #
# Convergencia
# --------------------------------------------------------------------------- #
def convergence_means(conv_data):
    """Curva media de convergencia (mejor objetivo visto hasta cada evaluación).

    Las ejecuciones pueden tener longitudes distintas; las más cortas se
    prolongan con su último valor (mejor estimación disponible).
    """
    means = {}
    for m, convs in conv_data.items():
        if not convs:
            continue
        length = max(len(c) for c in convs)
        acc = [0.0] * length
        for c in convs:
            last = c[-1] if c else float("nan")
            for i in range(length):
                acc[i] += c[i] if i < len(c) else last
        means[m] = [v / len(convs) for v in acc]
    return means


def convergence_speed(conv_data, frac=0.95):
    """Evaluaciones necesarias para alcanzar el `frac` de la mejora total
    conseguida por cada ejecución (media por algoritmo)."""
    speeds = {}
    for m, convs in conv_data.items():
        vals = []
        for c in convs:
            if len(c) < 2:
                continue
            f0 = c[0]
            fb = c[-1]
            span = f0 - fb
            if span <= 1e-12:
                vals.append(1.0)
                continue
            target = fb + (1.0 - frac) * span
            k = len(c)
            for i, v in enumerate(c):
                if v <= target:
                    k = i + 1
                    break
            vals.append(float(k))
        speeds[m] = st.mean(vals) if vals else float("nan")
    return speeds


# --------------------------------------------------------------------------- #
# Perfil de las mejores soluciones (cuestiones 13-18)
# --------------------------------------------------------------------------- #
def solution_profiles(best_per_meta):
    """Para cada metaheurística: unidades por tipo y orientaciones dominantes
    en su mejor solución."""
    profiles = {}
    for m, entry in best_per_meta.items():
        full = entry[2]
        used = full["used"]
        by_type = {data.WHEEL_SPECS[i][0]: used[i] for i in range(data.N_TYPES)}
        ori = [0, 0, 0]
        ori_by_type = {}
        big_flat = 0
        big_total = 0
        for p in full["placements"]:
            o = p["orient_idx"]
            ori[o] += 1
            ti = p["type_idx"]
            ori_by_type.setdefault(ti, [0, 0, 0])[o] += 1
            # "rueda grande" = diámetro >= 1.0 m
            if data.wheel_types()[ti]["diameter_m"] >= 1.0:
                big_total += 1
                if o == 0:
                    big_flat += 1
        profiles[m] = {
            "n_wheels": full["n_wheels"],
            "objective": full["objective"],
            "utilisation": full["utilisation"],
            "by_type": by_type,
            "orientations": {data.ORIENTATIONS[i]: ori[i] for i in range(3)},
            "dominant_orientation_by_type": {
                data.WHEEL_SPECS[ti][0]: data.ORIENTATIONS[
                    max(range(3), key=lambda k: cnts[k])]
                for ti, cnts in sorted(ori_by_type.items())
            },
            "pct_big_flat": (100.0 * big_flat / big_total) if big_total else float("nan"),
        }
    return profiles


# --------------------------------------------------------------------------- #
# Informe
# --------------------------------------------------------------------------- #
def _fmt(v, nd=3):
    if v != v:            # NaN
        return "n/d"
    return ("%%.%df" % nd) % v


def write_report(rows, sumry, stest, out_dir, conv_data, best_per_meta,
                 configs, max_evals, seeds):
    lines = []
    add = lines.append

    add("# Resultados experimentales — Carga de ruedas en un camión\n")
    add("Generado automáticamente por `main.py`.\n")
    add("- Camión: %.2f x %.2f x %.2f m (volumen **%.2f m³**)."
        % (data.TRUCK["length"], data.TRUCK["width"], data.TRUCK["height"],
           data.truck_volume()))
    add("- Ruedas disponibles: %d (%d tipos x %d uds.). Volumen total disponible "
        "%.2f m³ > volumen del camión => la **selección** es decisiva.\n"
        % (data.TOTAL_WHEELS, data.N_TYPES, data.AVAILABLE_PER_TYPE,
           data.total_available_volume()))
    add("- Metaheurísticas comparadas: **%s**." % ", ".join(stest["names"]))
    add("- Ejecuciones: %d independientes por algoritmo (%d en total), semillas "
        "%d...%d, presupuesto de %d evaluaciones por ejecución.\n"
        % (len(seeds), len(rows), seeds[0], seeds[-1], max_evals))

    # ---- 1. Estadísticos descriptivos ----
    add("## 1. Estadísticos por metaheurística (espacio vacío, m³; menor = mejor)\n")
    add("| Algoritmo | Mejor | Peor | Media | Mediana | D.típica | IQR | Rango | "
        "Utiliz. media | Ruedas (media) | Tiempo medio (s) |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sumry:
        add("| **%s** | %s | %s | %s | %s | %s | %s | %s | %.2f%% | %.1f | %.2f |"
            % (r["metaheuristic"], _fmt(r["best"]), _fmt(r["worst"]),
               _fmt(r["mean"]), _fmt(r["median"]), _fmt(r["std"]), _fmt(r["iqr"]),
               _fmt(r["range"]), 100 * r["mean_utilisation"], r["mean_wheels"],
               r["mean_time_s"]))

    add("\n### 1.1 ¿Calidad puntual o calidad sistemática?\n")
    add("| Algoritmo | %% ejecuciones <= 102%% del mejor global | %% ejecuciones por debajo de la mediana global |")
    add("|---|---|---|")
    for r in sumry:
        add("| %s | %.1f%% | %.1f%% |"
            % (r["metaheuristic"], r["pct_close_to_global_best"],
               r["pct_below_global_median"]))
    add("\nUn algoritmo con buen *mejor* valor pero bajo porcentaje en la primera "
        "columna obtiene buenas soluciones de forma **ocasional**; un porcentaje "
        "alto indica buen comportamiento **sistemático**.\n")

    # ---- 2. Contrastes estadísticos ----
    add("## 2. Análisis estadístico\n")
    add("**Hipótesis.**  H0: las tres metaheurísticas producen la misma "
        "distribución de espacio vacío.  H1: al menos una difiere.  "
        "Nivel de significación α = %.2f.\n" % stest["alpha"])
    add("**Normalidad (Jarque-Bera).**  Se usa porque la aproximación asintótica "
        "del ANOVA exige normalidad y los resultados de metaheurísticas suelen "
        "ser asimétricos.\n")
    add("| Algoritmo | JB | p | Conclusión |")
    add("|---|---|---|---|")
    for x in stest["normality"]:
        add("| %s | %s | %.4f | %s |"
            % (x["metaheuristic"], _fmt(x["jb"], 3), x["p"], x["verdict"]))

    add("\n**Contraste global:** %s (%s = %s, p = %.3e) -> %s.\n"
        % (stest["global_test"], stest["global_stat_name"],
           _fmt(stest["global_stat"], 4), stest["global_p"],
           "se rechaza H0" if stest["global_significant"] else "no se rechaza H0"))
    add("Rangos medios (Kruskal-Wallis, menor = mejor): "
        + ", ".join("%s = %.2f" % (k, v) for k, v in stest["mean_ranks"].items())
        + "\n")
    add("**Comparaciones por pares.**  Se aplica %s y se corrige el nivel de "
        "significación por comparaciones múltiples con **Holm-Bonferroni** "
        "(Bonferroni clásico exigiría α = %.4f).\n"
        % ("t de Welch" if stest["all_normal"] else "Mann-Whitney U",
           stest["bonferroni_alpha"]))
    add("| Pareja | Estadístico | p | p ajustado (Holm) | Significativo |")
    add("|---|---|---|---|---|")
    for pp in stest["pairwise"]:
        add("| %s | %s = %s | %.3e | %.3e | %s |"
            % (pp["pair"], pp["test"], _fmt(pp["stat"], 3), pp["p"],
               pp["p_holm"], "SÍ" if pp["significant_holm"] else "no"))

    # ---- 3. Convergencia ----
    add("\n## 3. Convergencia\n")
    speeds = convergence_speed(conv_data)
    if speeds:
        add("| Algoritmo | Evaluaciones medias para alcanzar el 95%% de su mejora total |")
        add("|---|---|")
        for m in stest["names"]:
            if m in speeds:
                add("| %s | %.0f |" % (m, speeds[m]))
        add("")
    # texto: quién es mejor al final de la curva
    means = convergence_means(conv_data)
    if means:
        final = {m: v[-1] for m, v in means.items()}
        best_final = min(final, key=lambda k: final[k])
        add("Al final del presupuesto la mejor curva media es la de **%s** "
            "(%.3f m³ de espacio vacío medio).\n" % (best_final, final[best_final]))

    # ---- 4. Perfil de las mejores soluciones ----
    add("\n## 4. Perfil de las mejores soluciones\n")
    profiles = solution_profiles(best_per_meta)
    add("| Algoritmo | Ruedas | Espacio vacío (m³) | Utilización | "
        "flat | parallel_to_end | parallel_to_side | %% ruedas grandes en flat |")
    add("|---|---|---|---|---|---|---|---|")
    for m in stest["names"]:
        p = profiles[m]
        o = p["orientations"]
        add("| %s | %d | %.3f | %.2f%% | %d | %d | %d | %.1f%% |"
            % (m, p["n_wheels"], p["objective"], 100 * p["utilisation"],
               o["flat"], o["parallel_to_end"], o["parallel_to_side"],
               p["pct_big_flat"]))
    add("\nUnidades cargadas de cada tipo en la mejor solución:\n")
    add("| Algoritmo | " + " | ".join(data.WHEEL_SPECS[i][0] for i in range(data.N_TYPES)) + " |")
    add("|---|" + "---|" * data.N_TYPES)
    for m in stest["names"]:
        bt = profiles[m]["by_type"]
        add("| %s | " % m + " | ".join(str(bt[data.WHEEL_SPECS[i][0]])
                                       for i in range(data.N_TYPES)) + " |")

    # ---- 5. Cuestiones ----
    add("\n## 5. Respuestas a las cuestiones del enunciado\n")
    objs = [r["objective_empty_m3"] for r in rows]
    nw = [r["n_wheels"] for r in rows]
    r_pearson = pearson(nw, objs)
    add("**¿Una solución que transporta más ruedas es necesariamente mejor?**  "
        "Correlación de Pearson entre nº de ruedas y espacio vacío sobre las %d "
        "ejecuciones: r = %.3f.  Un valor cercano a 0 (o positivo) indicaría que "
        "cargar más ruedas no garantiza menor espacio vacío; un valor claramente "
        "negativo indicaría lo contrario.\n" % (len(rows), r_pearson))

    best_mean = min(sumry, key=lambda r: r["mean"])
    worst_mean = max(sumry, key=lambda r: r["mean"])
    low_var = min(sumry, key=lambda r: r["std"])
    best_abs = min(sumry, key=lambda r: r["best"])
    add("**Mejor aprovechamiento medio:** **%s** (%.3f m³ de espacio vacío, "
        "%.2f%% de utilización).\n" % (best_mean["metaheuristic"],
                                       best_mean["mean"],
                                       100 * best_mean["mean_utilisation"]))
    add("**Menor variabilidad:** **%s** (D.típica = %.3f m³, IQR = %.3f m³).\n"
        % (low_var["metaheuristic"], low_var["std"], low_var["iqr"]))
    add("**Peor comportamiento medio:** %s (%.3f m³).\n"
        % (worst_mean["metaheuristic"], worst_mean["mean"]))
    add("**Mejor solución puntual:** %s (%.3f m³).\n"
        % (best_abs["metaheuristic"], best_abs["best"]))

    # tipos más frecuentes en las mejores soluciones
    tot = {}
    for m in stest["names"]:
        for k, v in profiles[m]["by_type"].items():
            tot[k] = tot.get(k, 0) + v
    top = sorted(tot.items(), key=lambda kv: -kv[1])[:5]
    add("**Tipos más frecuentes en las mejores soluciones:** "
        + ", ".join("%s (%d uds.)" % (k, v) for k, v in top) + ".\n")

    if speeds:
        fastest = min(speeds, key=lambda k: speeds[k])
        add("**Convergencia más rápida:** **%s** (%.0f evaluaciones de media para "
            "el 95%% de su mejora).\n" % (fastest, speeds[fastest]))

    if stest["global_significant"]:
        sig_pairs = [pp["pair"] for pp in stest["pairwise"] if pp["significant_holm"]]
        add("**¿Diferencias significativas?**  Sí: %s rechaza H0 (p = %.3e).  "
            "Parejas con diferencia significativa tras Holm: %s.\n"
            % (stest["global_test"], stest["global_p"],
               ", ".join(sig_pairs) if sig_pairs else "ninguna"))
    else:
        add("**¿Diferencias significativas?**  No: %s no rechaza H0 (p = %.3e).\n"
            % (stest["global_test"], stest["global_p"]))

    add("**Recomendación.**  Considerando conjuntamente la calidad media, la "
        "variabilidad y la velocidad de convergencia, se recomienda **%s**.\n"
        % best_mean["metaheuristic"])

    # ---- 6. Configuraciones ----
    add("\n## 6. Parámetros utilizados\n")
    add("```")
    for m in stest["names"]:
        add("%s: %s" % (m, configs.get(m, {})))
    add("```\n")

    path = os.path.join(out_dir, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    # Datos completos del análisis estadístico (reproducibilidad)
    stats_path = os.path.join(out_dir, "statistics.json")
    with open(stats_path, "w") as f:
        json.dump({
            "summary": sumry,
            "tests": stest,
            "convergence_speed": speeds,
            "profiles": profiles,
            "pearson_wheels_vs_objective": r_pearson,
            "configs": {m: configs.get(m, {}) for m in stest["names"]},
        }, f, indent=2, ensure_ascii=False, default=str)

    return path
