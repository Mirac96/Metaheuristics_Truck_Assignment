"""Autocomprobación: importaciones, estadística y una ejecución mínima de los 3 algoritmos."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import analysis, data, experiment, metaheuristics, solution, stats_tools as st
from src.geometry import Decoder

ok = True


def check(name, got, want, tol=1e-3):
    global ok
    good = abs(got - want) <= tol
    ok = ok and good
    print("%-42s got=%.6f want=%.6f  %s" % (name, got, want, "OK" if good else "FALLO"))


print("--- distribuciones ---")
check("norm_cdf(1.96)", st.norm_cdf(1.96), 0.9750, 1e-3)
check("chi2_sf(3.8415, 1)", st.chi2_sf(3.8415, 1), 0.05, 1e-3)
check("chi2_sf(5.9915, 2)", st.chi2_sf(5.9915, 2), 0.05, 1e-3)
check("t_sf_two_sided(2.0, 10)", st.t_sf_two_sided(2.0, 10), 0.0734, 1e-3)
check("f_sf(4.26, 2, 9) ~ 0.05", st.f_sf(4.2565, 2, 9), 0.05, 2e-3)

print("--- contrastes ---")
a = [10.2, 11.1, 9.8, 10.5, 12.0, 10.9, 9.9, 11.4]
b = [13.5, 12.8, 14.1, 13.0, 12.4, 13.9, 13.2, 12.7]
c = [10.5, 11.8, 10.1, 12.2, 11.0, 10.7, 11.6, 10.3]
H, p = st.kruskal_wallis([a, b, c])
print("Kruskal-Wallis H=%.4f p=%.5f (esperado p muy pequeno)" % (H, p))
check("Kruskal-Wallis p", p, 0.0005, 2e-3)
U, pu = st.mann_whitney_u(a, b)
print("Mann-Whitney U=%.1f p=%.6f" % (U, pu))
F, pf = st.anova_oneway([a, b, c])
print("ANOVA F=%.4f p=%.6f" % (F, pf))
t, pt = st.welch_ttest(a, b)
print("Welch t=%.4f p=%.6f" % (t, pt))
jb, pjb = st.jarque_bera(a)
print("Jarque-Bera JB=%.4f p=%.4f" % (jb, pjb))
print("Holm ajustado:", ["%.4f" % v for v in st.holm_adjust([0.01, 0.04, 0.03])])

print("\n--- algoritmos (30 evaluaciones) ---")
dec = Decoder()
import random
for name, cfg in experiment.DEFAULT_CONFIGS.items():
    rng = random.Random(7)
    res = metaheuristics.METAHEURISTICS[name](cfg, rng, dec, 30)
    print("%-4s vacio=%8.3f m3  util=%5.1f%%  ruedas=%3d  evals=%d  t=%.2fs"
          % (name, res["best_eval"]["objective"], 100 * res["best_eval"]["utilisation"],
             res["best_eval"]["n_wheels"], res["n_evals"], res["time"]))

print("\nRESULTADO GLOBAL:", "OK" if ok else "HAY FALLOS")
