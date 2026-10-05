"""
Estadística descriptiva y contrastes de hipótesis implementados DESDE CERO.

Este módulo sustituye por completo a bibliotecas externas de estadística
(scipy.stats, statsmodels, ...): todos los estadísticos y todos los p-valores
se calculan con la biblioteca estándar de Python (`math`).

Contenido:
  * Estadísticos descriptivos: media, mediana, cuantiles, varianza, desviación
    típica, rango intercuartílico, asimetría y curtosis.
  * Funciones de distribución: normal (vía math.erf / math.erfc), chi-cuadrado
    (gamma incompleta regularizada), t de Student y F de Fisher (beta
    incompleta regularizada, mediante fracción continua de Lentz).
  * Contrastes: Jarque-Bera (normalidad), ANOVA de un factor, Kruskal-Wallis,
    Mann-Whitney U y t de Welch.
  * Corrección para comparaciones múltiples: Bonferroni y Holm-Bonferroni.
"""

import math
from collections import Counter


# --------------------------------------------------------------------------- #
# Estadísticos descriptivos
# --------------------------------------------------------------------------- #
def mean(xs):
    return sum(xs) / float(len(xs))


def var(xs, ddof=1):
    n = len(xs)
    if n - ddof <= 0:
        return 0.0
    m = mean(xs)
    return sum((v - m) ** 2 for v in xs) / float(n - ddof)


def std(xs, ddof=1):
    return math.sqrt(var(xs, ddof))


def median(xs):
    return quantile(xs, 0.5)


def quantile(xs, q):
    """Cuantil tipo 7 (interpolación lineal), el mismo que usa R por defecto."""
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    if n == 1:
        return float(s[0])
    h = (n - 1) * q
    lo = int(math.floor(h))
    hi = min(lo + 1, n - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


def iqr(xs):
    return quantile(xs, 0.75) - quantile(xs, 0.25)


def skewness(xs):
    n = len(xs)
    m = mean(xs)
    m2 = sum((v - m) ** 2 for v in xs) / n
    if m2 <= 0:
        return 0.0
    m3 = sum((v - m) ** 3 for v in xs) / n
    return m3 / (m2 ** 1.5)


def kurtosis(xs):
    """Curtosis (no centrada en 3: la normal vale 3)."""
    n = len(xs)
    m = mean(xs)
    m2 = sum((v - m) ** 2 for v in xs) / n
    if m2 <= 0:
        return 3.0
    m4 = sum((v - m) ** 4 for v in xs) / n
    return m4 / (m2 ** 2)


# --------------------------------------------------------------------------- #
# Distribuciones
# --------------------------------------------------------------------------- #
def norm_cdf(z):
    """Función de distribución de la normal estándar."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _gser(a, x):
    """Serie para P(a, x) (gamma incompleta regularizada), válida si x < a+1."""
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(10000):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * 1e-14:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gcf(a, x):
    """Fracción continua para Q(a, x), válida si x >= a+1."""
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gammq(a, x):
    """Q(a, x) = 1 - P(a, x) de la gamma incompleta regularizada."""
    if x <= 0.0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _gser(a, x)
    return _gcf(a, x)


def chi2_sf(x, k):
    """P(chi2_k > x): supervivencia de la chi-cuadrado con k g. de libertad."""
    if x <= 0:
        return 1.0
    return gammq(k / 2.0, x / 2.0)


def _betacf(a, b, x):
    """Fracción continua (Lentz) para la beta incompleta."""
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 10000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def betai(a, b, x):
    """I_x(a, b): beta incompleta regularizada."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf_two_sided(t, df):
    """p-valor bilateral de la t de Student."""
    if df <= 0:
        return 1.0
    return betai(df / 2.0, 0.5, df / (df + t * t))


def f_sf(f, d1, d2):
    """P(F_{d1,d2} > f): supervivencia de la F de Fisher."""
    if f <= 0:
        return 1.0
    if d1 <= 0 or d2 <= 0:
        return 1.0
    return betai(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f))


# --------------------------------------------------------------------------- #
# Contrastes
# --------------------------------------------------------------------------- #
def _ranks(values):
    """Rangos medios (1 = menor), con empates resueltos por promedio."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _tie_term(values):
    """Suma de (t^3 - t) sobre los grupos de valores empatados."""
    counts = Counter(values)
    return sum(t ** 3 - t for t in counts.values() if t > 1)


def jarque_bera(xs):
    """Contraste de normalidad de Jarque-Bera. Devuelve (JB, p).

    Bajo H0 el estadístico se distribuye como chi-cuadrado con 2 g.l., cuya
    función de supervivencia es exp(-x/2).
    """
    n = len(xs)
    if n < 4:
        return float("nan"), float("nan")
    s = skewness(xs)
    k = kurtosis(xs)
    jb = n / 6.0 * (s * s + (k - 3.0) ** 2 / 4.0)
    return jb, math.exp(-jb / 2.0)


def anova_oneway(groups):
    """ANOVA de un factor. Devuelve (F, p)."""
    k = len(groups)
    n = sum(len(g) for g in groups)
    if k < 2 or n - k <= 0:
        return float("nan"), float("nan")
    grand = mean([v for g in groups for v in g])
    ssb = sum(len(g) * (mean(g) - grand) ** 2 for g in groups)
    ssw = 0.0
    for g in groups:
        mg = mean(g)
        ssw += sum((v - mg) ** 2 for v in g)
    dfb = k - 1
    dfw = n - k
    if ssw <= 0:
        return float("inf"), 0.0
    f = (ssb / dfb) / (ssw / dfw)
    return f, f_sf(f, dfb, dfw)


def kruskal_wallis(groups):
    """Kruskal-Wallis (alternativa no paramétrica al ANOVA). Devuelve (H, p)."""
    k = len(groups)
    allv = [v for g in groups for v in g]
    n = len(allv)
    if k < 2 or n == 0:
        return float("nan"), float("nan")
    ranks = _ranks(allv)
    h = 0.0
    idx = 0
    for g in groups:
        r = sum(ranks[idx:idx + len(g)])
        idx += len(g)
        h += r * r / float(len(g))
    h = 12.0 / (n * (n + 1.0)) * h - 3.0 * (n + 1.0)
    tie = _tie_term(allv)
    if tie > 0 and n > 2:
        h /= (1.0 - tie / float(n ** 3 - n))
    return h, chi2_sf(h, k - 1)


def mann_whitney_u(a, b):
    """Mann-Whitney U bilateral con aproximación normal y corrección por
    empates. Devuelve (U, p)."""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    combined = list(a) + list(b)
    n = n1 + n2
    ranks = _ranks(combined)
    r1 = sum(ranks[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)
    mu = n1 * n2 / 2.0
    tie = _tie_term(combined)
    var = n1 * n2 / 12.0 * ((n + 1) - tie / float(n * (n - 1))) if n > 1 else 0.0
    if var <= 0:
        return u, 1.0
    z = (abs(u - mu) - 0.5) / math.sqrt(var)      # corrección de continuidad
    return u, 2.0 * norm_sf(z)


def welch_ttest(a, b):
    """t de Welch (varianzas desiguales) bilateral. Devuelve (t, p)."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan"), float("nan")
    v1, v2 = var(a), var(b)
    se2 = v1 / n1 + v2 / n2
    if se2 <= 0:
        return 0.0, 1.0
    t = (mean(a) - mean(b)) / math.sqrt(se2)
    df = se2 ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    return t, t_sf_two_sided(t, df)


# --------------------------------------------------------------------------- #
# Comparaciones múltiples
# --------------------------------------------------------------------------- #
def holm_adjust(pvals):
    """Devuelve los p-valores ajustados por el procedimiento de Holm."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        if val > running:
            running = val
        if running > 1.0:
            running = 1.0
        adj[i] = running
    return adj


def bonferroni_alpha(alpha, n_comparisons):
    return alpha / float(max(1, n_comparisons))
