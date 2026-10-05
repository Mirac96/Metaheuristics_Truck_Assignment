"""
Modelo geométrico y decodificador de soluciones.

Hipótesis geométricas (justificadas en el informe):

  1. Cada rueda es un CILINDRO de diámetro exterior D_k y anchura w_k.
  2. Para garantizar la no superposición de forma sencilla y robusta, cada rueda
     se representa por su CAJA AXIS-ALINEADA (AABB) asociada a su orientación.
     Si dos AABB son disjuntas, los cilindros que contienen tampoco se solapan
     (condición SUFICIENTE), de modo que la factibilidad queda garantizada.
  3. La posición de una rueda es el centro de su AABB.  Origen en la esquina
     trasera-izquierda-inferior del camión: X = largo, Y = ancho, Z = alto
     (desde el suelo).  Coincide con el formato JSON del enunciado.

Extents de la AABB según orientación (en X, Y, Z):

    flat              -> (D, D, w)   eje vertical
    parallel_to_end   -> (w, D, D)   eje paralelo al largo (X)
    parallel_to_side  -> (D, w, D)   eje paralelo al ancho (Y)

El decodificador es una heurística constructiva determinista de PUNTOS EXTREMOS
(extreme-point / deepest-bottom-left-fill): dada la permutación y las
orientaciones, coloca las ruedas en ese orden escogiendo, para cada una, el
primer punto extremo factible en orden (z, y, x).  Nunca devuelve una solución
infactible: si una rueda no cabe, simplemente no se carga.

NOTA: este módulo sólo usa la biblioteca estándar de Python.
"""

import bisect

from . import data

EPS = 1e-9

_TYPES = data.wheel_types()

# Tabla precalculada _EXT[orientacion][tipo] = (ex, ey, ez) en metros.
_EXT = []
for _o in range(data.N_ORIENT):
    _row = []
    for _t in _TYPES:
        _D = _t["diameter_m"]
        _W = _t["width_m"]
        if _o == 0:          # flat
            _row.append((_D, _D, _W))
        elif _o == 1:        # parallel_to_end
            _row.append((_W, _D, _D))
        else:                # parallel_to_side
            _row.append((_D, _W, _D))
    _EXT.append(_row)

# Menor arista posible de cualquier caja.  Ninguna rueda cabe en un hueco que
# deje menos de esta distancia hasta la pared del camión, así que los puntos
# extremos situados más allá se descartan directamente (ahorro de tiempo).
_MIN_EXT = min(min(_r) for _row in _EXT for _r in _row)


def aabb_extents(orient_idx, type_idx):
    """Dimensiones totales (ex, ey, ez) de la AABB de una rueda, en metros."""
    return _EXT[orient_idx][type_idx]


class Decoder:
    """Decodificador (perm, orient) -> carga factible.

    Parámetros
    ----------
    cell    : tamaño de celda de la rejilla espacial usada para acelerar el
              test de solape.
    max_ep  : número máximo de puntos extremos mantenidos en memoria.
    max_try : número máximo de puntos extremos que se prueban por rueda.
    """

    def __init__(self, truck=None, cell=0.5, max_ep=600, max_try=150):
        self.truck = truck or data.TRUCK
        self.L = self.truck["length"]
        self.Wd = self.truck["width"]
        self.H = self.truck["height"]
        self.cell = cell
        self.max_ep = max_ep
        self.max_try = max_try
        self.ncx = int(self.L / cell) + 1
        self.ncy = int(self.Wd / cell) + 1
        self.ncz = int(self.H / cell) + 1
        self._vol = [_t["volume_m3"] for _t in _TYPES]

    # ------------------------------------------------------------------ #
    def decode(self, genotype, want_placements=False):
        """Devuelve (volume, used, placements).

        volume     : volumen total de CILINDROS cargados (m³).
        used       : lista con las unidades cargadas de cada tipo.
        placements : lista de dicts con la posición/orientación de cada rueda,
                     o None si want_placements es False (se usa durante la
                     búsqueda para no gastar tiempo construyéndola).
        """
        perm, orient = genotype
        L = self.L
        Wd = self.Wd
        H = self.H
        cell = self.cell
        ncx = self.ncx
        ncy = self.ncy
        ncz = self.ncz
        max_try = self.max_try
        max_ep = self.max_ep
        vol_table = self._vol
        avail_per = data.AVAILABLE_PER_TYPE
        n_types = data.N_TYPES

        # Un punto extremo sólo es útil si todavía deja _MIN_EXT hasta el borde.
        xlim = L - _MIN_EXT + EPS
        ylim = Wd - _MIN_EXT + EPS
        zlim = H - _MIN_EXT + EPS

        grid = {}
        bx0 = []
        by0 = []
        bz0 = []
        bx1 = []
        by1 = []
        bz1 = []
        btype = []
        borient = []
        used = [0] * n_types
        volume = 0.0

        # Puntos extremos ordenados por (z, y, x): empaquetado
        # "lo más profundo, abajo y a la izquierda" posible.
        ep = [(0.0, 0.0, 0.0)]
        ep_set = {(0.0, 0.0, 0.0)}

        for inst in perm:
            o = orient[inst]
            if o < 0:                      # data.SKIP -> rueda no seleccionada
                continue
            t = inst // avail_per
            if t >= n_types:
                continue
            ex, ey, ez = _EXT[o][t]
            if ex > L + EPS or ey > Wd + EPS or ez > H + EPS:
                continue

            tried = 0
            for pt in ep:
                if tried >= max_try:
                    break
                tried += 1
                pz, py, px = pt
                x1 = px + ex
                if x1 > L + EPS:
                    continue
                y1 = py + ey
                if y1 > Wd + EPS:
                    continue
                z1 = pz + ez
                if z1 > H + EPS:
                    continue

                # ---- test de solape con rejilla espacial ---------------- #
                ix0 = int(px / cell)
                ix1 = int(x1 / cell)
                if ix1 >= ncx:
                    ix1 = ncx - 1
                iy0 = int(py / cell)
                iy1 = int(y1 / cell)
                if iy1 >= ncy:
                    iy1 = ncy - 1
                iz0 = int(pz / cell)
                iz1 = int(z1 / cell)
                if iz1 >= ncz:
                    iz1 = ncz - 1

                clash = False
                for ix in range(ix0, ix1 + 1):
                    for iy in range(iy0, iy1 + 1):
                        for iz in range(iz0, iz1 + 1):
                            lst = grid.get((ix, iy, iz))
                            if not lst:
                                continue
                            for bi in lst:
                                if (bx0[bi] < x1 - EPS and bx1[bi] > px + EPS and
                                        by0[bi] < y1 - EPS and by1[bi] > py + EPS and
                                        bz0[bi] < z1 - EPS and bz1[bi] > pz + EPS):
                                    clash = True
                                    break
                            if clash:
                                break
                        if clash:
                            break
                    if clash:
                        break
                if clash:
                    continue

                # ---- colocar la caja ------------------------------------ #
                bi = len(bx0)
                bx0.append(px)
                by0.append(py)
                bz0.append(pz)
                bx1.append(x1)
                by1.append(y1)
                bz1.append(z1)
                btype.append(t)
                borient.append(o)
                for ix in range(ix0, ix1 + 1):
                    for iy in range(iy0, iy1 + 1):
                        for iz in range(iz0, iz1 + 1):
                            key = (ix, iy, iz)
                            lst = grid.get(key)
                            if lst is None:
                                grid[key] = [bi]
                            else:
                                lst.append(bi)
                used[t] += 1
                volume += vol_table[t]

                # ---- actualizar puntos extremos -------------------------- #
                ep.remove(pt)
                ep_set.discard(pt)
                if x1 <= xlim:
                    p = (pz, py, x1)
                    if p not in ep_set:
                        bisect.insort(ep, p)
                        ep_set.add(p)
                if y1 <= ylim:
                    p = (pz, y1, px)
                    if p not in ep_set:
                        bisect.insort(ep, p)
                        ep_set.add(p)
                if z1 <= zlim:
                    p = (z1, py, px)
                    if p not in ep_set:
                        bisect.insort(ep, p)
                        ep_set.add(p)

                if len(ep) > max_ep:
                    del ep[max_ep:]
                    ep_set = set(ep)
                break

        if not want_placements:
            return volume, used, None

        placements = []
        for bi in range(len(bx0)):
            placements.append({
                "type_idx": btype[bi],
                "orient_idx": borient[bi],
                "cx": (bx0[bi] + bx1[bi]) / 2.0,
                "cy": (by0[bi] + by1[bi]) / 2.0,
                "cz": (bz0[bi] + bz1[bi]) / 2.0,
                "ex": bx1[bi] - bx0[bi],
                "ey": by1[bi] - by0[bi],
                "ez": bz1[bi] - bz0[bi],
            })
        return volume, used, placements


def check_feasibility(placements, truck=None):
    """Verifica de forma INDEPENDIENTE que una solución es factible.

    Comprueba las tres condiciones exigidas por el enunciado:
      1. cada rueda está completamente dentro del espacio útil del camión;
      2. no hay solapes entre cajas (AABB disjuntas);
      3. no se usan más unidades de cada tipo de las disponibles.

    Devuelve un dict con `ok` y el detalle de las violaciones encontradas.
    """
    t = truck or data.TRUCK
    L, Wd, H = t["length"], t["width"], t["height"]
    boxes = []
    out_of_bounds = 0
    used = [0] * data.N_TYPES
    for p in placements:
        x0 = p["cx"] - p["ex"] / 2.0
        x1 = p["cx"] + p["ex"] / 2.0
        y0 = p["cy"] - p["ey"] / 2.0
        y1 = p["cy"] + p["ey"] / 2.0
        z0 = p["cz"] - p["ez"] / 2.0
        z1 = p["cz"] + p["ez"] / 2.0
        if x0 < -EPS or y0 < -EPS or z0 < -EPS or x1 > L + EPS or y1 > Wd + EPS or z1 > H + EPS:
            out_of_bounds += 1
        boxes.append((x0, y0, z0, x1, y1, z1))
        used[p["type_idx"]] += 1

    overlaps = 0
    n = len(boxes)
    for i in range(n):
        xi0, yi0, zi0, xi1, yi1, zi1 = boxes[i]
        for j in range(i + 1, n):
            xj0, yj0, zj0, xj1, yj1, zj1 = boxes[j]
            if (xi0 < xj1 - EPS and xi1 > xj0 + EPS and
                    yi0 < yj1 - EPS and yi1 > yj0 + EPS and
                    zi0 < zj1 - EPS and zi1 > zj0 + EPS):
                overlaps += 1

    over_stock = sum(max(0, used[k] - data.AVAILABLE_PER_TYPE) for k in range(data.N_TYPES))
    return {
        "ok": (out_of_bounds == 0 and overlaps == 0 and over_stock == 0),
        "n_wheels": n,
        "out_of_bounds": out_of_bounds,
        "overlaps": overlaps,
        "over_stock": over_stock,
        "used": used,
    }


def utilisation(volume_loaded):
    """Aprovechamiento = volumen de cilindros cargado / volumen del camión."""
    return volume_loaded / data.truck_volume()


def empty_space(volume_loaded):
    """Espacio vacío (m³) = volumen del camión - volumen cargado.
    Es la función objetivo (se minimiza)."""
    return data.truck_volume() - volume_loaded
