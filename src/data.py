"""
Datos reales del camión y de las ruedas disponibles (enunciado de la práctica).

Todas las dimensiones se trabajan en METROS para el cálculo y para la salida JSON
(coord_x, coord_y, coord_z en metros). Las cotas del camión son:

    Longitud (X) : 13 620 mm = 13.62 m
    Anchura (Y)  :  2 480 mm =  2.48 m
    Altura  (Z)  :  2 650 mm =  2.65 m

Cada rueda se modeliza como un CILINDRO de:
    - diámetro exterior nominal D (mm)
    - anchura (grosor) W (mm)

Las ruedas se pueden colocar en tres orientaciones (definidas en el enunciado,
apartado 13, formato JSON):

    flat              : tumbada, eje vertical (Z).
    parallel_to_end   : vertical, eje paralelo al largo del camión (X).
    parallel_to_side  : vertical, eje paralelo al ancho del camión (Y).
"""

# ---- Camión (cotas útiles rectangulares, en metros) -------------------------
TRUCK = {
    "length": 13.62,   # X
    "width": 2.48,     # Y
    "height": 2.65,    # Z
}

# ---- Ruedas disponibles -----------------------------------------------------
# code, width_mm, diameter_mm, available
WHEEL_SPECS = [
    ("R1",  165,  586.6, 15),
    ("R2",  195,  634.5, 15),
    ("R3",  205,  631.9, 15),
    ("R4",  225,  743.9, 15),
    ("R5",  235,  711.9, 15),
    ("R6",  275,  956.5, 15),
    ("R7",  315, 1012.5, 15),
    ("R8",  315, 1075.5, 15),
    ("R9",  385, 1072.0, 15),
    ("R10", 600, 1602.0, 15),
    ("R11", 650, 1911.8, 15),
    ("R12", 710, 2060.8, 15),
]

N_TYPES = len(WHEEL_SPECS)
AVAILABLE_PER_TYPE = WHEEL_SPECS[0][3]            # 15 de cada tipo
TOTAL_WHEELS = sum(s[3] for s in WHEEL_SPECS)    # 180

# Orientaciones (índices internos)
ORIENTATIONS = ["flat", "parallel_to_end", "parallel_to_side"]
N_ORIENT = len(ORIENTATIONS)

# Sentinel para "rueda no cargada" en el genotipo
SKIP = -1


def wheel_types():
    """Devuelve lista de dicts con la información de cada tipo de rueda.

    Cada dict contiene: code, width_m, diameter_m, volume_m3, available.
    """
    types = []
    for code, w_mm, d_mm, avail in WHEEL_SPECS:
        w = w_mm / 1000.0
        d = d_mm / 1000.0
        vol = 3.141592653589793 * (d / 2.0) ** 2 * w   # volumen del cilindro
        types.append({
            "code": code,
            "width_m": w,
            "diameter_m": d,
            "volume_m3": vol,
            "available": avail,
        })
    return types


def truck_volume():
    return TRUCK["length"] * TRUCK["width"] * TRUCK["height"]


def total_available_volume():
    return sum(t["volume_m3"] * t["available"] for t in wheel_types())
