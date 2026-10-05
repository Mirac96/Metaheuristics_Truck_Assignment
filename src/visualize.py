"""
Visualización de las mejores soluciones y de las curvas de convergencia.

Se dibuja cada rueda mediante su AABB (caja axis-alineada) como arista de
alambre, coloreada por tipo.  Las proporciones de la caja revelan la
orientación:

  - flat              : caja ancha y poco alta (eje vertical)
  - parallel_to_end   : estrecha en X, alta
  - parallel_to_side  : estrecha en Y, alta

También se muestra el contorno del camión, de modo que pueden identificarse los
límites, las ruedas cargadas, sus tamaños, posiciones, orientaciones y los
huecos libres.

NOTA: matplotlib es OPCIONAL y se usa ÚNICAMENTE para generar estos PNG de
apoyo.  El entregable de visualización del enunciado son los ficheros JSON, que
se generan sin ninguna dependencia externa.  Si matplotlib no está instalado,
el experimento se ejecuta igual y esta parte se omite.
"""

import os

from . import data


def _load_matplotlib():
    """Importa matplotlib bajo demanda.  Devuelve None si no está disponible."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def _box_edges(cx, cy, cz, ex, ey, ez):
    """Las 12 aristas de una caja como segmentos de 2 puntos."""
    x0, x1 = cx - ex / 2.0, cx + ex / 2.0
    y0, y1 = cy - ey / 2.0, cy + ey / 2.0
    z0, z1 = cz - ez / 2.0, cz + ez / 2.0
    corners = [
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    return [[corners[a], corners[b]] for a, b in edges]


def plot_solution(eval_info, meta_name, out_dir):
    """PNG 3D del empaquetado.  Devuelve la ruta o None si no hay matplotlib."""
    plt = _load_matplotlib()
    if plt is None:
        return None
    try:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
    except Exception:
        return None

    cmap = plt.cm.tab20
    colors = [cmap(float(i) / max(1, data.N_TYPES - 1)) for i in range(data.N_TYPES)]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    L, Wd, H = data.TRUCK["length"], data.TRUCK["width"], data.TRUCK["height"]
    ax.add_collection3d(Line3DCollection(
        _box_edges(L / 2, Wd / 2, H / 2, L, Wd, H),
        colors="black", linewidths=1.5, linestyles="dashed"))

    seg_list = []
    seg_colors = []
    for p in eval_info["placements"]:
        segs = _box_edges(p["cx"], p["cy"], p["cz"], p["ex"], p["ey"], p["ez"])
        seg_list.extend(segs)
        seg_colors.extend([colors[p["type_idx"]]] * len(segs))
    if seg_list:
        ax.add_collection3d(Line3DCollection(seg_list, colors=seg_colors,
                                             linewidths=0.8, alpha=0.9))

    ax.set_xlim(0, L)
    ax.set_ylim(0, Wd)
    ax.set_zlim(0, H)
    ax.set_xlabel("X (largo, m)")
    ax.set_ylabel("Y (ancho, m)")
    ax.set_zlabel("Z (alto, m)")
    ax.set_title("%s — %d ruedas, aprovechamiento %.1f%%\nespacio vacio = %.2f m³"
                 % (meta_name, eval_info["n_wheels"],
                    100 * eval_info["utilisation"], eval_info["objective"]))

    handles = [plt.Line2D([0], [0], color=colors[i], lw=3,
                          label=data.WHEEL_SPECS[i][0]) for i in range(data.N_TYPES)]
    ax.legend(handles=handles, loc="upper left", fontsize=7, ncol=2, title="Tipo")

    path = os.path.join(out_dir, "packing_%s.png" % meta_name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_convergence(conv_means, out_dir):
    """PNG con las curvas de convergencia medias."""
    plt = _load_matplotlib()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, curve in conv_means.items():
        ax.plot(range(1, len(curve) + 1), curve, label=m)
    ax.set_xlabel("Evaluaciones")
    ax.set_ylabel("Mejor espacio vacío (m³) — menor es mejor")
    ax.set_title("Curvas de convergencia medias (30 ejecuciones)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(out_dir, "convergence.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
