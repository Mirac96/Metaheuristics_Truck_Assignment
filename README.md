# Práctica: Optimización metaheurística de la carga de ruedas en un camión

Carga de 180 ruedas (12 tipos x 15 unidades) en un semirremolque de
13,62 x 2,48 x 2,65 m, minimizando el espacio vacío.

**Metaheurísticas implementadas desde cero:**

| Algoritmo | Paradigma de búsqueda |
|---|---|
| **GA** (Algoritmo Genético) | poblacional: selección por torneo + cruce OX + mutación + elitismo |
| **HS** (Harmony Search) | memoria de armonías: consideración de memoria (HMCR) + ajuste de tono (PAR/bw) |
| **IG** (Iterated Greedy) | destrucción - construcción voraz + aceptación por mejora |

> Ninguna de las tres usa bibliotecas externas. Todo el código (incluido el
> análisis estadístico) está escrito con la **biblioteca estándar de Python**.
> `matplotlib` es opcional y sólo genera los PNG de apoyo.

## Estructura

```
main.py                 punto de entrada (experimento completo)
src/data.py             datos del camión y de las ruedas
src/geometry.py         modelo geométrico (cilindro -> AABB) y decodificador
                        de puntos extremos (garantiza factibilidad)
src/solution.py         representación, operadores y evaluación
src/metaheuristics.py   GA, HS e IG
src/stats_tools.py      estadística y contrastes implementados a mano
                        (Jarque-Bera, ANOVA, Kruskal-Wallis, Mann-Whitney,
                        Welch, Holm-Bonferroni)
src/experiment.py       ajuste de parámetros, 30 ejecuciones, CSV y JSON
src/analysis.py         tablas, convergencia e informe automático
src/visualize.py        PNG 3D del empaquetado y curvas de convergencia
REPORT.md               memoria de la práctica
```

## Uso

```bash
python main.py                 # experimento completo: ajuste + 30 semillas
python main.py --no-tune       # sin ajuste de parámetros
python main.py --quick         # validación rápida (5 semillas, 200 eval.)
python main.py --max-evals 2000 --seeds 30
```

## Salidas (en `results/`)

- `results.csv` — una fila por ejecución: metaheurística, semilla, espacio
  vacío, aprovechamiento, nº de ruedas, unidades de cada tipo (R1..R12),
  evaluaciones, tiempo y factibilidad.
- `report.md` / `statistics.json` — estadísticos, contrastes y respuestas.
- `best_GA.json`, `best_HS.json`, `best_IG.json` — mejor solución de cada
  algoritmo en el formato del apartado 13 del enunciado (coord_x/y/z, type,
  orientation), listos para el software de visualización 3D.
- `convergence.png`, `packing_*.png` — gráficos de apoyo.

## Modelo (resumen)

Cada rueda es un cilindro que se representa por su **caja axis-alineada (AABB)**
según la orientación. Como el volumen del cilindro es `(pi/4)·D²·W` y el de su
AABB es `D²·W` (independientemente de la orientación), **maximizar el volumen de
cilindros cargado equivale a maximizar el volumen de cajas empaquetado**: la
orientación influye en la *factibilidad* del empaquetado, no en el objetivo.
El decodificador de puntos extremos garantiza que toda solución evaluada es
factible (dentro del camión y sin solapes).
