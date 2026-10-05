# Resultados experimentales — Carga de ruedas en un camión

Generado automáticamente por `main.py`.

- Camión: 13.62 x 2.48 x 2.65 m (volumen **89.51 m³**).
- Ruedas disponibles: 180 (12 tipos x 15 uds.). Volumen total disponible 103.35 m³ > volumen del camión => la **selección** es decisiva.

- Metaheurísticas comparadas: **GA, HS, IG**.
- Ejecuciones: 30 independientes por algoritmo (90 en total), semillas 1000...1029, presupuesto de 1200 evaluaciones por ejecución.

## 1. Estadísticos por metaheurística (espacio vacío, m³; menor = mejor)

| Algoritmo | Mejor | Peor | Media | Mediana | D.típica | IQR | Rango | Utiliz. media | Ruedas (media) | Tiempo medio (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| **GA** | 26.240 | 27.788 | 27.183 | 27.107 | 0.363 | 0.597 | 1.548 | 69.63% | 115.4 | 12.78 |
| **HS** | 26.816 | 28.788 | 27.685 | 27.723 | 0.416 | 0.573 | 1.973 | 69.07% | 115.4 | 16.10 |
| **IG** | 27.351 | 30.857 | 29.110 | 29.020 | 0.909 | 1.370 | 3.505 | 67.48% | 121.7 | 25.05 |

### 1.1 ¿Calidad puntual o calidad sistemática?

| Algoritmo | %% ejecuciones <= 102%% del mejor global | %% ejecuciones por debajo de la mediana global |
|---|---|---|
| GA | 6.7% | 93.3% |
| HS | 0.0% | 53.3% |
| IG | 0.0% | 3.3% |

Un algoritmo con buen *mejor* valor pero bajo porcentaje en la primera columna obtiene buenas soluciones de forma **ocasional**; un porcentaje alto indica buen comportamiento **sistemático**.

## 2. Análisis estadístico

**Hipótesis.**  H0: las tres metaheurísticas producen la misma distribución de espacio vacío.  H1: al menos una difiere.  Nivel de significación α = 0.05.

**Normalidad (Jarque-Bera).**  Se usa porque la aproximación asintótica del ANOVA exige normalidad y los resultados de metaheurísticas suelen ser asimétricos.

| Algoritmo | JB | p | Conclusión |
|---|---|---|---|
| GA | 0.256 | 0.8799 | normal |
| HS | 0.397 | 0.8201 | normal |
| IG | 0.899 | 0.6378 | normal |

**Contraste global:** ANOVA de un factor (F = 79.4567, p = 2.345e-20) -> se rechaza H0.

Rangos medios (Kruskal-Wallis, menor = mejor): GA = 21.40, HS = 42.00, IG = 73.10

**Comparaciones por pares.**  Se aplica t de Welch y se corrige el nivel de significación por comparaciones múltiples con **Holm-Bonferroni** (Bonferroni clásico exigiría α = 0.0167).

| Pareja | Estadístico | p | p ajustado (Holm) | Significativo |
|---|---|---|---|---|
| GA vs HS | t (Welch) = -4.984 | 6.162e-06 | 6.162e-06 | SÍ |
| GA vs IG | t (Welch) = -10.781 | 4.059e-13 | 1.218e-12 | SÍ |
| HS vs IG | t (Welch) = -7.802 | 1.321e-09 | 2.641e-09 | SÍ |

## 3. Convergencia

| Algoritmo | Evaluaciones medias para alcanzar el 95%% de su mejora total |
|---|---|
| GA | 650 |
| HS | 751 |
| IG | 732 |

Al final del presupuesto la mejor curva media es la de **GA** (27.183 m³ de espacio vacío medio).


## 4. Perfil de las mejores soluciones

| Algoritmo | Ruedas | Espacio vacío (m³) | Utilización | flat | parallel_to_end | parallel_to_side | %% ruedas grandes en flat |
|---|---|---|---|---|---|---|---|
| GA | 121 | 26.240 | 70.69% | 39 | 25 | 57 | 36.0% |
| HS | 104 | 26.816 | 70.04% | 29 | 23 | 52 | 25.0% |
| IG | 122 | 27.351 | 69.44% | 51 | 15 | 56 | 37.5% |

Unidades cargadas de cada tipo en la mejor solución:

| Algoritmo | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R10 | R11 | R12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GA | 13 | 14 | 14 | 10 | 9 | 11 | 6 | 8 | 9 | 9 | 7 | 11 |
| HS | 13 | 14 | 6 | 6 | 7 | 6 | 7 | 9 | 9 | 6 | 12 | 9 |
| IG | 12 | 12 | 8 | 13 | 8 | 13 | 10 | 10 | 12 | 7 | 5 | 12 |

## 5. Respuestas a las cuestiones del enunciado

**¿Una solución que transporta más ruedas es necesariamente mejor?**  Correlación de Pearson entre nº de ruedas y espacio vacío sobre las 90 ejecuciones: r = 0.391.  Un valor cercano a 0 (o positivo) indicaría que cargar más ruedas no garantiza menor espacio vacío; un valor claramente negativo indicaría lo contrario.

**Mejor aprovechamiento medio:** **GA** (27.183 m³ de espacio vacío, 69.63% de utilización).

**Menor variabilidad:** **GA** (D.típica = 0.363 m³, IQR = 0.597 m³).

**Peor comportamiento medio:** IG (29.110 m³).

**Mejor solución puntual:** GA (26.240 m³).

**Tipos más frecuentes en las mejores soluciones:** R2 (40 uds.), R1 (38 uds.), R12 (32 uds.), R6 (30 uds.), R9 (30 uds.).

**Convergencia más rápida:** **GA** (650 evaluaciones de media para el 95% de su mejora).

**¿Diferencias significativas?**  Sí: ANOVA de un factor rechaza H0 (p = 2.345e-20).  Parejas con diferencia significativa tras Holm: GA vs HS, GA vs IG, HS vs IG.

**Recomendación.**  Considerando conjuntamente la calidad media, la variabilidad y la velocidad de convergencia, se recomienda **GA**.


## 6. Parámetros utilizados

```
GA: {'pop': 20, 'p_cross': 0.9, 'p_perm': 0.01, 'p_orient': 0.02, 'p_skip': 0.01, 'tournament': 3, 'elite': 2}
HS: {'hms': 30, 'hmcr': 0.98, 'par': 0.01, 'bw': 3, 'par_o': 0.01, 'p_skip_rand': 0.15}
IG: {'d': 4, 'n_try': 3, 'ls': 10, 'patience': 20}
```

