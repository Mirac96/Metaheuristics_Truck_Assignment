# Práctica: Optimización metaheurística de la carga de ruedas en un camión

**Asignatura:** Metaheurísticas &nbsp;·&nbsp; **Grupo:** (completar nombres)

**Metaheurísticas:** Algoritmo Genético (GA), Harmony Search (HS) e Iterated Greedy (IG).

> **Nota sobre dependencias.** Todo el trabajo está implementado desde cero con la
> **biblioteca estándar de Python**: el modelo geométrico, el decodificador, los
> operadores, las tres metaheurísticas, el diseño experimental y el análisis
> estadístico (Jarque-Bera, ANOVA, Kruskal-Wallis, Mann-Whitney, Welch y la
> corrección de Holm-Bonferroni están programados a mano en
> `src/stats_tools.py`). No se emplea ningún paquete de optimización (DEAP,
> pymoo, mealpy…), ninguno estadístico (scipy, statsmodels) ni numpy/pandas.
> `matplotlib` se usa únicamente —y de forma opcional— para generar los PNG de
> apoyo; el entregable de visualización son los tres ficheros JSON.

---

## 1. Modelización matemática

### 1.1 Datos

- Camión (espacio útil rectangular):  
  \(L = 13{,}62\ \text{m}\), \(W = 2{,}48\ \text{m}\), \(H = 2{,}65\ \text{m}\);
  volumen \(V_T = 89{,}51\ \text{m}^3\).
- Ruedas: 12 tipos \(k = 1,\dots,12\) (\(R_1 \dots R_{12}\)), 15 unidades de cada
  uno, 180 en total. Cada tipo se define por su diámetro exterior nominal
  \(D_k\) y su anchura \(w_k\).
- Volumen total disponible: \(103{,}35\ \text{m}^3 > V_T\). **No caben todas**:
  la decisión de selección es relevante, y una solución con más ruedas no tiene
  por qué ser mejor.

### 1.2 Hipótesis geométricas

1. **La rueda es un cilindro** de diámetro \(D_k\) y grosor \(w_k\). Es el modelo
   físico real y el que define el volumen realmente transportado.
2. **Representación para el no solape: caja axis-alineada (AABB).** Cada rueda se
   representa por la caja envolvente axis-alineada correspondiente a su
   orientación. Si dos AABB son disjuntas, los cilindros que contienen también
   lo son (*condición suficiente*). Esto garantiza la factibilidad con un coste
   de comprobación \(O(1)\) por par de cajas y evita tests cilindro–cilindro
   costosos. El precio es conservador: se desperdicia el volumen de las esquinas
   de la caja. Es la aproximación estándar en problemas de *packing* con
   metaheurísticas cuando se exige garantía de no solape.
3. **Posición = centro de la AABB**, con origen en la esquina
   trasera–izquierda–inferior y ejes \(X\)=largo, \(Y\)=ancho, \(Z\)=alto. Es
   exactamente el formato exigido en el apartado 13.

### 1.3 Consecuencia importante de la hipótesis 2

El volumen del cilindro es \(V^c_k = \frac{\pi}{4}D_k^2 w_k\), mientras que el
volumen de la AABB es \(V^a_k = D_k^2 w_k\) **en cualquiera de las tres
orientaciones** (la caja siempre tiene lados \(D_k, D_k, w_k\), sólo permutados).
Por tanto

\[
V^c_k = \frac{\pi}{4}\,V^a_k \quad \forall k,\ \forall \text{orientación}
\;\Longrightarrow\;
\sum_{i\in S} V^c_{t(i)} = \frac{\pi}{4}\sum_{i\in S} V^a_{t(i)} .
\]

**Minimizar el espacio vacío equivale exactamente a maximizar el volumen total de
cajas empaquetado.** La orientación no altera el volumen que aporta una rueda,
pero sí su *forma*, y por tanto condiciona qué ruedas caben y cuánto espacio se
desperdicia. Ésta es la razón por la que la orientación es una decisión
relevante aunque no aparezca explícitamente en la función objetivo.

### 1.4 Orientaciones

\((e_x,e_y,e_z)\) = dimensiones totales de la AABB en cada eje:

| Orientación | Significado | \((e_x,e_y,e_z)\) |
|---|---|---|
| `flat` | tumbada, eje vertical (\(Z\)) | \((D_k, D_k, w_k)\) |
| `parallel_to_end` | vertical, eje paralelo al largo (\(X\)) | \((w_k, D_k, D_k)\) |
| `parallel_to_side` | vertical, eje paralelo al ancho (\(Y\)) | \((D_k, w_k, D_k)\) |

### 1.5 Variables de decisión

Se numeran las \(N=180\) instancias de rueda (\(i = 0,\dots,179\)); la instancia
\(i\) pertenece al tipo \(t(i) = \lfloor i/15\rfloor\). El genotipo es:

- **Orden de colocación** \(\pi\): permutación de \(\{0,\dots,179\}\).
- **Orientación / selección** \(o_i \in \{0,1,2,\texttt{SKIP}\}\): orientación de
  la instancia \(i\); \(o_i = \texttt{SKIP}\) significa que **no se carga**.

La posición \(\mathbf{p}_i\) no es una variable libre: la determina de forma
**determinista** el decodificador (§1.7). Las tres decisiones del enunciado
quedan así cubiertas: selección (\(o_i = \texttt{SKIP}\)), orientación (\(o_i\))
y posicionamiento (decodificador aplicado al orden \(\pi\)).

### 1.6 Función objetivo

\[
\min\ f(\pi,o) \;=\; V_T - \sum_{i\in S(\pi,o)} V^c_{t(i)},
\qquad
\text{con } \text{aprovechamiento } = \frac{\sum_{i\in S} V^c_{t(i)}}{V_T}.
\]

Las tres metaheurísticas optimizan **exactamente** esta función.

### 1.7 Restricciones y tratamiento de la infactibilidad

Para cada rueda cargada:

- **Contención:** \(0 \le x_i - e_x/2\) y \(x_i + e_x/2 \le L\) (y análogo en
  \(Y\), \(Z\)).
- **No solape:** las AABB de dos ruedas cualesquiera deben ser disjuntas
  (intersección vacía en al menos un eje).

El **decodificador garantiza ambas por construcción**: recorre \(\pi\) y sólo
coloca una rueda si cabe íntegra en el camión y no solapa con las ya colocadas;
si no, la rueda simplemente no se carga. En consecuencia **no existe ninguna
solución infactible** en todo el proceso de búsqueda: no se necesita penalización
ni reparación, y todas las soluciones devueltas son directamente utilizables.

### 1.8 Decodificador (posicionamiento)

Heurística constructiva determinista de **puntos extremos** (*deepest-bottom-left-fill*):

- Se mantiene un conjunto de puntos extremos (esquinas candidatas), inicializado
  con \((0,0,0)\) y ordenado por \((z,y,x)\), y una rejilla espacial para
  acelerar el test de solape.
- Para cada rueda, en el orden \(\pi\), se prueba el primer punto extremo
  factible en ese orden y se coloca allí; tras colocarla se añaden tres nuevos
  puntos extremos en las caras de la caja y se descartan los que quedan
  encerrados o superan el número máximo.
- Coste típico: ~12–40 ms por decodificación (180 ruedas), lo que permite
  decenas de miles de evaluaciones.

---

## 2. Las tres metaheurísticas

Las tres usan la **misma representación**, la **misma función objetivo**, el
**mismo decodificador** y el **mismo presupuesto de evaluaciones**. Sólo cambia
el mecanismo de búsqueda.

### 2.0 Generación de soluciones iniciales (común a las tres)

El orden en que el decodificador intenta colocar las ruedas es la decisión que
más influye en el resultado: si las ruedas grandes llegan al final, el espacio ya
está fragmentado y no caben. Por eso las tres metaheurísticas parten de un
generador común **aleatorizado pero sesgado**:

- se ordenan las 180 instancias por volumen de cilindro **decreciente**;
- cada clave se multiplica por un factor aleatorio \(1 + U(-\rho,\rho)\), con
  \(\rho = 0{,}25\), de modo que el orden resultante es "casi decreciente" pero
  distinto en cada ejecución y en cada individuo;
- la orientación de cada rueda se elige al azar entre las tres, con un 10 % de
  probabilidad de quedar marcada como `SKIP`.

Con \(\rho = 0\) se obtendría el orden puramente greedy (espacio vacío ≈
30,8 m³); con \(\rho = 1\) el orden sería casi aleatorio (≈ 47 m³). El valor
elegido sitúa el punto de partida en ≈ 31,9 m³ y deja margen real de mejora a
la búsqueda. **Las tres metaheurísticas usan exactamente el mismo generador**,
de modo que la inicialización no favorece a ninguna y la comparación sigue
siendo justa.

### 2.1 Algoritmo Genético (GA) — paradigma *poblacional*

| Aspecto | Diseño |
|---|---|
| Representación | \((\pi, o)\) |
| Soluciones iniciales | `pop` soluciones del generador común (§2.0) |
| Selección | torneo de tamaño `tournament` (compara espacio vacío) |
| Cruce | **OX** (*Order Crossover*) sobre \(\pi\) + cruce uniforme sobre \(o\) (indexada por rueda) |
| Mutación | por posición: intercambio en \(\pi\) (`p_perm`), nueva orientación (`p_orient`), conmutar `SKIP` (`p_skip`) |
| Elitismo | los `elite` mejores pasan sin alterar |
| Exploración | cruce OX entre individuos distintos + mutación |
| Explotación | presión selectiva del torneo + elitismo |
| Infactibilidad | imposible (decodificador) |
| Parámetros | `pop`, `p_cross`, `p_perm`, `p_orient`, `p_skip`, `tournament`, `elite` |
| Parada | presupuesto de \(E\) evaluaciones |

### 2.2 Harmony Search (HS) — paradigma de *memoria de armonías*

HS es una metaheurística inspirada en la improvisación musical. Aquí cada
"armonía" es una solución completa \((\pi,o)\) y cada "instrumento" es una
variable de decisión. Adaptación al problema:

| Aspecto | Diseño |
|---|---|
| Representación | \((\pi, o)\) — cada armonía de la memoria es una solución completa |
| Inicialización | memoria de `hms` armonías del generador común (§2.0) |
| Improvisación (permutación) | se toma una armonía de referencia; **con probabilidad \(1-\)HMCR** el contenido de cada posición se sustituye por una rueda elegida al azar (*selección aleatoria*), implementada como intercambio para preservar la validez de la permutación |
| Ajuste de tono (permutación) | **con probabilidad PAR** se intercambia la posición \(i\) con otra situada a distancia \(\le\) `bw` (*ancho de banda*) |
| Improvisación (orientaciones) | con probabilidad \(1-\)HMCR valor aleatorio; **con probabilidad `par_o`** se "afina": se pasa a una orientación vecina, se activa una rueda no cargada o se desactiva una cargada |
| Actualización de la memoria | si la nueva armonía mejora la peor de la memoria, la sustituye |
| Exploración | selección aleatoria (\(1-\)HMCR) y renovación continua de la memoria |
| Explotación | alta tasa de consideración de memoria (HMCR), ajuste de tono local (PAR/`bw`) y actualización elitista de la memoria |
| Infactibilidad | imposible (decodificador) |
| Parámetros | `hms`, `hmcr`, `par`, `bw`, `par_o`, `p_skip_rand` |
| Parada | presupuesto de \(E\) evaluaciones |

### 2.3 Iterated Greedy (IG) — paradigma de *destrucción–construcción*

| Aspecto | Diseño |
|---|---|
| Representación | \((\pi, o)\) |
| Solución inicial | del generador común (§2.0), igual que GA y HS |
| Destrucción | se eligen \(d\) ruedas actualmente cargadas al azar y se marcan `SKIP` (se retiran de la carga) |
| Construcción | se reinsertan una a una en orden aleatorio: para cada rueda se generan `n_try` candidatos (posición aleatoria en \(\pi\) + orientación aleatoria) y **se queda con el mejor** (*inserción voraz*) |
| Búsqueda local | `ls` movimientos de primer-mejor sobre el vecindario (intercambio en \(\pi\), cambio de orientación, conmutar `SKIP`, inversión de bloque) |
| Aceptación | se acepta la solución reconstruida si no empeora a la actual; en caso contrario se descarta. Si lleva `patience` iteraciones sin mejorar el mejor conocido, se reinicia desde éste |
| Exploración | destrucción aleatoria y reinicio desde la mejor solución |
| Explotación | reinserción voraz + búsqueda local |
| Infactibilidad | imposible (decodificador) |
| Parámetros | `d`, `n_try`, `ls`, `patience` |
| Parada | presupuesto de \(E\) evaluaciones |

### 2.4 Justificación de la elección

Las tres pertenecen a **paradigmas de búsqueda distintos**, que es lo que pide
el enunciado para que la comparación resulte de interés:

- **GA**: mantiene una *población* y recombina información de varios individuos.
- **HS**: mantiene una *memoria* de soluciones y genera una nueva cada vez
  combinando memoria, ajuste local y aleatoriedad (un único punto de búsqueda
  "virtual" por iteración, pero información acumulada en la memoria).
- **IG**: trabaja sobre una *única solución* mediante ciclos de
  destrucción–construcción con reconstrucción voraz.

Sólo el GA coincide con los métodos explicados en clase, cumpliendo la
restricción "tan sólo una de las tres podrá coincidir con las explicadas por el
profesor". Las tres están adaptadas específicamente al problema: la permutación
se manipula con operadores que la mantienen válida (OX, intercambios,
inserción), y la componente discreta (orientación + selección) se trata por
separado con operadores propios.

---

## 3. Ajuste de parámetros

- Se estudian los parámetros **propios de cada algoritmo** (los indicados en
  §2), que son los que controlan el equilibrio exploración/explotación.
- Procedimiento: **rejilla completa** evaluada con las **mismas 3 semillas** y
  el mismo presupuesto reducido para las tres metaheurísticas, seleccionando la
  configuración con **menor espacio vacío medio** (se evita elegir por el mejor
  resultado puntual, que premiaría la suerte).
- Las tres rejillas tienen el **mismo número de configuraciones**, de modo que
  ninguna metaheurística recibe un esfuerzo de calibración sustancialmente
  superior (apartado 7).
- Una vez fijados, los parámetros se **mantienen constantes** en las 30
  ejecuciones comparativas.

---

## 4. Comparación equitativa

Se garantiza la equivalencia mediante:

- el mismo camión, las mismas 180 ruedas y el mismo problema;
- la misma representación, función objetivo, restricciones y decodificador;
- el **mismo presupuesto computacional**: \(E\) evaluaciones por ejecución,
  contando **todas** las decodificaciones (incluidas las de la búsqueda local de
  IG y las de la población inicial), a través de un único contador;
- las **mismas 30 semillas** para los tres algoritmos, conservadas para
  reproducir los resultados;
- el mismo equipo y la misma sesión de ejecución.

---

## 5. Ejecuciones independientes e información registrada

- **30 ejecuciones independientes por algoritmo** (90 en total), semillas
  `1000 … 1029`, idénticas para los tres.
- `results/results.csv` registra por ejecución: metaheurística, semilla, valor
  de la función objetivo (espacio vacío), aprovechamiento, nº total de ruedas,
  unidades cargadas de cada tipo (\(R_1…R_{12}\)), nº de evaluaciones, tiempo y
  factibilidad.
- La mejor configuración de cada algoritmo se guarda completa en
  `best_<ALG>.json` (formato apartado 13), que permite **reconstruirla por
  completo**.

---

## 6. Análisis experimental y comparación estadística

- Estadísticos por algoritmo: mejor, peor, media, mediana, desviación típica,
  rango intercuartílico, rango, tiempo medio y utilización media.
- **Calidad sistemática frente a aciertos puntuales**: se mide el porcentaje de
  ejecuciones que quedan dentro del 2 % del mejor valor global y el porcentaje
  que supera la mediana global.
- **Convergencia**: curvas medias del mejor valor frente a evaluaciones y
  número medio de evaluaciones necesarias para alcanzar el 95 % de la mejora
  total de cada ejecución.
- **Contraste formal** (todo programado a mano):
  1. *Normalidad*: Jarque-Bera por grupo.
  2. *Contraste global*: ANOVA de un factor si los tres grupos son normales; en
     caso contrario **Kruskal-Wallis** (no paramétrico, adecuado porque los
     resultados de metaheurísticas suelen ser asimétricos).
  3. *Comparaciones por pares*: t de Welch o Mann-Whitney U, con corrección
     **Holm-Bonferroni** para las comparaciones múltiples (α = 0,05).
- Se reportan hipótesis, nivel de significación, pruebas, estadísticos,
  p-valores, p-valores ajustados e interpretación. **No se comparan sólo medias
  ni sólo mejores valores.**

Los resultados numéricos se generan automáticamente en `results/report.md` y
`results/statistics.json`.

---

## 7. Visualización

- Tres ficheros JSON (`best_GA.json`, `best_HS.json`, `best_IG.json`) con las
  coordenadas del centro de cada rueda (m), su tipo y su orientación, listos
  para el software de visualización 3D. Cada fichero respeta las dimensiones del
  camión, el inventario disponible y la no superposición (garantizado por
  construcción).
- Además se generan `packing_<ALG>.png` (contorno del camión + cajas de cada
  rueda coloreadas por tipo; las proporciones de la caja revelan la orientación
  y se ven los huecos libres) y `convergence.png`.

---

## 8. Respuestas a las cuestiones del enunciado

Las respuestas **cuantitativas** se generan automáticamente en
`results/report.md`; aquí se recoge el razonamiento:

1. **Modelización**: cilindro + AABB según orientación; genotipo \((\pi,o)\);
   posicionamiento por decodificador de puntos extremos (§1).
2. **Espacio vacío**: \(V_T - \sum_i V^c_{t(i)}\); equivalentemente, \(1\) menos
   el aprovechamiento.
3. **Representación de una solución**: permutación de las 180 instancias +
   vector de orientaciones con centinela `SKIP`.
4. **Selección de ruedas**: mediante `SKIP` en el vector de orientaciones y por
   el propio orden (una rueda que no cabe al final de la secuencia queda fuera).
5. **Orientaciones**: `flat`, `parallel_to_end`, `parallel_to_side`, cada una con
   su AABB asociada (§1.4).
6. **Garantía de no solape**: AABB disjuntas, comprobadas en el decodificador con
   una rejilla espacial; ninguna solución evaluada es infactible.
7. **Elección de las tres**: tres paradigmas distintos (poblacional, memoria,
   destrucción–construcción); sólo el GA coincide con los visto en clase (§2.4).
8. **Exploración / explotación**: descritas para cada algoritmo en §2.
9–12. **Aprovechamiento, variabilidad, convergencia y significación**: ver
`results/report.md` (tablas y contrastes).
13. **Combinaciones similares**: se comparan las unidades por tipo de las tres
   mejores soluciones (`results/report.md`, §4).
14. **Tipos más frecuentes**: se listan los tipos con más unidades en las mejores
   soluciones. Como el volumen crece con \(D^2 w\), los tipos grandes (R10–R12)
   dominan en volumen aunque no en número.
15–16. **Combinar ruedas grandes y pequeñas**: las grandes aportan mucho volumen
   pero dejan huecos irregulares; las pequeñas (R1–R5) son las que pueden
   rellenar esos huecos. El decodificador y la búsqueda explotan precisamente
   esa combinación.
17–18. **Orientaciones predominantes y relación con el tamaño**: se reporta la
   distribución de orientaciones y el porcentaje de ruedas grandes en `flat`.
   Cabe esperar que las ruedas grandes se coloquen tumbadas (`flat`), porque así
   su altura es sólo \(w_k\) y se pueden apilar en varias capas dentro de los
   2,65 m de altura útil, mientras que en vertical (\(D_k\) de alto) una rueda
   como la R12 (\(D = 2{,}06\) m) consumiría casi toda la altura.
19. **¿Más ruedas ⇒ mejor solución?**: no necesariamente; se cuantifica con la
   correlación de Pearson entre nº de ruedas y espacio vacío sobre las 90
   ejecuciones. El objetivo es volumen, no número de unidades.
20. **Recomendación**: ver `results/report.md` (se decide con la calidad media,
   la variabilidad y la velocidad de convergencia conjuntamente).

---

## 9. Entregables y reproducibilidad

- **Código**: `main.py` + `src/` (`data`, `geometry`, `solution`,
  `metaheuristics`, `stats_tools`, `experiment`, `analysis`, `visualize`).
- **Datos**: `requirements.txt` (sólo matplotlib, opcional).
- **Resultados**: `results/results.csv`, `results/report.md`,
  `results/statistics.json`, `results/convergence.png`, `results/packing_*.png`.
- **Mejores configuraciones**: `results/best_GA.json`, `results/best_HS.json`,
  `results/best_IG.json`.
- **Reproducir**: `python main.py` (experimento completo con ajuste de
  parámetros); `python main.py --quick` para una validación rápida.
