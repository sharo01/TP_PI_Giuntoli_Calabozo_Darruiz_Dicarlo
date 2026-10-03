## Problema 1: Ecualización local de histograma

### Descripción del problema

La ecualización de histograma redistribuye los niveles de gris de una imagen para aprovechar todo el rango, a partir del histograma de la imagen completa. Cuando una imagen tiene zonas cuyos detalles tienen una intensidad muy parecida a la de su entorno, esa ecualización global no alcanza: al usar todos los píxeles de la imagen, pierde la información local.

La ecualización local resuelve este problema. Desplaza una ventana de M×N píxeles a lo largo de la imagen, ecualiza el histograma de cada ventana y usa esa transformación solo para el píxel central. La consigna pide:

- **a)** Una función que implemente la ecualización local y reciba la imagen y el tamaño de la ventana.
- **b)** Aplicarla a `Imagen_con_detalles_escondidos.tif` para encontrar sus detalles ocultos.
- **c)** Analizar cómo influye el tamaño de la ventana en el resultado.

La solución está en `src/problema1_ecualizacion_local.py`.

### Análisis de la imagen

La imagen tiene cinco recuadros oscuros sobre un fondo claro. Para entender por qué los detalles no se ven, el script muestra su tamaño y cuenta cuántos píxeles tiene cada nivel de gris con `np.unique(img, return_counts=True)`:

```text
Tamaño: (256, 256), tipo: uint8
Nivel 0: 16872 píxeles
Nivel 1: 500 píxeles
Nivel 2: 130 píxeles
Nivel 3: 51 píxeles
Nivel 4: 161 píxeles
Nivel 5: 170 píxeles
Nivel 6: 98 píxeles
Nivel 7: 27 píxeles
Nivel 8: 130 píxeles
Nivel 9: 48 píxeles
Nivel 10: 996 píxeles
Nivel 11: 37 píxeles
Nivel 226: 1175 píxeles
Nivel 227: 43946 píxeles
Nivel 228: 1195 píxeles
```

La imagen mide 256×256 píxeles y usa solo 15 de los 256 niveles posibles. El histograma de la Figura 1 muestra que esos niveles están agrupados en los dos extremos del rango:

- **Niveles 0 a 11:** los recuadros oscuros y lo que hay dentro de ellos.
- **Niveles 226 a 228:** el fondo claro, que vale 227 en casi todos sus píxeles.

Lo que hay dentro de un recuadro se diferencia del negro en 11 niveles como máximo, y por eso no se distingue a simple vista.

![Imagen original y su histograma en escala logarítmica. Los ejes de la imagen están en píxeles.](../resultados/problema1/histograma_original.png)

### Ecualización global

Como referencia, se aplicó `cv2.equalizeHist` a la imagen completa. Como usa un único histograma, aplica la misma transformación en todas las zonas. El script muestra a qué nivel lleva cada nivel original:

```text
Nivel 0 -> 0
Nivel 1 -> 3
Nivel 2 -> 3
Nivel 3 -> 4
Nivel 4 -> 4
Nivel 5 -> 5
Nivel 6 -> 6
Nivel 7 -> 6
Nivel 8 -> 7
Nivel 9 -> 7
Nivel 10 -> 12
Nivel 11 -> 12
Nivel 226 -> 18
Nivel 227 -> 249
Nivel 228 -> 255
```

La Figura 2 muestra la imagen ecualizada y su histograma. La salida explica el resultado:

- **Los recuadros siguen sin mostrar detalles.** El nivel 0 es el mínimo y va a 0. Según la salida de la sección anterior, los niveles 1 a 11 suman 2 348 píxeles, menos del 5 % de los que no valen 0, así que la ecualización les asigna menos del 5 % del rango: quedan entre 3 y 12, todavía casi negros.
- **El fondo se llena de puntos.** El nivel 226 va a 18 y el 227 a 249. Una diferencia de un solo nivel en el fondo se convierte en un salto de 231 niveles, y por eso el fondo queda salpicado de puntos oscuros.

La ecualización global, entonces, no revela los detalles: hace falta una transformación que se adapte a cada zona de la imagen.

![Ecualización global y su histograma en escala logarítmica.](../resultados/problema1/histograma_global.png)

### Implementación de la ecualización local (ítem a)

La función `ecualizacion_local(img, ventana)` recibe la imagen y el tamaño de la ventana como una tupla `(M, N)`. Para cada píxel de la imagen:

1. Toma la ventana de M×N píxeles centrada en él.
2. Ecualiza el histograma de la ventana con `cv2.equalizeHist`.
3. Le asigna al píxel el nuevo nivel del centro de la ventana.

La transformación que aplica `cv2.equalizeHist` en el paso 2 es:

$$
s = \operatorname{round}\left( 255 \cdot \frac{\mathrm{CDF}(g) - \mathrm{CDF}_{\min}}{M \cdot N - \mathrm{CDF}_{\min}} \right)
$$

donde $g$ es el nivel del píxel central y $\mathrm{CDF}_{\min}$ es el primer valor no nulo de la distribución acumulada de la ventana.

**Bordes.** En los píxeles cercanos al borde, la ventana se sale de la imagen. Para resolverlo, antes de recorrerla se le agrega un marco con `cv2.copyMakeBorder` y `cv2.BORDER_REPLICATE`, que repite los píxeles extremos. El marco tiene la mitad de la ventana de cada lado; con una ventana par, un píxel menos abajo y a la derecha. Así, el recorte de M×N que empieza en la posición (fila, columna) de la imagen con borde queda centrado en el píxel (fila, columna) de la original. Para la ventana más grande, el script muestra:

```text
Tamaño con borde para una ventana de 61x61: (316, 316)
```

La Figura 3 compara la imagen original con la imagen con borde: se agregan 30 píxeles por lado.

![Imagen original y la misma imagen con el borde replicado para una ventana de 61×61.](../resultados/problema1/borde_replicado.png)

### Detalles ocultos (ítem b)

Se aplicó la función con ventanas de 5×5, 15×15, 31×31 y 61×61 píxeles. La Figura 4 compara los resultados con la imagen original y con la ecualización global. Con la ecualización local aparecen los cinco detalles ocultos, que se resumen en la tabla siguiente.

| Recuadro | Detalle oculto |
|:-------|:----------|
| Superior izquierdo | Un pequeño cuadrado claro |
| Superior derecho | Un segmento diagonal ascendente |
| Central | La letra minúscula «a» |
| Inferior izquierdo | Cuatro líneas horizontales paralelas |
| Inferior derecho | Un disco |

![Imagen original, ecualización global y ecualizaciones locales con cada tamaño de ventana.](../resultados/problema1/comparacion_ventanas.png)

### Influencia del tamaño de la ventana (ítem c)

La tabla siguiente resume lo que se observa al comparar las ecualizaciones locales de la Figura 4.

| Ventana | Observación |
|:---|:---------------|
| 5×5 | Los detalles aparecen, pero el fondo queda granulado. En el fondo casi todos los píxeles valen 227; en una ventana de solo 25 píxeles, un único 226 o 228 alcanza para producir un salto de nivel muy grande. |
| 15×15 | Las figuras se distinguen con fuerza, pero todavía se amplifican puntos y pequeños bloques del fondo, y aparecen puntos blancos sueltos dentro de los recuadros. |
| 31×31 | Es el mejor compromiso para esta imagen: las cinco figuras se leen bien y el fondo es menos dominante que con 5×5 o 15×15. |
| 61×61 | Las figuras se ven más apagadas y con degradados: el disco, por ejemplo, es más claro en el centro que en el borde. |

**Por qué aparecen degradados con 61×61.** En los ejes de la Figura 1 se ve que cada recuadro mide unos 60 píxeles de lado: la ventana de 61×61 es casi del tamaño de un recuadro. Por eso, en casi cualquier punto de un recuadro, la ventana incluye también parte del fondo claro. Esos píxeles quedan por encima de los detalles en la distribución acumulada, así que el detalle deja de ser el nivel más alto de la ventana y se lleva a un valor más bajo. Cuánto fondo entra depende de la posición: en el centro del recuadro entra menos y el detalle se ve más claro; cerca del borde entra más y se oscurece.

**La ventana de 31×31.** La Figura 5 muestra el resultado con la ventana elegida y su histograma, en el mismo formato que la Figura 2 para poder compararlos. La ecualización global asigna un único nivel de salida a cada nivel original, así que su histograma tiene solo barras aisladas: según la salida de la sección anterior, apenas 10 valores distintos (0, 3, 4, 5, 6, 7, 12, 18, 249 y 255). En la ecualización local, en cambio, la transformación depende de la vecindad de cada píxel: un mismo nivel original puede terminar en valores distintos, y el histograma ocupa prácticamente todo el rango entre 0 y 255. Por eso los detalles, que en la imagen original diferían del negro en pocos niveles, quedan claros sobre sus recuadros.

![Ecualización local con ventana de 31×31 y su histograma en escala logarítmica.](../resultados/problema1/histograma_local_31x31.png)

### Conclusiones

La ecualización global no revela los detalles porque reparte el rango según la cantidad de píxeles de toda la imagen, y los niveles de los detalles tienen muy pocos píxeles. La ecualización local, en cambio, calcula la transformación con la vecindad de cada píxel y estira los pocos niveles de cada recuadro a todo el rango.

El tamaño de la ventana tiene que ser mayor que los detalles, para que la ventana tenga variedad de niveles, y menor que las zonas que los contienen, para no mezclar el recuadro con el fondo. En esta imagen, con recuadros de unos 60 píxeles de lado, la ventana de 31×31 cumple las dos condiciones; las ventanas más chicas amplifican el ruido del fondo y la de 61×61 mezcla cada recuadro con el fondo.
