# Informe — Trabajo Práctico N° 1

Procesamiento de Imágenes I (IA 4.4) · Tecnicatura Universitaria en Inteligencia Artificial · FCEIA, UNR · 2026, 2° semestre

**Integrantes:** Sharo Giuntoli, Josías Calabozo, Ismael Darruiz, Sebastián Di Carlo

## Problema 1: Ecualización local de histograma

### Imagen de entrada

Se procesó la imagen `datos/Imagen_con_detalles_escondidos.tif`: una imagen TIFF monocromática de 256×256 píxeles y 8 bits, con cinco recuadros oscuros sobre un fondo claro.

### Método

Para cada posición de la imagen se obtiene el histograma local de una ventana de M×N píxeles centrada en el píxel, y con él se calcula el nuevo nivel del píxel central:

$$
s = \operatorname{round}\left( 255 \cdot \frac{\mathrm{CDF}(g) - \mathrm{CDF}_{\min}}{M \cdot N - \mathrm{CDF}_{\min}} \right)
$$

donde $g$ es el nivel del píxel central y $\mathrm{CDF}_{\min}$ es el primer valor no nulo de la distribución acumulada de la ventana. En ventanas constantes se conserva el nivel original. En los bordes se replican los píxeles extremos (`cv2.BORDER_REPLICATE`).

Como referencia se calculó también la ecualización global (`cv2.equalizeHist`), que usa un único histograma para toda la imagen y, por lo tanto, aplica la misma transformación en todas las zonas.

La implementación es la función `ecualizacion_local` de `src/problema1_ecualizacion_local.py`.

### Resultados

Se evaluaron ventanas de 5×5, 15×15, 31×31 y 61×61 píxeles.

![Imagen original, ecualización global y ecualizaciones locales con ventanas de 5×5, 15×15, 31×31 y 61×61.](../resultados/problema1/comparacion_ventanas.png)

*Figura 1. Imagen original, ecualización global y ecualizaciones locales. Se genera con `python src/problema1_ecualizacion_local.py`.*

### Detalles ocultos

La ecualización local permite distinguir cinco figuras que apenas contrastan con sus fondos en la imagen original:

| Recuadro | Detalle oculto |
|---|---|
| Superior izquierdo | Un pequeño cuadrado claro |
| Superior derecho | Un segmento diagonal ascendente |
| Central | La letra minúscula «a» |
| Inferior izquierdo | Cuatro líneas horizontales paralelas |
| Inferior derecho | Un disco circular |

### Comparación entre ecualización global y local

La ecualización global aumenta el contraste del fondo y hace más visible su textura, pero los detalles débiles continúan con poco contraste respecto de sus recuadros. La transformación local adapta el mapeo a cada zona y recupera las cinco figuras con claridad.

### Influencia del tamaño de la ventana

| Ventana | Observación |
|---|---|
| 5×5 | La ventana pequeña responde a variaciones muy localizadas. Los detalles aparecen, pero también se amplifica intensamente el ruido del fondo, que queda con aspecto granulado. |
| 15×15 | Las figuras se distinguen con fuerza. Aún se amplifican puntos y pequeños bloques del fondo, por lo que la imagen resulta visualmente fragmentada. |
| 31×31 | Ofrece el compromiso más favorable en esta imagen: las cinco figuras se leen bien y el fondo resulta menos dominante que con 5×5 o 15×15. |
| 61×61 | La vecindad más grande reduce la respuesta a diferencias pequeñas dentro de los recuadros: la letra, las líneas y la diagonal se ven más apagadas, y las figuras pierden nitidez. |

### Conclusión

El tamaño adecuado depende de la escala espacial de los detalles y del ruido. Para esta imagen, 31×31 brinda la mejor lectura general entre los tamaños comparados; ventanas menores aumentan el ruido y ventanas mayores suavizan el contraste local. La elección se basa en la inspección visual de los resultados y puede variar para otras imágenes.
