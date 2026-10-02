# Trabajo Práctico de Procesamiento de Imágenes

## Problema 1: ecualización local de histograma

El TP usa las bibliotecas trabajadas en la materia: **OpenCV (`cv2`)** para
leer y guardar imágenes, replicar bordes y realizar la ecualización global;
**NumPy** para los arreglos y los histogramas locales; y **Matplotlib** para
las comparaciones visuales y el informe.

El script `Ecualizacion de histograma.py` implementa la ecualización local para imágenes en escala de grises de 8 bits. Recorre la imagen con una ventana `M x N`, calcula el histograma de cada vecindad y transforma el nivel del píxel central. En los bordes replica los valores de los píxeles extremos, equivalente a `BORDER_REPLICATE` de OpenCV.

La función principal es `ecualizacion_local(imagen, tamano_ventana)`, donde `imagen` es un arreglo NumPy 2D y `tamano_ventana` es una tupla `(M, N)`. Admite dimensiones positivas pares o impares; para ventanas pares toma como centro el elemento de índice `M // 2, N // 2`.

## Entorno y dependencias

Versiones utilizadas para generar los resultados:

- Python 3.13
- NumPy 2.2.6
- Matplotlib 3.10.7
- OpenCV Contrib 4.12.0.88

Crear y activar el entorno virtual desde Git Bash en Windows:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
```

En Linux o macOS, usar `source .venv/bin/activate`.

## Ejecución

Desde la carpeta del repositorio, ejecutar:

```bash
python "Ecualizacion de histograma.py"
```

Por defecto, procesa `Img/Imagen_con_detalles_escondidos.tif` con ventanas de `5x5`, `15x15`, `31x31` y `61x61`. Los PNG se guardan en `Resultados/Problema1/imagenes/` para incorporarlos al informe.

Se puede indicar otra imagen, carpeta de salida y conjunto de ventanas:

```bash
python "Ecualizacion de histograma.py" --imagen Img/Imagen_con_detalles_escondidos.tif --salida Resultados/Prueba --ventanas 9x9 21x21 41x41
```

El tamaño de ventana se escribe como `M x N` sin espacios (por ejemplo, `15x21`). La imagen de entrada debe ser monocromática o convertible a escala de grises y estar representada en 8 bits.

El análisis detallado del informe corresponde a la imagen incluida y a las cuatro ventanas predeterminadas. Si se ejecuta con otra imagen o tamaños personalizados, se deben actualizar las imágenes incluidas en el informe Markdown.

## Entregables

- `Ecualizacion de histograma.py`: función y programa de análisis.
- `Resultados/Problema1/comparacion_visual.md`: informe en Markdown con la comparación visual y el análisis.
- `requirements.txt`: versiones fijadas de las dependencias.
- `Resultados/Problema1/imagenes/`: imágenes procesadas para incorporar al informe.
## Problema 2: validación de planillas

El script `Validacion de planillas.py` detecta automáticamente la grilla,
valida los seis campos de cada registro y procesa en ciclo las cuatro imágenes
`grade_sheet_1.png` a `grade_sheet_4.png`.

```bash
python "Validacion de planillas.py"
```

Para cada planilla se guardan en `Resultados/Problema2/` un CSV con los
resultados `OK`/`MAL` y una imagen que muestra los crops de nombre de los
registros válidos con condición final `R` (recupera) o `L` (libre).

Se puede procesar una planilla concreta o elegir otra carpeta de salida:

```bash
python "Validacion de planillas.py" --imagenes Img/grade_sheet_1.png --salida Resultados/Prueba2
```
