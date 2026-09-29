# Trabajo Práctico de Procesamiento de Imágenes

## Problema 1: ecualización local de histograma

El script `Ecualizacion de histograma.py` implementa la ecualización local para imágenes en escala de grises de 8 bits. Recorre la imagen con una ventana `M x N`, calcula el histograma de cada vecindad y transforma el nivel del píxel central. En los bordes replica los valores de los píxeles extremos, equivalente a `BORDER_REPLICATE` de OpenCV.

La función principal es `ecualizacion_local(imagen, tamano_ventana)`, donde `imagen` es un arreglo NumPy 2D y `tamano_ventana` es una tupla `(M, N)`. Admite dimensiones positivas pares o impares; para ventanas pares toma como centro el elemento de índice `M // 2, N // 2`.

## Entorno y dependencias

Versiones utilizadas para generar los resultados:

- Python 3.13.5
- NumPy 2.3.2
- Pillow 11.3.0
- Matplotlib 3.11.2

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

Por defecto, procesa `Img/Imagen_con_detalles_escondidos.tif` con ventanas de `5x5`, `15x15`, `31x31` y `61x61`. Los PNG se guardan en `Resultados/Problema1/` junto con una comparación visual.

Se puede indicar otra imagen, carpeta de salida y conjunto de ventanas:

```bash
python "Ecualizacion de histograma.py" --imagen Img/Imagen_con_detalles_escondidos.tif --salida Resultados/Prueba --ventanas 9x9 21x21 41x41
```

El tamaño de ventana se escribe como `M x N` sin espacios (por ejemplo, `15x21`). La imagen de entrada debe ser monocromática o convertible a escala de grises y estar representada en 8 bits.

El análisis detallado del informe corresponde a la imagen incluida y a las cuatro ventanas predeterminadas. Si se ejecuta con otra imagen o tamaños personalizados, el PDF muestra la comparación y la configuración utilizada, sin presentar esas conclusiones como si fueran del caso de referencia.

## Entregables

- `Ecualizacion de histograma.py`: función y programa de análisis.
- `requirements.txt`: versiones fijadas de las dependencias.
- `Resultados/Problema1/`: imágenes procesadas y comparación generadas al ejecutar el script.
- `Resultados/Problema1/Informe_Problema1.pdf`: informe del ejercicio con metodología, detalles recuperados y análisis del tamaño de ventana. Se genera junto con las imágenes.
