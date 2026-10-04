# Trabajo Práctico N° 1 — Procesamiento de Imágenes I

Tecnicatura Universitaria en Inteligencia Artificial · FCEIA · Universidad Nacional de Rosario
Procesamiento de Imágenes I (IA 4.4) · 2026, 2° semestre

**Integrantes:** Josías Calabozo, Ismael Darruiz, Sebastián Di Carlo, Sharo Giuntoli

El trabajo resuelve los dos problemas planteados en la [consigna](docs/consigna_TP1_2026_C2.pdf):

1. **Ecualización local de histograma:** implementación propia y análisis de una imagen con detalles ocultos.
2. **Validación de planillas de calificaciones:** verificación automática de los campos de cada registro a partir de la imagen de la planilla.

## Estructura del repositorio

```text
.
├── src/                                    Código fuente
│   ├── problema1_ecualizacion_local.py
│   └── problema2_validacion_planillas.py
├── datos/                                  Imágenes de entrada provistas por la cátedra
│   ├── Imagen_con_detalles_escondidos.tif
│   ├── grade_sheet_1.png … grade_sheet_4.png
│   └── grade_sheet_empty.png               Planilla vacía (referencia)
├── resultados/                             Salidas generadas por los scripts
│   ├── problema1/                          Imágenes ecualizadas y figuras del informe
│   └── problema2/                          CSV de validación e imagen de no aprobados por planilla
├── docs/                                   Consigna e informe
│   ├── consigna_TP1_2026_C2.pdf
│   ├── informe.md                          Fuente editable del informe
│   └── informe.pdf                         Informe
├── requirements.txt
└── README.md
```

## Requisitos

- Python 3.14
- numpy 2.5.2
- opencv-contrib-python 5.0.0.93
- matplotlib 3.11.1
- pandas 3.0.6

Las mismas versiones están fijadas en [`requirements.txt`](requirements.txt), que se usa en la instalación.

### Instalación

Desde la raíz del repositorio, en Git Bash (Windows):

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
```

En PowerShell, el entorno se activa con `.venv\Scripts\Activate.ps1`; en Linux o macOS, con `source .venv/bin/activate`.

## Ejecución

Los scripts no reciben argumentos: leen de `datos/` y escriben en `resultados/`. Los comandos se ejecutan desde la raíz del repositorio: los scripts usan rutas relativas a esa carpeta, para poder ejecutarlos también línea por línea.

Cada figura se guarda y después se muestra en una ventana de Matplotlib, y el script se detiene hasta que se cierra: el Problema 1 abre 5 ventanas y el Problema 2, 9.

Para generar los resultados sin abrir ventanas, se usa la variable de entorno `MPLBACKEND=Agg`. Las figuras se guardan igual, y Matplotlib avisa que no puede mostrarlas. La forma de definirla depende de la terminal:

- En Git Bash, la variable se escribe antes del comando y vale solo para esa ejecución:

  ```bash
  MPLBACKEND=Agg python src/problema1_ecualizacion_local.py
  ```

- En PowerShell, la variable se define en un comando aparte y queda activa en toda la sesión de la terminal:

  ```powershell
  $env:MPLBACKEND = "Agg"
  python src/problema1_ecualizacion_local.py
  ```

  Para volver a ver las ventanas en esa misma terminal, se borra la variable con `Remove-Item Env:MPLBACKEND`.

### Problema 1

```bash
python src/problema1_ecualizacion_local.py
```

Procesa `datos/Imagen_con_detalles_escondidos.tif` con ventanas de 5×5, 15×15, 31×31 y 61×61. Muestra por pantalla el tamaño de la imagen, cuántos píxeles tiene cada nivel de gris, a qué nivel lleva cada uno la ecualización global y el tamaño de la imagen con el borde replicado. Guarda en `resultados/problema1/`:

- `imagen_original.png`
- `ecualizacion_global.png`: ecualización global con `cv2.equalizeHist`, como referencia.
- `ecualizacion_local_<M>x<N>.png`: una imagen por cada ventana.
- `histograma_original.png`: imagen original y su histograma.
- `histograma_global.png`: ecualización global y su histograma.
- `borde_replicado.png`: imagen original y la misma imagen con el borde que agrega `cv2.copyMakeBorder` para la ventana más grande. Sobre la imagen con borde se marcan, con `cv2.rectangle`, los límites de la original y la ventana de su primer píxel.
- `comparacion_ventanas.png`: figura con la imagen original, la ecualización global y todas las ecualizaciones locales, generada con Matplotlib.
- `histograma_local_31x31.png`: ecualización local con la ventana de 31×31 y su histograma.

Para probar otros tamaños de ventana, se modifica la lista `VENTANAS` al comienzo del script.

### Problema 2

```bash
python src/problema2_validacion_planillas.py
```

Primero muestra el detalle de cada paso con `datos/grade_sheet_1.png` (y el caso del «100» con `datos/grade_sheet_4.png`). Como ejemplo, imprime el inicio y el fin de las líneas detectadas, las componentes de la celda Legajo del registro 1, los caracteres y espacios que cuenta en esa celda y en la del nombre, la validación campo por campo del registro 19 de `datos/grade_sheet_2.png` y las dos medidas con las que lee las letras de la condición final. Después procesa en ciclo las planillas `datos/grade_sheet_1.png` a `datos/grade_sheet_4.png`. Para cada una muestra por pantalla el umbral de Otsu, el resultado (`OK` o `MAL`) de cada campo de cada registro y un resumen con la cantidad de registros correctos, de libres y de alumnos que recuperan. Guarda en `resultados/problema2/`:

- `validacion_grade_sheet_<id>.csv`: resultado de la validación de cada registro.
- `no_aprobados_grade_sheet_<id>.png`: alumnos no aprobados con registro válido.
- `binarizacion.png`, `proyecciones.png`, `grilla.png` y `condicion.png`: detalle de cada paso con la primera planilla.
- `componentes.png`: componentes conectadas del «100» de la cuarta planilla, con un umbral fijo de 160 y con el de Otsu.

## Descripción de la solución

### Problema 1: ecualización local de histograma

La función `ecualizacion_local(img, ventana)` recibe una imagen en escala de grises (`uint8`) y una tupla `(M, N)`. Para cada píxel toma la ventana de M×N centrada en él, la ecualiza con `cv2.equalizeHist` y conserva el nuevo valor del píxel central. La transformación que aplica `cv2.equalizeHist` sobre la ventana es:

```text
s = round( 255 · (CDF(r) − CDF_min) / (M·N − CDF_min) )
```

donde `r` es el nivel del píxel central y `CDF_min` es el primer valor no nulo de la distribución acumulada de la ventana. Si la ventana tiene un único nivel de gris, se conserva el valor original.

- **Bordes:** se replican los píxeles extremos con `cv2.copyMakeBorder` y `cv2.BORDER_REPLICATE`.
- **Ventanas pares:** se admiten; el centro es el elemento de índice `(M // 2, N // 2)`.

El análisis de los detalles ocultos y de la influencia del tamaño de la ventana está en el [informe](docs/informe.pdf).

### Problema 2: validación de planillas

El algoritmo está en la función `validar_planilla(img)`, que recibe únicamente la imagen de una planilla en escala de grises, muestra por pantalla el resultado de cada campo de cada registro y devuelve esos resultados y los recortes del nombre de los alumnos libres y de los que recuperan. El ciclo final la aplica a las cuatro planillas y genera el CSV (`guardar_csv`) y la imagen de no aprobados (`guardar_no_aprobados`) de cada una.

1. **Detección de la grilla.** Como propone la consigna, se umbraliza la imagen con `img_th = img < th` (con `th` elegido por el método de Otsu en cada planilla), se suman los píxeles oscuros por fila y por columna (`img_rows` e `img_cols`) y se marcan las líneas con `img_rows_th = img_rows > th_row` e `img_cols_th = img_cols > th_col`, donde `th_row` y `th_col` son la mitad del ancho y del alto. El inicio y el fin de cada línea se encuentran con `np.diff` y `np.argwhere`, como en el ejemplo de los renglones de la Unidad 1, así que las líneas pueden tener cualquier grosor; después se verifica que haya 22 horizontales y 8 verticales.
2. **Recorte de celdas.** Cada campo se recorta desde el píxel siguiente al fin de una línea hasta el anterior al inicio de la siguiente.
3. **Caracteres y palabras.** En cada celda se obtienen las componentes conectadas (conectividad 8) y se descartan las de 1 px con el filtro por área de la consigna (`ix_area = stats[:, -1] > th_area`). Las componentes que se superponen en horizontal se cuentan como un solo carácter, como la Ñ y su tilde. Una separación horizontal entre caracteres consecutivos mayor a 0,6 veces la altura del carácter más alto de la celda se cuenta como espacio entre palabras.
4. **Validación.** Cada campo se evalúa con estos criterios; una celda vacía no cumple ninguno y se marca `MAL`.

   | Campo             | Restricción de la consigna                      | Criterio implementado                                                                                                    |
   | ----------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
   | Legajo            | 8 caracteres, una única palabra                 | 8 caracteres y ningún espacio                                                                                            |
   | Nombre y Apellido | Al menos dos palabras y no más de 12 caracteres | Al menos un espacio y no más de 12 caracteres (los espacios no cuentan, porque no están entre los caracteres permitidos) |
   | Parcial 1, 2 y 3  | 1 o 2 caracteres consecutivos                   | 1 o 2 caracteres y ningún espacio                                                                                        |
   | Condición Final   | Un único carácter                               | Exactamente 1 carácter                                                                                                   |

5. **Salidas.**
   - **CSV:** columnas `ID`, `Legajo`, `Nombre y Apellido`, `Parcial 1`, `Parcial 2`, `Parcial 3` y `Condición Final`. El `ID` corresponde al orden del registro en la planilla y cada celda vale `OK` o `MAL`. Codificación UTF-8.
   - **Imagen de no aprobados:** incluye los registros con todos los campos `OK` y condición final `L` o `R`. Muestra el recorte del campo Nombre y Apellido de cada alumno en dos columnas, `LIBRE (L)` y `RECUPERA (R)`, con la cantidad de alumnos en el título de cada una; la letra de la condición se identifica por su forma: L y R tienen un trazo vertical a la izquierda de toda su altura y A no, y entre L y R solo la R tiene tinta en su mitad superior derecha.

## Documentación

- [`docs/consigna_TP1_2026_C2.pdf`](docs/consigna_TP1_2026_C2.pdf): enunciado del trabajo práctico.
- [`docs/informe.pdf`](docs/informe.pdf): informe con el análisis de los resultados.
- [`docs/informe.md`](docs/informe.md): fuente editable del informe, a partir de la cual se genera el PDF.
