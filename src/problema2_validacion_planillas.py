"""Problema 2: validación automática de planillas de calificaciones."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Rutas relativas a la raíz del repositorio, desde donde se ejecuta el script.
CARPETA_DATOS = Path("datos")
CARPETA_SALIDA = Path("resultados") / "problema2"
CAMPOS = [
    "Legajo",
    "Nombre y Apellido",
    "Parcial 1",
    "Parcial 2",
    "Parcial 3",
    "Condición Final",
]
# Una fila o columna es una línea de la tabla si más de la mitad de sus
# píxeles son oscuros. En las cuatro planillas las líneas ocupan al menos el
# 63 % del ancho o del alto, y las filas o columnas con texto, como mucho el 41 %.
PROPORCION_LINEA = 0.5


def centros_de_lineas(proyeccion: np.ndarray, umbral: float) -> list[int]:
    """Devuelve el centro de cada tramo de la proyección que supera el umbral.

    Las líneas de la tabla tienen más de un píxel de ancho, así que cada línea
    aparece como un tramo de posiciones consecutivas por encima del umbral.
    """
    centros = []
    inicio = None
    for posicion, es_linea in enumerate(proyeccion > umbral):
        if es_linea and inicio is None:
            inicio = posicion
        elif not es_linea and inicio is not None:
            centros.append(round((inicio + posicion - 1) / 2))
            inicio = None
    return centros


def detectar_grilla(img_th: np.ndarray) -> tuple[list[int], list[int]]:
    """Devuelve las posiciones de las líneas horizontales y verticales de la tabla."""
    # Las líneas cruzan casi toda la tabla, así que tienen muchos más píxeles
    # oscuros que cualquier fila o columna con texto.
    img_rows = np.sum(img_th, 1)
    img_cols = np.sum(img_th, 0)
    filas = centros_de_lineas(img_rows, PROPORCION_LINEA * img_th.shape[1])
    columnas = centros_de_lineas(img_cols, PROPORCION_LINEA * img_th.shape[0])
    if len(columnas) != 8:
        raise ValueError("No se pudo detectar la grilla de la planilla.")
    return filas, columnas


def recortar_celda(img, filas, columnas, registro: int, campo: str) -> np.ndarray:
    """Devuelve la celda de un registro y un campo, sin las líneas de la tabla."""
    # El registro 1 está entre la segunda y la tercera línea horizontal (la
    # primera franja es el encabezado). Los campos empiezan en la segunda
    # columna, porque la primera es "Nro.".
    i = CAMPOS.index(campo) + 1
    # Las líneas miden 1 píxel; el margen de 2 píxeles las deja afuera junto con
    # los grises de sus bordes.
    return img[filas[registro] + 2 : filas[registro + 1] - 2, columnas[i] + 2 : columnas[i + 1] - 2]


def analizar_campo(celda_th: np.ndarray) -> tuple[int, int]:
    """Devuelve la cantidad de caracteres y de espacios entre palabras de una celda."""
    _, _, stats, _ = cv2.connectedComponentsWithStats(celda_th.astype(np.uint8), 8)
    stats = stats[1:]  # La componente 0 es el fondo.
    # Se descartan las componentes de 1 píxel; el punto de "1.0" mide 2.
    stats = stats[stats[:, -1] > 1]
    stats = stats[np.argsort(stats[:, 0])]  # De izquierda a derecha.

    caracteres = []  # Inicio y fin horizontal de cada carácter.
    for x, _, ancho, _, _ in stats:
        # Una componente que se superpone en horizontal con la anterior es
        # parte del mismo carácter, como la tilde de la Ñ.
        if caracteres and x < caracteres[-1][1]:
            caracteres[-1][1] = max(caracteres[-1][1], x + ancho)
        else:
            caracteres.append([x, x + ancho])

    # En las cuatro planillas, entre letras de una misma palabra hay como mucho
    # 4 píxeles y entre palabras, al menos 10. Una separación mayor a 7 es un
    # espacio.
    espacios = 0
    for anterior, siguiente in zip(caracteres, caracteres[1:]):
        if siguiente[0] - anterior[1] > 7:
            espacios += 1
    return len(caracteres), espacios


def validar(campo: str, caracteres: int, espacios: int) -> bool:
    """Indica si un campo cumple la restricción de la consigna."""
    if campo == "Legajo":
        return caracteres == 8 and espacios == 0
    if campo == "Nombre y Apellido":
        # Los espacios no cuentan como caracteres: no están entre los permitidos.
        return espacios >= 1 and caracteres <= 12
    if campo == "Condición Final":
        return caracteres == 1
    # Parcial 1, 2 y 3.
    return 1 <= caracteres <= 2 and espacios == 0


def leer_condicion(celda_th: np.ndarray) -> str:
    """Devuelve "A", "L" o "R" según la forma de la letra de la condición final."""
    _, _, stats, _ = cv2.connectedComponentsWithStats(celda_th.astype(np.uint8), 8)
    # La componente más grande, sin contar el fondo, es la letra.
    x, y, ancho, alto, _ = stats[1:][np.argmax(stats[1:, -1])]
    letra = celda_th[y : y + alto, x : x + ancho]
    # L y R tienen un trazo vertical a la izquierda de toda su altura; en la A,
    # la primera columna tiene tinta en menos de la mitad de las filas.
    if letra[:, 0].mean() < 0.5:
        return "A"
    # L no tiene tinta en su mitad superior derecha; R sí.
    if not letra[: alto // 2, ancho // 2 :].any():
        return "L"
    return "R"


# --- Detalle del procesamiento de grade_sheet_1 --------------------------------
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
img = cv2.imread(str(CARPETA_DATOS / "grade_sheet_1.png"), cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("No se pudo leer la imagen: grade_sheet_1.png")

# Binarización: el umbral se elige automáticamente con el método de Otsu.
th, _ = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
img_th = img < th
print(f"Umbral de Otsu para grade_sheet_1: {th:.0f}")

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.imshow(img, cmap="gray", vmin=0, vmax=255)
plt.title("Planilla original")
plt.subplot(122)
plt.imshow(~img_th, cmap="gray")
plt.title(f"Binarizada (img < {th:.0f})")
plt.savefig(CARPETA_SALIDA / "binarizacion.png", dpi=150, bbox_inches="tight")
plt.show()

# Proyecciones: cantidad de píxeles oscuros por fila y por columna.
img_rows = np.sum(img_th, 1)
img_cols = np.sum(img_th, 0)
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(img_rows)
plt.axhline(PROPORCION_LINEA * img_th.shape[1], color="red", linestyle="--")
plt.title("Píxeles oscuros por fila")
plt.xlabel("Fila")
plt.subplot(122)
plt.plot(img_cols)
plt.axhline(PROPORCION_LINEA * img_th.shape[0], color="red", linestyle="--")
plt.title("Píxeles oscuros por columna")
plt.xlabel("Columna")
plt.savefig(CARPETA_SALIDA / "proyecciones.png", dpi=150, bbox_inches="tight")
plt.show()

# Grilla detectada, dibujada sobre la planilla.
filas, columnas = detectar_grilla(img_th)
print(f"Líneas detectadas: {len(filas)} horizontales y {len(columnas)} verticales")
img_grilla = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
for fila in filas:
    cv2.line(img_grilla, (columnas[0], fila), (columnas[-1], fila), (255, 0, 0), 2)
for columna in columnas:
    cv2.line(img_grilla, (columna, filas[0]), (columna, filas[-1]), (0, 0, 255), 2)
plt.figure(figsize=(8, 7))
plt.imshow(img_grilla)
plt.title("Grilla detectada")
plt.savefig(CARPETA_SALIDA / "grilla.png", dpi=150, bbox_inches="tight")
plt.show()

# Componentes conectadas de una celda, con el umbral fijo de 160 que se usaba
# antes y con el de Otsu. Con 160 entran los grises del borde de las letras y
# algunas letras vecinas quedan unidas en una sola componente.
plt.figure(figsize=(8, 3))
for i, umbral in enumerate([160, th], start=1):
    celda_th = recortar_celda(img, filas, columnas, 8, "Nombre y Apellido") < umbral
    caracteres, espacios = analizar_campo(celda_th)
    img_celda = cv2.cvtColor(np.uint8(~celda_th) * 255, cv2.COLOR_GRAY2RGB)
    _, _, stats, _ = cv2.connectedComponentsWithStats(celda_th.astype(np.uint8), 8)
    for x, y, ancho, alto, _ in stats[1:]:
        cv2.rectangle(img_celda, (x, y), (x + ancho - 1, y + alto - 1), (255, 0, 0), 1)
    plt.subplot(2, 1, i)
    plt.imshow(img_celda, interpolation="nearest")
    plt.title(f"Umbral {umbral:.0f}: {caracteres} caracteres y {espacios} espacio")
    plt.axis("off")
plt.tight_layout()
plt.savefig(CARPETA_SALIDA / "componentes.png", dpi=150, bbox_inches="tight")
plt.show()

# Letras de la condición final de los registros 1, 2 y 4, recortadas a su
# tamaño, y cómo las lee leer_condicion.
plt.figure(figsize=(6, 3))
for i, registro in enumerate([1, 2, 4], start=1):
    celda_th = recortar_celda(img_th, filas, columnas, registro, "Condición Final")
    # Posiciones de los píxeles con tinta, para recortar la letra.
    filas_tinta, columnas_tinta = np.nonzero(celda_th)
    letra = celda_th[filas_tinta.min() : filas_tinta.max() + 1]
    letra = letra[:, columnas_tinta.min() : columnas_tinta.max() + 1]
    plt.subplot(1, 3, i)
    plt.imshow(~letra, cmap="gray", interpolation="nearest")
    plt.title(f"Registro {registro}: {leer_condicion(celda_th)}")
    plt.axis("off")
plt.savefig(CARPETA_SALIDA / "condicion.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Procesamiento de las planillas --------------------------------------------
for ruta in sorted(CARPETA_DATOS.glob("grade_sheet_[0-9].png")):
    img = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {ruta}")
    print(f"=== Planilla {ruta.name} ===")

    # --- Binarización y detección de la grilla ---
    # El umbral se elige automáticamente con el método de Otsu; los píxeles
    # más oscuros que el umbral son tinta.
    th, _ = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    img_th = img < th
    print(f"Umbral de Otsu: {th:.0f}")
    filas, columnas = detectar_grilla(img_th)

    # --- Validación de cada registro ---
    resultados = []
    no_aprobados = []  # Crop del nombre y condición de cada alumno no aprobado.
    # La primera franja de la tabla es el encabezado; cada una de las
    # siguientes es un registro.
    for registro in range(1, len(filas) - 1):
        print(f"Registro {registro}:")
        validacion = []
        for campo in CAMPOS:
            celda_th = recortar_celda(img_th, filas, columnas, registro, campo)
            caracteres, espacios = analizar_campo(celda_th)
            estado = "OK" if validar(campo, caracteres, espacios) else "MAL"
            validacion.append(estado)
            print(f"{campo}: {estado}")
        print()
        resultados.append(validacion)

        # Solo se informan los registros con todos los campos correctos. La
        # última celda recorrida es la de la condición final.
        if all(estado == "OK" for estado in validacion):
            condicion = leer_condicion(celda_th)
            if condicion in ("L", "R"):
                img_nombre = recortar_celda(img, filas, columnas, registro, "Nombre y Apellido")
                no_aprobados.append((img_nombre, condicion))

    # --- CSV con los resultados ---
    # Una fila por registro; el ID es su número de orden en la planilla.
    tabla = pd.DataFrame(resultados, columns=CAMPOS, index=range(1, len(resultados) + 1))
    tabla.to_csv(CARPETA_SALIDA / f"validacion_{ruta.stem}.csv", index_label="ID")

    # --- Imagen de alumnos no aprobados ---
    # Un panel por alumno, con el crop de su nombre y la condición como título.
    plt.figure(figsize=(5, 0.8 * max(1, len(no_aprobados)) + 0.6))
    for i, (img_nombre, condicion) in enumerate(no_aprobados, start=1):
        plt.subplot(len(no_aprobados), 1, i)
        plt.imshow(img_nombre, cmap="gray", vmin=0, vmax=255)
        if condicion == "R":
            plt.title("RECUPERA (R)", color="darkorange")
        else:
            plt.title("LIBRE (L)", color="red")
        plt.axis("off")
    if not no_aprobados:
        plt.text(0.5, 0.5, "No hay alumnos no aprobados con registros válidos.", ha="center")
        plt.axis("off")
    plt.suptitle(f"No aprobados - {ruta.stem}")
    plt.tight_layout()
    plt.savefig(CARPETA_SALIDA / f"no_aprobados_{ruta.stem}.png", dpi=150, bbox_inches="tight")
    plt.show()

print(f"Resultados guardados en: {CARPETA_SALIDA}")
