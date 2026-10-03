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
    filas = centros_de_lineas(img_rows, 0.55 * img_th.shape[1])
    columnas = centros_de_lineas(img_cols, 0.55 * img_th.shape[0])
    if len(columnas) != 8:
        raise ValueError("No se pudo detectar la grilla de la planilla.")
    return filas, columnas


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

    # Entre letras de una misma palabra hay hasta 3 píxeles; una separación
    # mayor a 5 es un espacio.
    espacios = 0
    for anterior, siguiente in zip(caracteres, caracteres[1:]):
        if siguiente[0] - anterior[1] > 5:
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
    # L y R tienen un trazo vertical a la izquierda de toda su altura; A no.
    if letra[:, 0].mean() < 0.8:
        return "A"
    # L no tiene tinta en su mitad superior derecha; R sí.
    if not letra[: alto // 2, ancho // 2 :].any():
        return "L"
    return "R"


# --- Procesamiento de las planillas --------------------------------------------
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

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
    filas, columnas = detectar_grilla(img_th)

    # --- Validación de cada registro ---
    resultados = []
    no_aprobados = []  # Crop del nombre y condición de cada alumno no aprobado.
    # La primera franja de la tabla es el encabezado; cada una de las
    # siguientes es un registro.
    for registro, (arriba, abajo) in enumerate(zip(filas[1:], filas[2:]), start=1):
        print(f"Registro {registro}:")
        validacion = []
        for i, campo in enumerate(CAMPOS):
            # Los campos empiezan en la segunda columna (la primera es "Nro.").
            # El margen de 2 píxeles deja afuera las líneas de la tabla.
            izquierda = columnas[i + 1] + 2
            derecha = columnas[i + 2] - 2
            celda_th = img_th[arriba + 2 : abajo - 2, izquierda:derecha]
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
                img_nombre = img[arriba + 2 : abajo - 2, columnas[2] + 2 : columnas[3] - 2]
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
