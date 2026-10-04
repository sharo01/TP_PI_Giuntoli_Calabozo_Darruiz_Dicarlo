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


def imshow(
    img: np.ndarray,
    new_fig: bool = False,
    title: str | None = None,
    color_img: bool = False,
    blocking: bool = False,
    colorbar: bool = False,
    ticks: bool = False,
) -> None:
    """Muestra una imagen con matplotlib.

    Es la función imshow de los ejemplos de la Unidad 3, con dos cambios. Por
    defecto dibuja en la figura actual y sin barra de color, porque las figuras
    del TP tienen varios paneles y se guardan antes de mostrarse. Y las imágenes
    de 8 bits se muestran con el rango completo, de 0 a 255, para que matplotlib
    no reescale el contraste.
    """
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    elif img.dtype == np.uint8:
        plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    else:
        plt.imshow(img, cmap="gray")
    plt.title(title)
    if not ticks:
        plt.xticks([])
        plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)


def detectar_grilla(img_th: np.ndarray) -> tuple[list[int], list[int]]:
    """Devuelve las posiciones de las líneas horizontales y verticales de la tabla."""
    img_rows = np.sum(img_th, 1)  # Píxeles oscuros de cada fila.
    img_cols = np.sum(img_th, 0)  # Píxeles oscuros de cada columna.
    img_rows_th = img_rows > PROPORCION_LINEA * img_th.shape[1]
    img_cols_th = img_cols > PROPORCION_LINEA * img_th.shape[0]
    filas = [i for i in range(len(img_rows_th)) if img_rows_th[i]]
    columnas = [i for i in range(len(img_cols_th)) if img_cols_th[i]]
    # Cada línea mide 1 píxel, así que cada posición encontrada es una línea.
    # La tabla tiene 22 líneas horizontales (encabezado y 20 registros) y 8
    # verticales; si una línea fuera más gruesa, se contaría más de una vez.
    if len(filas) != 22 or len(columnas) != 8:
        raise ValueError("No se pudo detectar la grilla de la planilla.")
    return filas, columnas


def cargar_planilla(nombre: str):
    """Lee una planilla, la binariza y detecta su grilla.

    Devuelve la imagen, el umbral, la imagen binaria (True donde hay tinta) y las
    posiciones de las líneas horizontales y verticales de la tabla.
    """
    img = cv2.imread(str(CARPETA_DATOS / f"{nombre}.png"), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {nombre}.png")
    # El umbral se elige automáticamente con el método de Otsu.
    th, _ = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    img_th = img < th
    filas, columnas = detectar_grilla(img_th)
    return img, th, img_th, filas, columnas


def recortar_celda(img, filas, columnas, registro: int, campo: str) -> np.ndarray:
    """Devuelve la celda de un registro y un campo, sin las líneas de la tabla."""
    # El registro 1 está entre la segunda y la tercera línea horizontal (la
    # primera franja es el encabezado). Los campos empiezan en la segunda
    # columna, porque la primera es "Nro.".
    i = CAMPOS.index(campo) + 1
    # El margen de 2 píxeles deja afuera las líneas y los grises de sus bordes.
    return img[filas[registro] + 2 : filas[registro + 1] - 2, columnas[i] + 2 : columnas[i + 1] - 2]


def analizar_campo(celda_th: np.ndarray) -> tuple[int, int]:
    """Devuelve la cantidad de caracteres y de espacios entre palabras de una celda."""
    _, _, stats, _ = cv2.connectedComponentsWithStats(celda_th.astype(np.uint8), 8)
    stats = stats[1:]  # La componente 0 es el fondo.
    # Se descartan las componentes de 1 píxel; el punto de "1.0" mide 2.
    stats = stats[stats[:, -1] > 1]
    # Inicio y fin horizontal de cada componente, de izquierda a derecha.
    tramos = sorted([x, x + ancho] for x, _, ancho, _, _ in stats)

    caracteres = []  # Inicio y fin horizontal de cada carácter.
    for inicio, fin in tramos:
        # Una componente que se superpone en horizontal con la anterior es
        # parte del mismo carácter, como la tilde de la Ñ.
        if caracteres and inicio < caracteres[-1][1]:
            caracteres[-1][1] = max(caracteres[-1][1], fin)
        else:
            caracteres.append([inicio, fin])

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


def recortar_letra(celda_th: np.ndarray) -> np.ndarray:
    """Recorta la letra de una celda que tiene un único carácter."""
    _, _, stats, _ = cv2.connectedComponentsWithStats(celda_th.astype(np.uint8), 8)
    stats = stats[1:]  # La componente 0 es el fondo.
    stats = stats[stats[:, -1] > 1]  # Igual que en analizar_campo.
    x, y, ancho, alto, _ = stats[0]
    return celda_th[y : y + alto, x : x + ancho]


def leer_condicion(celda_th: np.ndarray) -> str:
    """Devuelve "A", "L" o "R" según la forma de la letra de la condición final."""
    letra = recortar_letra(celda_th)
    alto, ancho = letra.shape
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
img, th, img_th, filas, columnas = cargar_planilla("grade_sheet_1")
print(f"Umbral de Otsu para grade_sheet_1: {th:.0f}")

plt.figure(figsize=(12, 5))
plt.subplot(121)
imshow(img, title="Planilla original")
plt.subplot(122)
imshow(~img_th, title=f"Binarizada (img < {th:.0f})")
plt.savefig(CARPETA_SALIDA / "binarizacion.png", dpi=150, bbox_inches="tight")
plt.show()

# Proyecciones: cantidad de píxeles oscuros por fila y por columna.
img_rows = np.sum(img_th, 1)
img_cols = np.sum(img_th, 0)
umbral_filas = PROPORCION_LINEA * img_th.shape[1]
umbral_columnas = PROPORCION_LINEA * img_th.shape[0]
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(img_rows)
plt.plot([0, len(img_rows)], [umbral_filas, umbral_filas], "r--")
plt.title("Píxeles oscuros por fila")
plt.xlabel("Fila")
plt.subplot(122)
plt.plot(img_cols)
plt.plot([0, len(img_cols)], [umbral_columnas, umbral_columnas], "r--")
plt.title("Píxeles oscuros por columna")
plt.xlabel("Columna")
plt.savefig(CARPETA_SALIDA / "proyecciones.png", dpi=150, bbox_inches="tight")
plt.show()

# Grilla detectada, dibujada sobre la planilla.
print(f"Líneas detectadas: {len(filas)} horizontales y {len(columnas)} verticales")
img_grilla = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
for fila in filas:
    cv2.line(img_grilla, (columnas[0], fila), (columnas[-1], fila), (255, 0, 0), 2)
for columna in columnas:
    cv2.line(img_grilla, (columna, filas[0]), (columna, filas[-1]), (0, 0, 255), 2)
plt.figure(figsize=(8, 7))
imshow(img_grilla, title="Grilla detectada", color_img=True)
plt.savefig(CARPETA_SALIDA / "grilla.png", dpi=150, bbox_inches="tight")
plt.show()

# Componentes conectadas de una celda, con un umbral fijo de 160 y con el de
# Otsu. Con 160 entran los grises del borde de las letras y algunas letras
# vecinas quedan unidas en una sola componente.
plt.figure(figsize=(8, 3))
for i, umbral in enumerate([160, th], start=1):
    celda_th = recortar_celda(img, filas, columnas, 8, "Nombre y Apellido") < umbral
    caracteres, espacios = analizar_campo(celda_th)
    img_celda = cv2.cvtColor(np.uint8(~celda_th) * 255, cv2.COLOR_GRAY2RGB)
    _, _, stats, _ = cv2.connectedComponentsWithStats(celda_th.astype(np.uint8), 8)
    for x, y, ancho, alto, _ in stats[1:]:
        cv2.rectangle(img_celda, (x, y), (x + ancho - 1, y + alto - 1), (255, 0, 0), 1)
    plt.subplot(2, 1, i)
    titulo = f"Umbral {umbral:.0f}: {caracteres} caracteres y {espacios} espacio"
    imshow(img_celda, title=titulo, color_img=True)
plt.tight_layout()
plt.savefig(CARPETA_SALIDA / "componentes.png", dpi=150, bbox_inches="tight")
plt.show()

# Letras de la condición final de los registros 1, 2 y 4, y cómo las lee
# leer_condicion.
plt.figure(figsize=(6, 3))
for i, registro in enumerate([1, 2, 4], start=1):
    celda_th = recortar_celda(img_th, filas, columnas, registro, "Condición Final")
    plt.subplot(1, 3, i)
    imshow(~recortar_letra(celda_th), title=f"Registro {registro}: {leer_condicion(celda_th)}")
plt.savefig(CARPETA_SALIDA / "condicion.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Procesamiento de las planillas --------------------------------------------
for numero in range(1, 5):
    nombre = f"grade_sheet_{numero}"
    img, th, img_th, filas, columnas = cargar_planilla(nombre)
    print(f"=== Planilla {nombre}.png ===")
    print(f"Umbral de Otsu: {th:.0f}")

    # --- Validación de cada registro ---
    resultados = []
    libres = []  # Crops del nombre de los alumnos libres.
    recuperan = []  # Crops del nombre de los alumnos que recuperan.
    for registro in range(1, 21):
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

        # Solo se informan los alumnos no aprobados con todos los campos correctos.
        if all(estado == "OK" for estado in validacion):
            celda_th = recortar_celda(img_th, filas, columnas, registro, "Condición Final")
            img_nombre = recortar_celda(img, filas, columnas, registro, "Nombre y Apellido")
            condicion = leer_condicion(celda_th)
            if condicion == "L":
                libres.append(img_nombre)
            elif condicion == "R":
                recuperan.append(img_nombre)

    # --- CSV con los resultados ---
    # Una fila por registro; el ID es su número de orden en la planilla.
    tabla = pd.DataFrame(resultados, columns=CAMPOS, index=range(1, 21))
    tabla.to_csv(CARPETA_SALIDA / f"validacion_{nombre}.csv", index_label="ID")

    # --- Imagen de alumnos no aprobados ---
    # Dos columnas: los libres a la izquierda y los que recuperan a la derecha.
    # En la fila i de la figura, plt.subplot numera la columna izquierda como
    # 2 * i + 1 y la derecha como 2 * i + 2.
    filas_figura = max(len(libres), len(recuperan), 1)
    plt.figure(figsize=(8, 0.6 * filas_figura + 1))
    grupos = [("LIBRE (L)", "red", libres), ("RECUPERA (R)", "darkorange", recuperan)]
    for columna, (titulo, color, nombres) in enumerate(grupos, start=1):
        for i in range(filas_figura):
            plt.subplot(filas_figura, 2, 2 * i + columna)
            if i < len(nombres):
                imshow(nombres[i])
            # Sin marco: los nombres se leen como una lista, no como imágenes.
            plt.axis("off")
            if i == 0:
                plt.title(f"{titulo}: {len(nombres)} alumnos", color=color)
    plt.suptitle(f"Alumnos no aprobados - {nombre}")
    plt.tight_layout()
    plt.savefig(CARPETA_SALIDA / f"no_aprobados_{nombre}.png", dpi=150, bbox_inches="tight")
    plt.show()

print(f"Resultados guardados en: {CARPETA_SALIDA}")
