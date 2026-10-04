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
# Una separación entre caracteres consecutivos mayor a esta fracción de la
# altura del carácter más alto de la celda es un espacio entre palabras. En las
# cuatro planillas, entre letras de una misma palabra la separación es como
# mucho 0,36 veces esa altura y entre palabras, al menos 0,86 veces.
PROPORCION_ESPACIO = 0.6


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


def leer_planilla(nombre: str) -> np.ndarray:
    """Lee una planilla de la carpeta de datos, en escala de grises."""
    img = cv2.imread(str(CARPETA_DATOS / f"{nombre}.png"), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {nombre}.png")
    return img


def inicio_fin(lineas_th: np.ndarray) -> np.ndarray:
    """Devuelve el inicio y el fin de cada línea marcada en lineas_th, una por fila.

    Es el método del ejemplo de los renglones de la Unidad 1. np.diff vale True
    donde lineas_th cambia de valor: justo antes del inicio de cada línea y en
    su fin, alternados.
    """
    x = np.diff(lineas_th)
    lineas_idxs = np.argwhere(x)
    # Los inicios están en las posiciones pares y quedan un píxel antes.
    lineas_idxs[0::2] += 1
    return np.reshape(lineas_idxs, (-1, 2))


def detectar_grilla(img: np.ndarray):
    """Umbraliza la planilla y devuelve el umbral, la imagen binaria y las líneas de la tabla."""
    # Como propone la consigna; el umbral se elige con el método de Otsu.
    th, _ = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    img_th = img < th
    img_rows = np.sum(img_th, 1)  # Píxeles oscuros de cada fila.
    img_cols = np.sum(img_th, 0)  # Píxeles oscuros de cada columna.
    th_row = PROPORCION_LINEA * img_th.shape[1]
    th_col = PROPORCION_LINEA * img_th.shape[0]
    img_rows_th = img_rows > th_row
    img_cols_th = img_cols > th_col
    # Una línea puede tener más de un píxel de grosor, así que se busca su
    # inicio y su fin.
    filas = inicio_fin(img_rows_th)
    columnas = inicio_fin(img_cols_th)
    # La tabla tiene 22 líneas horizontales (encabezado y 20 registros) y 8
    # verticales.
    if len(filas) != 22 or len(columnas) != 8:
        raise ValueError("No se pudo detectar la grilla de la planilla.")
    return th, img_th, filas, columnas


def recortar_celda(img, filas, columnas, registro: int, campo: str) -> np.ndarray:
    """Devuelve la celda de un registro y un campo, sin las líneas de la tabla."""
    # El registro 1 está entre la segunda y la tercera línea horizontal (la
    # primera franja es el encabezado). Los campos empiezan en la segunda
    # columna, porque la primera es "Nro.".
    i = CAMPOS.index(campo) + 1
    # La celda va desde el píxel siguiente al fin de una línea hasta el
    # anterior al inicio de la siguiente.
    return img[filas[registro][1] + 1 : filas[registro + 1][0], columnas[i][1] + 1 : columnas[i + 1][0]]


def componentes(celda_th: np.ndarray) -> np.ndarray:
    """Devuelve las componentes de la celda (x, y, ancho, alto, área) sin el fondo ni las de 1 px."""
    _, _, stats, _ = cv2.connectedComponentsWithStats(celda_th.astype(np.uint8), 8)
    stats = stats[1:]  # La componente 0 es el fondo.
    # Se descartan las componentes de 1 píxel, como restos de las líneas o
    # píxeles sueltos; el punto de "1.0" mide 2.
    th_area = 1
    ix_area = stats[:, -1] > th_area
    return stats[ix_area, :]


def analizar_campo(celda_th: np.ndarray) -> tuple[int, int]:
    """Devuelve la cantidad de caracteres y de espacios entre palabras de una celda."""
    stats = componentes(celda_th)
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
    # Un espacio es una separación mayor a PROPORCION_ESPACIO veces la altura
    # del carácter más alto de la celda (stats[:, 3] es el alto).
    espacios = 0
    for anterior, siguiente in zip(caracteres, caracteres[1:]):
        if siguiente[0] - anterior[1] > PROPORCION_ESPACIO * stats[:, 3].max():
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
    x, y, ancho, alto, _ = componentes(celda_th)[0]
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


def validar_planilla(img: np.ndarray):
    """Valida los campos de cada registro de una planilla.

    Recibe únicamente la imagen de la planilla y muestra por pantalla si cada
    campo de cada registro es correcto (OK) o incorrecto (MAL). Devuelve esos
    resultados y los recortes del nombre de los alumnos libres y de los que
    recuperan.
    """
    th, img_th, filas, columnas = detectar_grilla(img)
    print(f"Umbral de Otsu: {th:.0f}")
    resultados = []
    libres = []  # Recortes del nombre de los alumnos libres.
    recuperan = []  # Recortes del nombre de los alumnos que recuperan.
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
    return resultados, libres, recuperan


# --- Detalle del procesamiento de grade_sheet_1 --------------------------------
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
img = leer_planilla("grade_sheet_1")
th, img_th, filas, columnas = detectar_grilla(img)
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
th_row = PROPORCION_LINEA * img_th.shape[1]
th_col = PROPORCION_LINEA * img_th.shape[0]
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(img_rows)
plt.plot([0, len(img_rows)], [th_row, th_row], "r--")
plt.title("Píxeles oscuros por fila")
plt.xlabel("Fila")
plt.subplot(122)
plt.plot(img_cols)
plt.plot([0, len(img_cols)], [th_col, th_col], "r--")
plt.title("Píxeles oscuros por columna")
plt.xlabel("Columna")
plt.savefig(CARPETA_SALIDA / "proyecciones.png", dpi=150, bbox_inches="tight")
plt.show()

# Grilla detectada, dibujada sobre la planilla en el inicio de cada línea.
print(f"Líneas detectadas: {len(filas)} horizontales y {len(columnas)} verticales")
img_grilla = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
for inicio, _ in filas:
    cv2.line(img_grilla, (columnas[0][0], inicio), (columnas[-1][0], inicio), (255, 0, 0), 2)

for inicio, _ in columnas:
    cv2.line(img_grilla, (inicio, filas[0][0]), (inicio, filas[-1][0]), (0, 0, 255), 2)

plt.figure(figsize=(8, 7))
imshow(img_grilla, title="Grilla detectada", color_img=True)
plt.savefig(CARPETA_SALIDA / "grilla.png", dpi=150, bbox_inches="tight")
plt.show()

# Componentes conectadas del Parcial 1 del registro 2 de grade_sheet_4, que
# vale "100", con un umbral fijo de 160 y con el de Otsu. Con 160 entran los
# grises del borde de los dígitos y los dos ceros quedan unidos.
img_4 = leer_planilla("grade_sheet_4")
th_4, _, filas_4, columnas_4 = detectar_grilla(img_4)
celda_100 = recortar_celda(img_4, filas_4, columnas_4, 2, "Parcial 1")
plt.figure(figsize=(8, 2.5))
for i, umbral in enumerate([160, th_4], start=1):
    celda_th = celda_100 < umbral
    caracteres, _ = analizar_campo(celda_th)
    img_celda = cv2.cvtColor((~celda_th).astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)
    for x, y, ancho, alto, _ in componentes(celda_th):
        cv2.rectangle(img_celda, (x, y), (x + ancho - 1, y + alto - 1), (255, 0, 0), 1)
    plt.subplot(1, 2, i)
    imshow(img_celda, title=f"Umbral {umbral:.0f}: {caracteres} caracteres", color_img=True)

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
    print(f"=== Planilla {nombre}.png ===")
    resultados, libres, recuperan = validar_planilla(leer_planilla(nombre))
    # --- CSV con los resultados ---
    # Una fila por registro; el ID es su número de orden en la planilla.
    tabla = pd.DataFrame(resultados, columns=CAMPOS, index=range(1, 21))
    tabla.to_csv(CARPETA_SALIDA / f"validacion_{nombre}.csv", index_label="ID")
    # --- Imagen de alumnos no aprobados ---
    # Dos columnas: los libres a la izquierda y los que recuperan a la derecha.
    # En la fila i de la figura, plt.subplot numera la columna izquierda como
    # 2 * i + 1 y la derecha como 2 * i + 2.
    filas_figura = max(len(libres), len(recuperan), 1)
    # Los lugares vacíos de una columna se completan con un recorte en blanco,
    # para que los títulos de las dos columnas queden a la misma altura.
    recortes = libres + recuperan
    blanco = np.full_like(recortes[0], 255) if recortes else None
    plt.figure(figsize=(8, 0.6 * filas_figura + 1))
    grupos = [("LIBRE (L)", "red", libres), ("RECUPERA (R)", "darkorange", recuperan)]
    for columna, (titulo, color, nombres) in enumerate(grupos, start=1):
        for i in range(filas_figura):
            plt.subplot(filas_figura, 2, 2 * i + columna)
            if i < len(nombres):
                imshow(nombres[i])
            elif blanco is not None:
                imshow(blanco)
            # Sin marco: los nombres se leen como una lista, no como imágenes.
            plt.axis("off")
            if i == 0:
                plt.title(f"{titulo}: {len(nombres)} alumnos", color=color)
    plt.suptitle(f"Alumnos no aprobados - {nombre}")
    plt.tight_layout()
    plt.savefig(CARPETA_SALIDA / f"no_aprobados_{nombre}.png", dpi=150, bbox_inches="tight")
    plt.show()

print(f"Resultados guardados en: {CARPETA_SALIDA}")
