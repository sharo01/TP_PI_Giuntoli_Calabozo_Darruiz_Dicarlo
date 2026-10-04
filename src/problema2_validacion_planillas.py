"""Problema 2: validación automática de planillas de calificaciones.

Pasos para cada planilla:

1. detectar_grilla: umbraliza la imagen y encuentra las líneas de la tabla.
2. recortar_celda: recorta cada campo de cada registro, entre las líneas.
3. componentes y analizar_campo: cuentan los caracteres y los espacios del campo.
4. validar: decide si el campo cumple la restricción de la consigna (OK o MAL).
5. medir_letra y leer_condicion: leen la letra de la Condición Final (A, L o R).

validar_planilla hace los pasos 1 a 5 y muestra el resultado por pantalla
(ítem a). guardar_no_aprobados genera la imagen del ítem b y guardar_csv, el CSV
del ítem c. El ciclo del final del script recorre las cuatro planillas (ítem d).
"""

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
    """Devuelve una fila [inicio, fin] por cada tramo de valores True de lineas_th.

    Es el método del ejemplo de los renglones de la Unidad 1. Por ejemplo:
    lineas_th = [False, False, True, True, True, False, False] devuelve [[2, 4]].
    """
    # np.diff compara cada valor con el siguiente: vale True donde lineas_th
    # cambia, o sea, un lugar antes del inicio de cada tramo y en su fin.
    # En el ejemplo, en las posiciones 1 y 4.
    cambios = np.diff(lineas_th)
    # Posiciones de los cambios, alternadas: antes del inicio, fin, antes del
    # inicio, fin, ... En el ejemplo, [1, 4].
    lineas_idxs = np.argwhere(cambios)
    # Los inicios están en las posiciones pares y quedan un lugar antes: se les
    # suma 1. En el ejemplo, [2, 4].
    lineas_idxs[0::2] += 1
    # Se agrupan de a dos: cada fila es el inicio y el fin de un tramo.
    return np.reshape(lineas_idxs, (-1, 2))


def detectar_grilla(img: np.ndarray):
    """Umbraliza la planilla y encuentra las líneas de la tabla, como propone la consigna.

    Devuelve el umbral de Otsu (th), la imagen binaria (img_th, True donde hay
    tinta) y las líneas de la tabla como pares [inicio, fin]: filas tiene las 22
    horizontales, de arriba hacia abajo, y columnas, las 8 verticales, de
    izquierda a derecha.
    """
    # 1. Umbralizar la imagen; el umbral th se elige con el método de Otsu.
    th, _ = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    img_th = img < th
    # 2. Sumar los píxeles oscuros de cada fila y de cada columna.
    img_rows = np.sum(img_th, 1)
    img_cols = np.sum(img_th, 0)
    # 3. Marcar las filas y columnas que forman parte de una línea: las que
    # tienen oscuros más de la mitad de sus píxeles.
    th_row = PROPORCION_LINEA * img_th.shape[1]  # Mitad del ancho de la imagen.
    th_col = PROPORCION_LINEA * img_th.shape[0]  # Mitad del alto de la imagen.
    img_rows_th = img_rows > th_row
    img_cols_th = img_cols > th_col
    # 4. Una línea puede tener más de un píxel de grosor, así que se busca su
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
    # El registro 1 está entre las líneas horizontales filas[1] y filas[2]
    # (entre filas[0] y filas[1] está el encabezado). El campo i está entre las
    # líneas verticales columnas[i] y columnas[i + 1]; se suma 1 porque la
    # primera columna de la tabla es "Nro.".
    i = CAMPOS.index(campo) + 1
    # Cada línea es un par [inicio, fin]. La celda empieza en el píxel siguiente
    # al fin de la línea de arriba (o de la izquierda) y termina en el inicio de
    # la línea de abajo (o de la derecha), que el slicing no incluye.
    arriba = filas[registro][1] + 1
    abajo = filas[registro + 1][0]
    izquierda = columnas[i][1] + 1
    derecha = columnas[i + 1][0]
    return img[arriba:abajo, izquierda:derecha]


def componentes(celda_th: np.ndarray) -> np.ndarray:
    """Devuelve una fila [x, y, ancho, alto, área] por cada componente de la celda.

    No incluye el fondo ni las componentes de 1 píxel. Las filas no están
    ordenadas de izquierda a derecha: OpenCV numera las componentes en el orden
    en que las encuentra al recorrer la celda de arriba hacia abajo.
    """
    _, _, stats, _ = cv2.connectedComponentsWithStats(celda_th.astype(np.uint8), 8)
    # La fila 0 de stats es siempre el fondo de la celda.
    stats = stats[1:]
    # Filtro por área de la consigna: se descartan las componentes de 1 píxel,
    # como restos de las líneas o píxeles sueltos; el punto de "1.0" mide 2.
    # stats[:, -1] es la columna del área.
    th_area = 1
    ix_area = stats[:, -1] > th_area
    return stats[ix_area, :]


def analizar_campo(celda_th: np.ndarray) -> tuple[int, int]:
    """Devuelve la cantidad de caracteres y de espacios entre palabras de una celda."""
    stats = componentes(celda_th)
    # De cada componente solo importa dónde empieza y dónde termina en
    # horizontal: el tramo [x, x + ancho]. Se ordenan de izquierda a derecha.
    tramos = []
    for x, y, ancho, alto, area in stats:
        tramos.append([x, x + ancho])
    tramos.sort()
    # Cada tramo es un carácter, salvo que se superponga en horizontal con el
    # anterior: entonces es parte del mismo carácter, como la tilde de la Ñ.
    caracteres = []  # Tramo [inicio, fin] de cada carácter.
    for inicio, fin in tramos:
        if caracteres and inicio < caracteres[-1][1]:
            caracteres[-1][1] = max(caracteres[-1][1], fin)
        else:
            caracteres.append([inicio, fin])
    # Una celda vacía no tiene caracteres ni espacios.
    if len(caracteres) == 0:
        return 0, 0
    # Un espacio es una separación entre dos caracteres seguidos mayor a
    # PROPORCION_ESPACIO veces la altura del carácter más alto de la celda.
    # stats[:, 3] es la columna del alto.
    alto_maximo = stats[:, 3].max()
    limite_espacio = PROPORCION_ESPACIO * alto_maximo
    espacios = 0
    for anterior, siguiente in zip(caracteres, caracteres[1:]):
        # Del fin de un carácter al inicio del siguiente.
        separacion = siguiente[0] - anterior[1]
        if separacion > limite_espacio:
            espacios += 1
    return len(caracteres), espacios


def validar(campo: str, caracteres: int, espacios: int) -> bool:
    """Indica si un campo cumple la restricción de la consigna."""
    if campo == "Legajo":
        # 8 caracteres en una sola palabra.
        return caracteres == 8 and espacios == 0
    if campo == "Nombre y Apellido":
        # Al menos dos palabras y no más de 12 caracteres. Los espacios no
        # cuentan como caracteres: no están entre los permitidos.
        return espacios >= 1 and caracteres <= 12
    if campo == "Condición Final":
        # Un único carácter.
        return caracteres == 1
    # Parcial 1, 2 y 3: 1 o 2 caracteres, sin espacios entre ellos.
    return 1 <= caracteres <= 2 and espacios == 0


def recortar_letra(celda_th: np.ndarray) -> np.ndarray:
    """Recorta la letra de una celda que tiene un único carácter."""
    # La única componente de la celda es la letra.
    x, y, ancho, alto, _ = componentes(celda_th)[0]
    # El rectángulo que la contiene.
    return celda_th[y : y + alto, x : x + ancho]


def medir_letra(celda_th: np.ndarray):
    """Devuelve las dos medidas de la letra de una celda en las que se basa leer_condicion."""
    letra = recortar_letra(celda_th)
    alto, ancho = letra.shape
    # Fracción de las filas de la letra que tienen tinta en su primera columna
    # (el promedio de valores True y False, que valen 1 y 0).
    tinta_columna_izquierda = letra[:, 0].mean()
    # Si hay tinta en la mitad derecha de la mitad de arriba de la letra.
    tinta_arriba_derecha = letra[: alto // 2, ancho // 2 :].any()
    return tinta_columna_izquierda, tinta_arriba_derecha


def leer_condicion(celda_th: np.ndarray) -> str:
    """Devuelve "A", "L" o "R" según la forma de la letra de la condición final."""
    tinta_columna_izquierda, tinta_arriba_derecha = medir_letra(celda_th)
    # L y R tienen un trazo vertical a la izquierda de toda su altura; en la A,
    # la primera columna tiene tinta en menos de la mitad de las filas.
    if tinta_columna_izquierda < 0.5:
        return "A"
    # L no tiene tinta arriba a la derecha; R sí.
    if not tinta_arriba_derecha:
        return "L"
    return "R"


def validar_planilla(img: np.ndarray):
    """Valida los campos de cada registro de una planilla (ítem a).

    Recibe únicamente la imagen de la planilla y muestra por pantalla si cada
    campo de cada registro es correcto (OK) o incorrecto (MAL). Devuelve:

    - resultados: una lista por registro, con "OK" o "MAL" para cada campo.
    - libres y recuperan: los recortes del nombre de los alumnos no aprobados,
      solo de los registros con todos los campos OK.
    """
    th, img_th, filas, columnas = detectar_grilla(img)
    print(f"Umbral de Otsu: {th:.0f}")
    resultados = []
    libres = []
    recuperan = []
    for registro in range(1, 21):
        print(f"Registro {registro}:")
        validacion = []  # "OK" o "MAL" para cada campo de este registro.
        for campo in CAMPOS:
            celda_th = recortar_celda(img_th, filas, columnas, registro, campo)
            caracteres, espacios = analizar_campo(celda_th)
            if validar(campo, caracteres, espacios):
                estado = "OK"
            else:
                estado = "MAL"
            validacion.append(estado)
            print(f"{campo}: {estado}")
        print()
        resultados.append(validacion)
        # Solo se informan los alumnos no aprobados con todos los campos correctos.
        if all(estado == "OK" for estado in validacion):
            celda_condicion = recortar_celda(img_th, filas, columnas, registro, "Condición Final")
            # El nombre se recorta de la imagen original, no de la binaria.
            img_nombre = recortar_celda(img, filas, columnas, registro, "Nombre y Apellido")
            condicion = leer_condicion(celda_condicion)
            if condicion == "L":
                libres.append(img_nombre)
            elif condicion == "R":
                recuperan.append(img_nombre)
    return resultados, libres, recuperan


def guardar_csv(resultados, nombre: str) -> None:
    """Guarda el resultado de la validación en un CSV, una fila por registro (ítem c)."""
    # El ID de cada fila es el número de orden del registro en la planilla.
    tabla = pd.DataFrame(resultados, columns=CAMPOS, index=range(1, 21))
    tabla.to_csv(CARPETA_SALIDA / f"validacion_{nombre}.csv", index_label="ID")


def guardar_no_aprobados(libres, recuperan, nombre: str) -> None:
    """Arma y guarda la imagen con los alumnos libres y los que recuperan (ítem b)."""
    # Dos columnas, los libres a la izquierda y los que recuperan a la derecha,
    # con un nombre por fila. Hace falta al menos una fila para los títulos.
    filas_figura = max(len(libres), len(recuperan), 1)
    # Los lugares vacíos de una columna se completan con un recorte en blanco,
    # para que los títulos de las dos columnas queden a la misma altura.
    recortes = libres + recuperan
    blanco = None
    if recortes:
        blanco = np.full_like(recortes[0], 255)
    plt.figure(figsize=(8, 0.6 * filas_figura + 1))
    grupos = [("LIBRE (L)", "red", libres), ("RECUPERA (R)", "darkorange", recuperan)]
    for columna, (titulo, color, nombres) in enumerate(grupos, start=1):
        for i in range(filas_figura):
            # plt.subplot numera los lugares de izquierda a derecha y de arriba
            # hacia abajo: en la fila i, la columna 1 es el lugar 2 * i + 1 y la
            # columna 2, el lugar 2 * i + 2.
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

# Proyecciones: cantidad de píxeles oscuros por fila y por columna, con los
# umbrales th_row y th_col (los mismos pasos que en detectar_grilla).
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

# Líneas detectadas. Cada una es un par [inicio, fin]; como en estas planillas
# las líneas miden 1 píxel, el inicio y el fin coinciden.
print(f"Líneas detectadas: {len(filas)} horizontales y {len(columnas)} verticales")
print(f"Primeras líneas horizontales [inicio, fin]: {filas[:3].tolist()}")
print(f"Primeras líneas verticales [inicio, fin]: {columnas[:3].tolist()}")
# Grilla detectada, dibujada sobre la planilla en el inicio de cada línea.
img_grilla = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
for inicio, _ in filas:
    cv2.line(img_grilla, (columnas[0][0], inicio), (columnas[-1][0], inicio), (255, 0, 0), 2)

for inicio, _ in columnas:
    cv2.line(img_grilla, (inicio, filas[0][0]), (inicio, filas[-1][0]), (0, 0, 255), 2)

plt.figure(figsize=(8, 7))
imshow(img_grilla, title="Grilla detectada", color_img=True)
plt.savefig(CARPETA_SALIDA / "grilla.png", dpi=150, bbox_inches="tight")
plt.show()

# Ejemplo con la celda Legajo del registro 1: dónde se recorta, sus
# componentes y lo que cuenta analizar_campo.
celda_ejemplo = recortar_celda(img_th, filas, columnas, 1, "Legajo")
alto_celda, ancho_celda = celda_ejemplo.shape
print(f"Celda Legajo del registro 1: {alto_celda} x {ancho_celda} píxeles")
print("Sus componentes, una por fila [x, y, ancho, alto, área]:")
print(componentes(celda_ejemplo))
for campo in ["Legajo", "Nombre y Apellido"]:
    celda_ejemplo = recortar_celda(img_th, filas, columnas, 1, campo)
    caracteres, espacios = analizar_campo(celda_ejemplo)
    print(f"{campo} del registro 1: caracteres = {caracteres}, espacios = {espacios}")

# Ejemplo de validación de un registro con campos correctos e incorrectos, el
# registro 19 de grade_sheet_2: lo que cuenta analizar_campo en cada campo y lo
# que decide validar.
img_2 = leer_planilla("grade_sheet_2")
_, img_th_2, filas_2, columnas_2 = detectar_grilla(img_2)
print("Registro 19 de grade_sheet_2:")
for campo in CAMPOS:
    celda_th = recortar_celda(img_th_2, filas_2, columnas_2, 19, campo)
    caracteres, espacios = analizar_campo(celda_th)
    if validar(campo, caracteres, espacios):
        estado = "OK"
    else:
        estado = "MAL"
    print(f"{campo}: caracteres = {caracteres}, espacios = {espacios} -> {estado}")

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
    # Celda en blanco y negro, con un rectángulo rojo alrededor de cada componente.
    img_celda = cv2.cvtColor((~celda_th).astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)
    for x, y, ancho, alto, _ in componentes(celda_th):
        cv2.rectangle(img_celda, (x, y), (x + ancho - 1, y + alto - 1), (255, 0, 0), 1)
    plt.subplot(1, 2, i)
    imshow(img_celda, title=f"Umbral {umbral:.0f}: {caracteres} caracteres", color_img=True)

plt.tight_layout()
plt.savefig(CARPETA_SALIDA / "componentes.png", dpi=150, bbox_inches="tight")
plt.show()

# Letras de la condición final de los registros 1, 2 y 4, las dos medidas de
# medir_letra y cómo las lee leer_condicion.
plt.figure(figsize=(6, 3))
for i, registro in enumerate([1, 2, 4], start=1):
    celda_th = recortar_celda(img_th, filas, columnas, registro, "Condición Final")
    tinta_columna_izquierda, tinta_arriba_derecha = medir_letra(celda_th)
    condicion = leer_condicion(celda_th)
    print(
        f"Registro {registro}: tinta_columna_izquierda = {tinta_columna_izquierda:.2f},"
        f" tinta_arriba_derecha = {tinta_arriba_derecha} -> {condicion}"
    )
    plt.subplot(1, 3, i)
    imshow(~recortar_letra(celda_th), title=f"Registro {registro}: {condicion}")

plt.savefig(CARPETA_SALIDA / "condicion.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Procesamiento de las planillas (ítem d) -----------------------------------
for numero in range(1, 5):
    nombre = f"grade_sheet_{numero}"
    print(f"=== Planilla {nombre}.png ===")
    img = leer_planilla(nombre)
    resultados, libres, recuperan = validar_planilla(img)  # Ítem a.
    # Resumen de la planilla: cuántos registros tienen todos los campos OK y,
    # de esos alumnos, cuántos están libres y cuántos recuperan.
    registros_ok = 0
    for validacion in resultados:
        if "MAL" not in validacion:
            registros_ok += 1
    print(
        f"Registros con todos los campos OK: {registros_ok},"
        f" libres: {len(libres)}, recuperan: {len(recuperan)}"
    )
    guardar_csv(resultados, nombre)  # Ítem c.
    guardar_no_aprobados(libres, recuperan, nombre)  # Ítem b.

print(f"Resultados guardados en: {CARPETA_SALIDA}")
