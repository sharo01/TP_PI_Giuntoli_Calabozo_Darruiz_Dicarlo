"""Problema 1: ecualización local de histograma."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Rutas relativas a la raíz del repositorio, desde donde se ejecuta el script.
IMAGEN = Path("datos") / "Imagen_con_detalles_escondidos.tif"
CARPETA_SALIDA = Path("resultados") / "problema1"
VENTANAS = [(5, 5), (15, 15), (31, 31), (61, 61)]


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


def ecualizacion_local(img: np.ndarray, ventana: tuple[int, int]) -> np.ndarray:
    """Ecualiza cada píxel con el histograma de la ventana M x N centrada en él.

    Cada ventana se ecualiza con ``cv2.equalizeHist`` y se conserva el valor
    resultante del píxel central. Los bordes se completan replicando los
    píxeles extremos.
    """
    alto_ventana, ancho_ventana = ventana
    # Posición del píxel central dentro de la ventana. También es la cantidad
    # de filas que hay sobre él y de columnas que hay a su izquierda.
    fila_centro = alto_ventana // 2
    columna_centro = ancho_ventana // 2
    # Imagen con un marco de píxeles replicados, para que la ventana de los
    # píxeles del borde no se salga de la imagen.
    img_ext = cv2.copyMakeBorder(
        img,
        fila_centro,
        alto_ventana - fila_centro - 1,
        columna_centro,
        ancho_ventana - columna_centro - 1,
        cv2.BORDER_REPLICATE,
    )

    img_eq = np.empty_like(img)
    alto, ancho = img.shape
    for fila in range(alto):
        for columna in range(ancho):
            # Recorte M x N que empieza en (fila, columna) de img_ext. Por
            # el marco, ese recorte queda centrado en el píxel (fila, columna)
            # de la imagen original.
            vecindad = img_ext[
                fila : fila + alto_ventana, columna : columna + ancho_ventana
            ]
            # Se ecualiza toda la vecindad pero solo se conserva su centro.
            vecindad_eq = cv2.equalizeHist(vecindad)
            img_eq[fila, columna] = vecindad_eq[fila_centro, columna_centro]

    return img_eq


# --- Imagen original -----------------------------------------------------------
img = cv2.imread(str(IMAGEN), cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"No se pudo leer la imagen: {IMAGEN}")

CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(CARPETA_SALIDA / "imagen_original.png"), img)
print(f"Tamaño: {img.shape}, tipo: {img.dtype}")

# Cantidad de píxeles de cada nivel de gris.
niveles, cantidades = np.unique(img, return_counts=True)
for nivel, cantidad in zip(niveles, cantidades):
    print(f"Nivel {nivel}: {cantidad} píxeles")

# Histograma en escala logarítmica: con escala lineal, los niveles 0 y 227
# tapan al resto.
plt.figure(figsize=(12, 4))
plt.subplot(121)
imshow(img, title="Imagen original", ticks=True)
plt.subplot(122)
plt.hist(img.flatten(), bins=256, range=(0, 256), log=True)
plt.title("Histograma")
plt.savefig(CARPETA_SALIDA / "histograma_original.png", dpi=180, bbox_inches="tight")
plt.show()

# --- Ecualización global -------------------------------------------------------
img_heq = cv2.equalizeHist(img)
cv2.imwrite(str(CARPETA_SALIDA / "ecualizacion_global.png"), img_heq)

# Nivel que le asigna la ecualización global a cada nivel original.
for nivel in niveles:
    print(f"Nivel {nivel} -> {img_heq[img == nivel][0]}")

plt.figure(figsize=(12, 4))
plt.subplot(121)
imshow(img_heq, title="Ecualización global", ticks=True)
plt.subplot(122)
plt.hist(img_heq.flatten(), bins=256, range=(0, 256), log=True)
plt.title("Histograma")
plt.savefig(CARPETA_SALIDA / "histograma_global.png", dpi=180, bbox_inches="tight")
plt.show()

# --- Borde replicado -----------------------------------------------------------
# Imagen con el borde que agrega la ventana más grande.
alto, ancho = VENTANAS[-1]
img_ext = cv2.copyMakeBorder(
    img, alto // 2, alto // 2, ancho // 2, ancho // 2, cv2.BORDER_REPLICATE
)
print(f"Tamaño con borde para una ventana de {alto}x{ancho}: {img_ext.shape}")

# Como ejemplo, se dibujan sobre la imagen con borde los límites de la original
# y la ventana de su primer píxel. Se pasa a RGB para dibujar en color; cv2
# recibe los puntos como (columna, fila).
img_ext_color = cv2.cvtColor(img_ext, cv2.COLOR_GRAY2RGB)
# Posición del primer píxel de la original dentro de la imagen con borde.
fila_0, columna_0 = alto // 2, ancho // 2
# Límites de la imagen original, en azul.
cv2.rectangle(
    img_ext_color,
    (columna_0, fila_0),
    (columna_0 + img.shape[1] - 1, fila_0 + img.shape[0] - 1),
    color=(0, 0, 255),
    thickness=1,
)
# Ventana del primer píxel, en rojo: empieza en la esquina de la imagen con borde.
cv2.rectangle(
    img_ext_color, (0, 0), (ancho - 1, alto - 1), color=(255, 0, 0), thickness=1
)
# Primer píxel de la original, en el centro de su ventana.
cv2.circle(
    img_ext_color, (columna_0, fila_0), radius=2, color=(255, 0, 0), thickness=-1
)

# Original y con borde lado a lado: los ejes en píxeles muestran el tamaño.
plt.figure(figsize=(10, 5))
plt.subplot(121)
imshow(img, title="Imagen original", ticks=True)
plt.subplot(122)
imshow(
    img_ext_color,
    title=f"Con borde replicado (ventana de {alto}x{ancho})",
    color_img=True,
    ticks=True,
)
plt.savefig(CARPETA_SALIDA / "borde_replicado.png", dpi=180, bbox_inches="tight")
plt.show()

# --- Ecualización local con cada ventana ---------------------------------------
imgs_locales = {}
for alto, ancho in VENTANAS:
    img_local = ecualizacion_local(img, (alto, ancho))
    imgs_locales[(alto, ancho)] = img_local
    nombre = f"ecualizacion_local_{alto}x{ancho}.png"
    cv2.imwrite(str(CARPETA_SALIDA / nombre), img_local)
    print(f"Generada ventana {alto}x{ancho}")

# --- Comparación entre todas las ecualizaciones --------------------------------
paneles = [("Original", img), ("Ecualización global", img_heq)]
paneles += [
    (f"Local {alto}x{ancho}", img_local)
    for (alto, ancho), img_local in imgs_locales.items()
]

columnas = 3
# División redondeando hacia arriba: filas suficientes para todos los paneles.
filas = (len(paneles) + columnas - 1) // columnas
plt.figure(figsize=(12, 4 * filas))
# plt.subplot numera los paneles desde 1, de izquierda a derecha y de
# arriba hacia abajo.
for i, (titulo, img_panel) in enumerate(paneles, start=1):
    plt.subplot(filas, columnas, i)
    imshow(img_panel, title=titulo)

plt.tight_layout()
plt.savefig(CARPETA_SALIDA / "comparacion_ventanas.png", dpi=180, bbox_inches="tight")
plt.show()

# --- Ventana elegida: 31x31 ----------------------------------------------------
# Misma figura que la de la ecualización global, para comparar los histogramas.
img_31 = imgs_locales[(31, 31)]
plt.figure(figsize=(12, 4))
plt.subplot(121)
imshow(img_31, title="Ecualización local 31x31", ticks=True)
plt.subplot(122)
plt.hist(img_31.flatten(), bins=256, range=(0, 256), log=True)
plt.title("Histograma")
plt.savefig(CARPETA_SALIDA / "histograma_local_31x31.png", dpi=180, bbox_inches="tight")
print(f"Imágenes guardadas en: {CARPETA_SALIDA}")
plt.show()
