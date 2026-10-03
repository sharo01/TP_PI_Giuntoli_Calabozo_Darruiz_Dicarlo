"""Problema 1: ecualización local de histograma."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
IMAGEN = RAIZ / "datos" / "Imagen_con_detalles_escondidos.tif"
CARPETA_SALIDA = RAIZ / "resultados" / "problema1"
VENTANAS = [(5, 5), (15, 15), (31, 31), (61, 61)]


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


def guardar_comparacion(
    ruta: Path,
    img: np.ndarray,
    img_heq: np.ndarray,
    imgs_locales: dict[tuple[int, int], np.ndarray],
) -> None:
    """Guarda una figura con la imagen original y todas sus ecualizaciones."""
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
        # vmin/vmax fijos para que matplotlib no reescale el contraste.
        plt.imshow(img_panel, cmap="gray", vmin=0, vmax=255)
        plt.title(titulo)
        plt.axis("off")

    plt.suptitle("Problema 1 — Ecualización local de histograma", fontsize=15)
    plt.tight_layout()
    plt.savefig(ruta, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    img = cv2.imread(str(IMAGEN), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {IMAGEN}")

    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(CARPETA_SALIDA / "imagen_original.png"), img)

    img_heq = cv2.equalizeHist(img)
    cv2.imwrite(str(CARPETA_SALIDA / "ecualizacion_global.png"), img_heq)

    imgs_locales = {}
    for alto, ancho in VENTANAS:
        img_local = ecualizacion_local(img, (alto, ancho))
        imgs_locales[(alto, ancho)] = img_local
        nombre = f"ecualizacion_local_{alto}x{ancho}.png"
        cv2.imwrite(str(CARPETA_SALIDA / nombre), img_local)
        print(f"Generada ventana {alto}x{ancho}")

    guardar_comparacion(
        CARPETA_SALIDA / "comparacion_ventanas.png", img, img_heq, imgs_locales
    )
    print(f"Imágenes guardadas en: {CARPETA_SALIDA}")


if __name__ == "__main__":
    main()
