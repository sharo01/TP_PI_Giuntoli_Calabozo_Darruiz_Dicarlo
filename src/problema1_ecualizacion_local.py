"""Problema 1: ecualización local de histograma."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
IMAGEN = RAIZ / "datos" / "Imagen_con_detalles_escondidos.tif"
CARPETA_SALIDA = RAIZ / "resultados" / "problema1"
VENTANAS = [(5, 5), (15, 15), (31, 31), (61, 61)]


def ecualizacion_local(imagen: np.ndarray, ventana: tuple[int, int]) -> np.ndarray:
    """Ecualiza cada píxel con el histograma de la ventana M x N centrada en él.

    Cada ventana se ecualiza con ``cv2.equalizeHist`` y se conserva el valor
    resultante del píxel central. Los bordes se completan replicando los
    píxeles extremos.
    """
    alto_ventana, ancho_ventana = ventana
    arriba = alto_ventana // 2
    izquierda = ancho_ventana // 2
    extendida = cv2.copyMakeBorder(
        imagen,
        arriba,
        alto_ventana - arriba - 1,
        izquierda,
        ancho_ventana - izquierda - 1,
        cv2.BORDER_REPLICATE,
    )

    salida = np.empty_like(imagen)
    alto, ancho = imagen.shape
    for fila in range(alto):
        for columna in range(ancho):
            vecindad = extendida[
                fila : fila + alto_ventana, columna : columna + ancho_ventana
            ]
            salida[fila, columna] = cv2.equalizeHist(vecindad)[arriba, izquierda]

    return salida


def guardar_comparacion(
    ruta: Path,
    original: np.ndarray,
    global_: np.ndarray,
    locales: dict[tuple[int, int], np.ndarray],
) -> None:
    """Guarda una figura con la imagen original y todas sus ecualizaciones."""
    paneles = [("Original", original), ("Ecualización global", global_)]
    paneles += [
        (f"Local {alto}x{ancho}", imagen) for (alto, ancho), imagen in locales.items()
    ]

    columnas = 3
    filas = (len(paneles) + columnas - 1) // columnas
    figura, ejes = plt.subplots(filas, columnas, figsize=(12, 4 * filas))
    ejes = np.ravel(ejes)
    for eje, (titulo, imagen) in zip(ejes, paneles):
        eje.imshow(imagen, cmap="gray", vmin=0, vmax=255)
        eje.set_title(titulo)
    for eje in ejes:
        eje.axis("off")

    figura.suptitle("Problema 1 — Ecualización local de histograma", fontsize=15)
    figura.tight_layout()
    figura.savefig(ruta, dpi=180, bbox_inches="tight")
    plt.close(figura)


def main() -> None:
    original = cv2.imread(str(IMAGEN), cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {IMAGEN}")

    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(CARPETA_SALIDA / "imagen_original.png"), original)

    global_ = cv2.equalizeHist(original)
    cv2.imwrite(str(CARPETA_SALIDA / "ecualizacion_global.png"), global_)

    locales = {}
    for alto, ancho in VENTANAS:
        resultado = ecualizacion_local(original, (alto, ancho))
        locales[(alto, ancho)] = resultado
        nombre = f"ecualizacion_local_{alto}x{ancho}.png"
        cv2.imwrite(str(CARPETA_SALIDA / nombre), resultado)
        print(f"Generada ventana {alto}x{ancho}")

    guardar_comparacion(
        CARPETA_SALIDA / "comparacion_ventanas.png", original, global_, locales
    )
    print(f"Imágenes guardadas en: {CARPETA_SALIDA}")


if __name__ == "__main__":
    main()
