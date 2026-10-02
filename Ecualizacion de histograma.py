"""Problema 1: ecualización local de histograma."""

import argparse
from pathlib import Path

import cv2
import numpy as np

VENTANAS_POR_DEFECTO = [(5, 5), (15, 15), (31, 31), (61, 61)]


def ecualizacion_local(imagen: np.ndarray, ventana: tuple[int, int]) -> np.ndarray:
    """Ecualiza cada píxel con el histograma de su vecindad.

    La imagen debe ser gris, de tipo ``uint8``. Los bordes se completan
    replicando el píxel más cercano, igual que ``cv2.BORDER_REPLICATE``.
    """
    if imagen.ndim != 2 or imagen.dtype != np.uint8:
        raise ValueError("La imagen debe ser una matriz uint8 en escala de grises.")

    alto_ventana, ancho_ventana = ventana
    if alto_ventana <= 0 or ancho_ventana <= 0:
        raise ValueError("Las dimensiones de la ventana deben ser positivas.")

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
    pixeles_ventana = alto_ventana * ancho_ventana

    for fila in range(alto):
        histograma = np.bincount(
            extendida[fila : fila + alto_ventana, :ancho_ventana].ravel(),
            minlength=256,
        )

        for columna in range(ancho):
            nivel = imagen[fila, columna]
            primer_nivel = np.argmax(histograma > 0)
            cdf_min = histograma[primer_nivel]
            denominador = pixeles_ventana - cdf_min

            if denominador == 0:
                salida[fila, columna] = nivel
            else:
                cdf = histograma[: int(nivel) + 1].sum()
                salida[fila, columna] = np.uint8(
                    np.clip(
                        np.floor(255 * (cdf - cdf_min) / denominador + 0.5),
                        0,
                        255,
                    )
                )

            if columna < ancho - 1:
                sale = extendida[fila : fila + alto_ventana, columna]
                entra = extendida[fila : fila + alto_ventana, columna + ancho_ventana]
                histograma -= np.bincount(sale, minlength=256)
                histograma += np.bincount(entra, minlength=256)

    return salida


def interpretar_ventana(texto: str) -> tuple[int, int]:
    """Convierte el texto ``15x21`` en la tupla ``(15, 21)``."""
    try:
        alto, ancho = (int(valor) for valor in texto.lower().split("x"))
        if alto > 0 and ancho > 0:
            return alto, ancho
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("Use un tamaño positivo, por ejemplo 15x21.")


def guardar_imagen(ruta: Path, imagen: np.ndarray) -> None:
    """Guarda una imagen y avisa si OpenCV no pudo escribirla."""
    if not cv2.imwrite(str(ruta), imagen):
        raise OSError(f"No se pudo guardar: {ruta}")


def procesar(imagen_path: Path, salida_path: Path, ventanas: list[tuple[int, int]]) -> None:
    """Lee una imagen, aplica las ecualizaciones y guarda los resultados."""
    original = cv2.imread(str(imagen_path), cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {imagen_path}")

    salida_path.mkdir(parents=True, exist_ok=True)
    imagenes_path = salida_path / "imagenes"
    imagenes_path.mkdir(exist_ok=True)
    guardar_imagen(imagenes_path / "imagen_original.png", original)

    global_ = cv2.equalizeHist(original)
    guardar_imagen(imagenes_path / "ecualizacion_global.png", global_)

    for ventana in ventanas:
        resultado = ecualizacion_local(original, ventana)
        alto, ancho = ventana
        guardar_imagen(
            imagenes_path / f"ecualizacion_local_{alto}x{ancho}.png", resultado
        )
        print(f"Generada ventana {alto}x{ancho}")

    print(f"Imágenes guardadas en: {imagenes_path}")


def main() -> None:
    raiz = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Ecualización local de histograma.")
    parser.add_argument(
        "--imagen",
        type=Path,
        default=raiz / "Img" / "Imagen_con_detalles_escondidos.tif",
        help="Imagen de entrada (por defecto, la del TP).",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=raiz / "Resultados" / "Problema1",
        help="Carpeta de resultados.",
    )
    parser.add_argument(
        "--ventanas",
        nargs="+",
        type=interpretar_ventana,
        default=VENTANAS_POR_DEFECTO,
        metavar="MxN",
        help="Ventanas a aplicar; por ejemplo: 5x5 15x15.",
    )
    argumentos = parser.parse_args()
    procesar(argumentos.imagen, argumentos.salida, argumentos.ventanas)


if __name__ == "__main__":
    main()
