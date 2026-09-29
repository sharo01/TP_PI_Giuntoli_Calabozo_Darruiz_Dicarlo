"""Ecualización local de histograma para el Problema 1 del TP de PI."""

import argparse
import textwrap
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages


VENTANAS_POR_DEFECTO = [(5, 5), (15, 15), (31, 31), (61, 61)]


def ecualizacion_local(
    imagen: np.ndarray, tamano_ventana: tuple[int, int]
) -> np.ndarray:
    """Ecualiza cada píxel usando el histograma de su vecindad MxN.

    La entrada debe ser una imagen 2D uint8. Los bordes se extienden
    replicando los valores extremos. En una ventana constante se conserva el
    valor original para evitar una división por cero y cambios artificiales.
    """
    if not isinstance(imagen, np.ndarray) or imagen.ndim != 2:
        raise ValueError("La imagen debe ser un arreglo NumPy 2D en escala de grises.")
    if imagen.dtype != np.uint8:
        raise ValueError("La imagen debe tener tipo uint8 (intensidades entre 0 y 255).")
    if len(tamano_ventana) != 2:
        raise ValueError("El tamaño de ventana debe ser una tupla (M, N).")

    alto_ventana, ancho_ventana = map(int, tamano_ventana)
    if alto_ventana <= 0 or ancho_ventana <= 0:
        raise ValueError("Las dimensiones de la ventana deben ser positivas.")
    if imagen.size == 0:
        raise ValueError("La imagen no puede estar vacía.")

    # Para dimensiones pares se fija el ancla en el elemento M//2, N//2.
    arriba = alto_ventana // 2
    abajo = alto_ventana - 1 - arriba
    izquierda = ancho_ventana // 2
    derecha = ancho_ventana - 1 - izquierda
    extendida = np.pad(
        imagen,
        ((arriba, abajo), (izquierda, derecha)),
        mode="edge",
    )

    alto, ancho = imagen.shape
    cantidad = alto_ventana * ancho_ventana
    salida = np.empty_like(imagen)

    # Se desliza horizontalmente el histograma: en cada paso se quita la
    # columna que sale de la ventana y se agrega la que ingresa.
    for fila in range(alto):
        histograma = np.bincount(
            extendida[fila : fila + alto_ventana, :ancho_ventana].ravel(),
            minlength=256,
        )

        for columna in range(ancho):
            nivel = int(imagen[fila, columna])
            primer_nivel = int(np.argmax(histograma > 0))
            cdf_min = int(histograma[primer_nivel])
            denominador = cantidad - cdf_min

            if denominador <= 0:
                salida[fila, columna] = nivel
            else:
                cdf_nivel = int(histograma[: nivel + 1].sum())
                transformado = 255.0 * (cdf_nivel - cdf_min) / denominador
                salida[fila, columna] = np.uint8(
                    np.clip(np.floor(transformado + 0.5), 0, 255)
                )

            if columna + 1 < ancho:
                columna_que_sale = np.bincount(
                    extendida[fila : fila + alto_ventana, columna], minlength=256
                )
                columna_que_entra = np.bincount(
                    extendida[
                        fila : fila + alto_ventana,
                        columna + ancho_ventana,
                    ],
                    minlength=256,
                )
                histograma += columna_que_entra - columna_que_sale

    return salida


def ecualizacion_global(imagen: np.ndarray) -> np.ndarray:
    """Ecualiza globalmente una imagen uint8 para usarla como referencia."""
    histograma = np.bincount(imagen.ravel(), minlength=256)
    cdf = histograma.cumsum()
    niveles_presentes = np.flatnonzero(histograma)
    cdf_min = int(cdf[niveles_presentes[0]])
    denominador = imagen.size - cdf_min
    if denominador <= 0:
        return imagen.copy()
    tabla = np.clip(
        np.floor((cdf - cdf_min) * 255.0 / denominador + 0.5), 0, 255
    ).astype(np.uint8)
    return tabla[imagen]


def analizar_tamano(texto: str) -> tuple[int, int]:
    """Convierte, por ejemplo, '15x21' en (15, 21)."""
    try:
        partes = texto.lower().split("x")
        if len(partes) != 2:
            raise ValueError
        alto, ancho = (int(parte) for parte in partes)
        if alto <= 0 or ancho <= 0:
            raise ValueError
        return alto, ancho
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"'{texto}' no es válido; use dos enteros positivos, por ejemplo 15x21."
        ) from exc


def guardar_imagen(ruta: Path, arreglo: np.ndarray) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arreglo).save(ruta)


def guardar_comparacion(
    original: np.ndarray,
    global_: np.ndarray,
    locales: dict[tuple[int, int], np.ndarray],
    ruta: Path,
) -> None:
    paneles = [("Original", original), ("Ecualización global", global_)]
    paneles.extend(
        (f"Local {alto}x{ancho}", resultado)
        for (alto, ancho), resultado in locales.items()
    )
    columnas = 3
    filas = (len(paneles) + columnas - 1) // columnas
    figura, ejes = plt.subplots(filas, columnas, figsize=(12, 4 * filas))
    ejes = np.asarray(ejes).reshape(-1)
    for eje, (titulo, imagen) in zip(ejes, paneles):
        eje.imshow(imagen, cmap="gray", vmin=0, vmax=255)
        eje.set_title(titulo)
        eje.axis("off")
    for eje in ejes[len(paneles) :]:
        eje.axis("off")
    figura.suptitle("Problema 1 — Ecualización local de histograma", fontsize=15)
    figura.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ruta, dpi=180, bbox_inches="tight")
    plt.close(figura)


def guardar_informe(
    original: np.ndarray,
    global_: np.ndarray,
    locales: dict[tuple[int, int], np.ndarray],
    ruta: Path,
    analisis_referencia: bool,
) -> None:
    """Genera un informe PDF con la comparación y el análisis del ejercicio."""
    paneles = [("Imagen original", original), ("Ecualización global", global_)]
    paneles.extend(
        (f"Ecualización local {alto}x{ancho}", resultado)
        for (alto, ancho), resultado in locales.items()
    )
    columnas = 3
    filas = (len(paneles) + columnas - 1) // columnas
    ruta.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(ruta) as pdf:
        figura, ejes = plt.subplots(filas, columnas, figsize=(11.7, 8.3))
        ejes = np.asarray(ejes).reshape(-1)
        for eje, (titulo, imagen) in zip(ejes, paneles):
            eje.imshow(imagen, cmap="gray", vmin=0, vmax=255)
            eje.set_title(titulo, fontsize=10)
            eje.axis("off")
        for eje in ejes[len(paneles) :]:
            eje.axis("off")
        figura.suptitle("Problema 1 — Ecualización local de histograma", fontsize=15)
        figura.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(figura, dpi=180)
        plt.close(figura)

        paginas = [
            (
                "1. Método y detalles recuperados",
                [
                    (
                        "Descripción e implementación",
                        "Se procesó una imagen TIFF monocromática de 256×256 píxeles y 8 bits. "
                        "Para cada posición se obtiene el histograma local de M×N píxeles. "
                        "La salida se calcula como round(255 × (CDF(g) − CDFmín) / "
                        "(M·N − CDFmín)), donde g es el nivel del píxel central y CDFmín "
                        "es el primer valor no nulo de la acumulada. En ventanas constantes "
                        "se conserva el nivel original. En los bordes se replican los píxeles "
                        "extremos, equivalente a BORDER_REPLICATE.",
                    ),
                    (
                        "Detalles observados",
                        "La ecualización local permite distinguir cinco figuras que apenas "
                        "contrastan con sus fondos en la imagen original: (1) un pequeño "
                        "cuadrado claro dentro del recuadro negro superior izquierdo; (2) "
                        "un segmento diagonal ascendente en el recuadro superior derecho; "
                        "(3) la letra minúscula «a» en el recuadro central; (4) cuatro líneas "
                        "horizontales paralelas en el recuadro inferior izquierdo; y (5) un "
                        "disco circular en el recuadro inferior derecho.",
                    ),
                    (
                        "Comparación global/local",
                        "La ecualización global aumenta el contraste del fondo y hace más "
                        "visible su textura, pero los detalles débiles continúan con poco "
                        "contraste respecto de sus recuadros. La transformación local adapta "
                        "el mapeo a cada zona y recupera las cinco figuras con claridad.",
                    ),
                ],
            ),
            (
                "2. Influencia del tamaño de ventana",
                [
                    (
                        "5×5",
                        "La ventana pequeña responde a variaciones muy localizadas. Los "
                        "detalles aparecen, pero también se amplifica intensamente el ruido "
                        "del fondo, que queda con aspecto granulado.",
                    ),
                    (
                        "15×15",
                        "Las figuras se distinguen con fuerza. Aún se amplifican puntos y "
                        "pequeños bloques del fondo, por lo que la imagen resulta visualmente "
                        "fragmentada.",
                    ),
                    (
                        "31×31",
                        "Ofrece el compromiso más favorable en esta imagen: las cinco figuras "
                        "se leen bien y el fondo resulta menos dominante que con 5×5 o 15×15.",
                    ),
                    (
                        "61×61",
                        "La vecindad más grande reduce la respuesta a diferencias pequeñas "
                        "dentro de los recuadros: la letra, las líneas y la diagonal se ven más "
                        "apagadas, y las figuras pierden nitidez.",
                    ),
                    (
                        "Conclusión",
                        "El tamaño adecuado depende de la escala espacial de los detalles y "
                        "del ruido. Para esta imagen, 31×31 brinda la mejor lectura general "
                        "entre los tamaños comparados; ventanas menores aumentan el ruido y "
                        "ventanas mayores suavizan el contraste local. La elección se basa en "
                        "inspección visual de los resultados y puede variar para otras imágenes.",
                    ),
                ],
            ),
        ]

        if not analisis_referencia:
            tamanos = ", ".join(f"{alto}x{ancho}" for alto, ancho in locales)
            paginas = [
                (
                    "Método y configuración ejecutada",
                    [
                        (
                            "Entrada",
                            f"Se procesó una imagen monocromática de "
                            f"{original.shape[1]}x{original.shape[0]} píxeles.",
                        ),
                        (
                            "Ventanas",
                            f"Se aplicaron ventanas de {tamanos}. La primera página "
                            "muestra la imagen original, la ecualización global y los "
                            "resultados locales correspondientes.",
                        ),
                        (
                            "Interpretación",
                            "El efecto de cada tamaño depende de la imagen procesada. "
                            "Compare la visibilidad de los detalles con la textura y el "
                            "ruido del fondo para elegir la ventana adecuada.",
                        ),
                    ],
                )
            ]

        for titulo, secciones in paginas:
            figura = plt.figure(figsize=(11.7, 8.3))
            figura.text(0.07, 0.94, titulo, fontsize=16, weight="bold", va="top")
            y = 0.86
            for encabezado, parrafo in secciones:
                figura.text(0.07, y, encabezado, fontsize=12, weight="bold", va="top")
                y -= 0.045
                texto = textwrap.fill(parrafo, width=112)
                figura.text(
                    0.07,
                    y,
                    texto,
                    fontsize=10.5,
                    va="top",
                    linespacing=1.3,
                )
                y -= 0.025 + 0.021 * (texto.count("\n") + 1)
            pdf.savefig(figura)
            plt.close(figura)


def ejecutar(
    ruta_imagen: Path, carpeta_salida: Path, ventanas: Sequence[tuple[int, int]]
) -> None:
    if not ruta_imagen.is_file():
        raise FileNotFoundError(f"No se encontró la imagen de entrada: {ruta_imagen}")

    original = np.asarray(Image.open(ruta_imagen).convert("L"), dtype=np.uint8)
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    guardar_imagen(carpeta_salida / "imagen_original.png", original)

    global_ = ecualizacion_global(original)
    guardar_imagen(carpeta_salida / "ecualizacion_global.png", global_)

    locales: dict[tuple[int, int], np.ndarray] = {}
    for ventana in ventanas:
        resultado = ecualizacion_local(original, ventana)
        locales[ventana] = resultado
        alto, ancho = ventana
        guardar_imagen(
            carpeta_salida / f"ecualizacion_local_{alto}x{ancho}.png", resultado
        )
        print(f"Generada ventana {alto}x{ancho}")

    guardar_comparacion(
        original,
        global_,
        locales,
        carpeta_salida / "comparacion_ventanas.png",
    )
    imagen_referencia = (
        Path(__file__).resolve().parent
        / "Img"
        / "Imagen_con_detalles_escondidos.tif"
    )
    analisis_referencia = (
        ruta_imagen.resolve() == imagen_referencia.resolve()
        and list(ventanas) == VENTANAS_POR_DEFECTO
    )
    guardar_informe(
        original,
        global_,
        locales,
        carpeta_salida / "Informe_Problema1.pdf",
        analisis_referencia,
    )
    print(f"Resultados guardados en: {carpeta_salida}")


def main() -> None:
    raiz = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Aplica ecualización local de histograma a una imagen gris."
    )
    parser.add_argument(
        "--imagen",
        type=Path,
        default=raiz / "Img" / "Imagen_con_detalles_escondidos.tif",
        help="Ruta de la imagen de entrada (por defecto, la imagen del TP).",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=raiz / "Resultados" / "Problema1",
        help="Carpeta donde se guardan las imágenes de salida.",
    )
    parser.add_argument(
        "--ventanas",
        nargs="+",
        type=analizar_tamano,
        default=VENTANAS_POR_DEFECTO,
        metavar="MxN",
        help="Tamaños de ventana, por ejemplo: 5x5 15x15 31x31 61x61.",
    )
    argumentos = parser.parse_args()
    ejecutar(argumentos.imagen, argumentos.salida, argumentos.ventanas)


if __name__ == "__main__":
    main()
