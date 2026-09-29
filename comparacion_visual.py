"""Funciones para generar la comparación visual y el informe del TP."""

import textwrap
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def crear_comparacion(
    original: np.ndarray,
    global_: np.ndarray,
    locales: dict[tuple[int, int], np.ndarray],
) -> plt.Figure:
    """Devuelve una figura con la imagen original y sus resultados."""
    paneles = [("Original", original), ("Ecualización global", global_)]
    paneles += [(f"Local {alto}x{ancho}", img) for (alto, ancho), img in locales.items()]

    columnas = 3
    filas = (len(paneles) + columnas - 1) // columnas
    figura, ejes = plt.subplots(filas, columnas, figsize=(12, 4 * filas))

    for eje, (titulo, imagen) in zip(np.ravel(ejes), paneles):
        eje.imshow(imagen, cmap="gray", vmin=0, vmax=255)
        eje.set_title(titulo)
        eje.axis("off")
    for eje in np.ravel(ejes)[len(paneles) :]:
        eje.axis("off")

    figura.suptitle("Problema 1 — Ecualización local de histograma")
    figura.tight_layout()
    return figura


def pagina_de_texto(
    original: np.ndarray, ventanas: list[tuple[int, int]]
) -> plt.Figure:
    """Crea la página escrita del informe."""
    tamanos = ", ".join(f"{alto}×{ancho}" for alto, ancho in ventanas)
    secciones = [
        (
            "Método",
            f"Se procesó una imagen en escala de grises de {original.shape[1]}×"
            f"{original.shape[0]} píxeles. Para cada píxel se calcula el histograma "
            "de una ventana local y se transforma su intensidad con la función de "
            "distribución acumulada. En los bordes se utiliza "
            "cv2.BORDER_REPLICATE.",
        ),
        (
            "Implementación",
            "OpenCV se usa para leer y guardar las imágenes, replicar los bordes y "
            "aplicar la ecualización global (cv2.equalizeHist). NumPy permite "
            "calcular y actualizar los histogramas locales. Se compararon las "
            f"ventanas: {tamanos}.",
        ),
        (
            "Análisis",
            "La ecualización global mejora el contraste de toda la imagen con una "
            "única transformación. La local se adapta a cada vecindad, por lo que "
            "puede destacar detalles tenues. Las ventanas pequeñas realzan cambios "
            "locales pero también pueden amplificar ruido; las grandes producen un "
            "resultado más estable, aunque reducen detalles pequeños.",
        ),
        (
            "Conclusión",
            "El tamaño de ventana debe elegirse según la escala de los detalles y "
            "el nivel de ruido de la imagen. La comparación visual de la primera "
            "página permite seleccionar el compromiso más conveniente para cada "
            "caso de estudio.",
        ),
    ]

    figura = plt.figure(figsize=(11.7, 8.3))
    figura.text(0.07, 0.94, "Informe — método y análisis", fontsize=16, weight="bold")
    posicion_y = 0.86
    for titulo, parrafo in secciones:
        figura.text(0.07, posicion_y, titulo, fontsize=12, weight="bold")
        posicion_y -= 0.04
        texto = textwrap.fill(parrafo, width=112)
        figura.text(0.07, posicion_y, texto, fontsize=10.5, va="top", linespacing=1.4)
        posicion_y -= 0.045 + 0.023 * (texto.count("\n") + 1)
    return figura


def guardar_resultados_visual(
    original: np.ndarray,
    global_: np.ndarray,
    locales: dict[tuple[int, int], np.ndarray],
    ventanas: list[tuple[int, int]],
    carpeta: Path,
) -> None:
    """Guarda la comparación PNG y el informe PDF de dos páginas."""
    comparacion = crear_comparacion(original, global_, locales)
    comparacion.savefig(carpeta / "comparacion_ventanas.png", dpi=180)

    with PdfPages(carpeta / "Informe_Problema1.pdf") as pdf:
        pdf.savefig(comparacion)
        texto = pagina_de_texto(original, ventanas)
        pdf.savefig(texto)
        plt.close(texto)
    plt.close(comparacion)
