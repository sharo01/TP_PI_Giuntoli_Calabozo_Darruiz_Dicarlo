"""Problema 2: validación automática de planillas de calificaciones."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Rutas relativas a la raíz del repositorio, desde donde se ejecuta el script.
CARPETA_DATOS = Path("datos")
CARPETA_SALIDA = Path("resultados") / "problema2"

CAMPOS =("Legajo", "Nombre y Apellido", "Parcial 1", "Parcial 2", "Parcial 3", "Condición Final")


@dataclass
class Campo:
    imagen: np.ndarray
    cantidad: int
    espacios: int
    hay_espacios: bool


def centros_de_lineas(proyeccion: np.ndarray, umbral: float) -> list[int]:
    """Devuelve el centro de cada tramo continuo que supera el umbral."""
    mascara = proyeccion > umbral
    cambios = np.flatnonzero(np.diff(np.r_[False, mascara, False]))
    return [int(round((inicio + fin - 1) / 2)) for inicio, fin in zip(cambios[::2], cambios[1::2])]


def detectar_grilla(imagen: np.ndarray) -> tuple[list[int], list[int]]:
    """Detecta las líneas de la tabla sin depender de su escala o posición."""
    tinta = imagen < 160
    filas = centros_de_lineas(tinta.sum(axis=1), tinta.shape[1] * 0.55)
    columnas = centros_de_lineas(tinta.sum(axis=0), tinta.shape[0] * 0.55)
    if len(filas) < 3 or len(columnas) != 8:
        raise ValueError("No se pudo detectar la grilla esperada de la planilla.")
    return filas, columnas


def analizar_campo(celda: np.ndarray) -> Campo:
    """Obtiene caracteres y separaciones entre palabras de una celda."""
    binaria = (celda < 160).astype(np.uint8)
    _, _, estadisticas, _ = cv2.connectedComponentsWithStats(binaria, 8)
    componentes = estadisticas[1:][estadisticas[1:, cv2.CC_STAT_AREA] >= 3]
    componentes = componentes[np.argsort(componentes[:, cv2.CC_STAT_LEFT])]
    separaciones = []
    for anterior, siguiente in zip(componentes, componentes[1:]):
        fin = anterior[cv2.CC_STAT_LEFT] + anterior[cv2.CC_STAT_WIDTH]
        separaciones.append(siguiente[cv2.CC_STAT_LEFT] - fin)
    # El espacio entre letras es de hasta 3 píxeles en las planillas provistas.
    espacios = sum(hueco > 5 for hueco in separaciones)
    return Campo(celda, len(componentes), espacios, espacios > 0)


def validar(campos: list[Campo]) -> list[str]:
    """Aplica las restricciones de longitud y separación del enunciado."""
    legajo, nombre, parcial1, parcial2, parcial3, condicion = campos
    return [
        "OK" if legajo.cantidad == 8 and not legajo.hay_espacios else "MAL",
        "OK" if nombre.espacios >= 1 and 2 <= nombre.cantidad + nombre.espacios <= 12 else "MAL",
        "OK" if 1 <= parcial1.cantidad <= 2 and not parcial1.hay_espacios else "MAL",
        "OK" if 1 <= parcial2.cantidad <= 2 and not parcial2.hay_espacios else "MAL",
        "OK" if 1 <= parcial3.cantidad <= 2 and not parcial3.hay_espacios else "MAL",
        "OK" if condicion.cantidad == 1 else "MAL",
    ]


def leer_condicion(celda: np.ndarray) -> str | None:
    """Distingue A, L y R para señalizar R y L en la imagen de salida."""
    binaria = (celda < 160).astype(np.uint8)
    _, _, estadisticas, _ = cv2.connectedComponentsWithStats(binaria, 8)
    componentes = estadisticas[1:]
    componentes = componentes[componentes[:, cv2.CC_STAT_AREA] >= 3]
    if len(componentes) != 1:
        return None
    x, y, ancho, alto, _ = componentes[0]
    caracter = binaria[y : y + alto, x : x + ancho]
    # L no tiene tinta a la derecha en su mitad superior. R tiene, además,
    # un trazo superior horizontal más ancho que A.
    if not caracter[: alto // 2, ancho // 2 :].any():
        return "L"
    promedio_superior = caracter[: max(1, alto // 3)].sum() / max(1, alto // 3)
    return "R" if promedio_superior >= ancho * 0.55 else "A"


def escribir_csv(ruta: Path, resultados: list[list[str]]) -> None:
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["ID", *CAMPOS])
        for identificador, validacion in enumerate(resultados, start=1):
            escritor.writerow([identificador, *validacion])


def guardar_no_aprobados(ruta: Path, alumnos: list[tuple[np.ndarray, str]]) -> None:
    """Guarda crops de nombre con una etiqueta RECUPERA (R) o LIBRE (L)."""
    alto_fila, margen, etiqueta = 38, 10, 150
    ancho_nombres = max((nombre.shape[1] for nombre, _ in alumnos), default=300)
    salida = np.full((margen * 2 + alto_fila * max(1, len(alumnos)), ancho_nombres + etiqueta + margen * 3, 3), 255, np.uint8)
    if not alumnos:
        cv2.putText(salida, "No hay alumnos no aprobados con registros validos.", (margen, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    for indice, (nombre, condicion) in enumerate(alumnos):
        y = margen + indice * alto_fila
        alto = min(nombre.shape[0], alto_fila - 4)
        salida[y : y + alto, margen : margen + nombre.shape[1]] = cv2.cvtColor(nombre[:alto], cv2.COLOR_GRAY2BGR)
        texto = "RECUPERA (R)" if condicion == "R" else "LIBRE (L)"
        color = (0, 140, 255) if condicion == "R" else (0, 0, 220)
        cv2.putText(salida, texto, (ancho_nombres + margen * 2, y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    if not cv2.imwrite(str(ruta), salida):
        raise OSError(f"No se pudo guardar {ruta}")


def procesar_planilla(ruta_imagen: Path, salida: Path) -> None:
    imagen = cv2.imread(str(ruta_imagen), cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        raise FileNotFoundError(f"No se pudo leer {ruta_imagen}")
    filas, columnas = detectar_grilla(imagen)
    resultados: list[list[str]] = []
    no_aprobados: list[tuple[np.ndarray, str]] = []
    # La primera franja es el encabezado; las siguientes son registros.
    for indice, (arriba, abajo) in enumerate(zip(filas[1:], filas[2:]), start=1):
        campos = [analizar_campo(imagen[arriba + 2 : abajo - 2, columnas[columna] + 2 : columnas[columna + 1] - 2]) for columna in range(1, 7)]
        validacion = validar(campos)
        resultados.append(validacion)
        print(f"Registro {indice}:")
        for nombre_campo, estado in zip(CAMPOS, validacion):
            print(f"{nombre_campo}: {estado}")
        condicion = leer_condicion(campos[-1].imagen)
        if all(estado == "OK" for estado in validacion) and condicion in {"L", "R"}:
            no_aprobados.append((campos[1].imagen, condicion))
    salida.mkdir(parents=True, exist_ok=True)
    nombre = ruta_imagen.stem
    escribir_csv(salida / f"validacion_{nombre}.csv", resultados)
    guardar_no_aprobados(salida / f"no_aprobados_{nombre}.png", no_aprobados)
    print(f"CSV: {salida / f'validacion_{nombre}.csv'}")
    print(f"Imagen: {salida / f'no_aprobados_{nombre}.png'}")


def main() -> None:
    for ruta in sorted(CARPETA_DATOS.glob("grade_sheet_[0-9].png")):
        procesar_planilla(ruta, CARPETA_SALIDA)


if __name__ == "__main__":
    main()
