"""
==============================================================
GARMIN MAP BUILDER
Versión: 1.0

Autor:
Juan Carlos Ramírez Gil

Descripción
-----------
Convierte automáticamente un GeoTIFF georreferenciado
en un Garmin Custom Map (KMZ) compatible con dispositivos
Garmin GPSMAP, Oregon, Montana y eTrex.
==============================================================
"""

from pathlib import Path
import logging
import sys
import rasterio

# ==========================================================
# CONFIGURACIÓN DEL USUARIO
# ==========================================================

INPUT_TIFF = Path(
    r"D:\Trabajo_Arboles_Atalaya\Map_atalaya.tif"
)

OUTPUT_FOLDER = INPUT_TIFF.parent

# ==========================================================
# CONFIGURACIÓN GENERAL
# ==========================================================

PROJECT_NAME = "GARMIN MAP BUILDER"
VERSION = "1.0"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)

# ==========================================================
# FUNCIONES
# ==========================================================

def print_header():

    print("=" * 60)
    print(PROJECT_NAME)
    print(f"Versión {VERSION}")
    print("=" * 60)
    print()


def validate_input():

    if not INPUT_TIFF.exists():
        raise FileNotFoundError(
            f"\nNo existe el archivo:\n{INPUT_TIFF}"
        )

    if INPUT_TIFF.suffix.lower() not in [".tif", ".tiff"]:
        raise ValueError(
            "El archivo seleccionado no es un GeoTIFF."
        )


def get_output_file():

    return OUTPUT_FOLDER / f"{INPUT_TIFF.stem}.kmz"


# ==========================================================
# MÓDULO 2
# ==========================================================

def read_geotiff():

    try:

        dataset = rasterio.open(INPUT_TIFF)

        return dataset

    except Exception as e:

        raise RuntimeError(
            f"No fue posible abrir el GeoTIFF.\n{e}"
        )


def print_dataset_info(dataset):

    print("=" * 60)
    print("INFORMACIÓN DEL GEOTIFF")
    print("=" * 60)

    print(f"Archivo           : {INPUT_TIFF.name}")
    print(f"Driver            : {dataset.driver}")
    print(f"Bandas            : {dataset.count}")

    print()

    print(f"Ancho             : {dataset.width:,} px")
    print(f"Alto              : {dataset.height:,} px")

    print()

    print(f"CRS               : {dataset.crs}")

    epsg = dataset.crs.to_epsg()

    if epsg:
        print(f"EPSG              : {epsg}")
    else:
        print("EPSG              : No identificado")

    print()

    print(f"Resolución X      : {dataset.res[0]}")
    print(f"Resolución Y      : {dataset.res[1]}")

    print()

    print("EXTENSIÓN")

    print(f"X mínimo          : {dataset.bounds.left}")
    print(f"Y mínimo          : {dataset.bounds.bottom}")
    print(f"X máximo          : {dataset.bounds.right}")
    print(f"Y máximo          : {dataset.bounds.top}")

    print()

    print(f"Tipo de dato      : {dataset.dtypes[0]}")

    try:
        print(f"Compresión        : {dataset.compression}")
    except:
        print("Compresión        : No disponible")

    print("=" * 60)


# ==========================================================
# MAIN
# ==========================================================

def main():

    dataset = None

    print_header()

    try:

        # ----------------------------
        # MÓDULO 1
        # ----------------------------

        validate_input()

        output_kmz = get_output_file()

        print(f"GeoTIFF : {INPUT_TIFF}")
        print(f"Salida  : {output_kmz}")

        print()

        # ----------------------------
        # MÓDULO 2
        # ----------------------------

        dataset = read_geotiff()

        print_dataset_info(dataset)

        print()
        print("✓ Módulo 1 completado.")
        print("✓ Módulo 2 completado.")

    except Exception as e:

        logging.exception(e)

    finally:

        if dataset is not None:
            dataset.close()
            print("\nGeoTIFF cerrado correctamente.")


if __name__ == "__main__":
    main()








# ==========================================================
# MÓDULO 3
# DIAGNÓSTICO DEL MAPA
# ==========================================================

import math


def analyze_map(dataset):
    """
    Analiza las características del GeoTIFF y determina
    si es compatible con un Garmin Custom Map.

    Parámetros
    ----------
    dataset : rasterio.DatasetReader

    Retorna
    -------
    dict
        Información diagnóstica del raster.
    """

    # ------------------------------------------------------
    # Configuración Garmin
    # ------------------------------------------------------

    TILE_SIZE = 1024
    MAX_TILES = 100

    # ------------------------------------------------------
    # Dimensiones
    # ------------------------------------------------------

    width = dataset.width
    height = dataset.height

    total_pixels = width * height

    # ------------------------------------------------------
    # Bandas
    # ------------------------------------------------------

    bands = dataset.count

    has_alpha = bands == 4

    # ------------------------------------------------------
    # Resolución
    # ------------------------------------------------------

    pixel_size_x = abs(dataset.res[0])
    pixel_size_y = abs(dataset.res[1])

    # ------------------------------------------------------
    # Sistema de coordenadas
    # ------------------------------------------------------

    epsg = dataset.crs.to_epsg()

    # ------------------------------------------------------
    # Número de mosaicos
    # ------------------------------------------------------

    tiles_x = math.ceil(width / TILE_SIZE)
    tiles_y = math.ceil(height / TILE_SIZE)

    total_tiles = tiles_x * tiles_y

    # ------------------------------------------------------
    # Compatibilidad Garmin
    # ------------------------------------------------------

    compatible = total_tiles <= MAX_TILES

    # ------------------------------------------------------
    # Tamaño aproximado en memoria
    # ------------------------------------------------------

    bytes_per_pixel = bands

    estimated_size_bytes = (
        total_pixels * bytes_per_pixel
    )

    estimated_size_mb = (
        estimated_size_bytes / 1024 / 1024
    )

    # ------------------------------------------------------
    # Resultado
    # ------------------------------------------------------

    diagnostics = {

        "width": width,

        "height": height,

        "pixels": total_pixels,

        "bands": bands,

        "alpha": has_alpha,

        "pixel_size_x": pixel_size_x,

        "pixel_size_y": pixel_size_y,

        "epsg": epsg,

        "tiles_x": tiles_x,

        "tiles_y": tiles_y,

        "total_tiles": total_tiles,

        "compatible": compatible,

        "estimated_size_mb": round(
            estimated_size_mb,
            2
        )

    }

    return diagnostics


# ==========================================================
# FUNCIÓN DE REPORTE
# ==========================================================

def print_diagnostics(info):
    """
    Imprime el diagnóstico del mapa.
    """

    print()

    print("=" * 60)
    print("DIAGNÓSTICO DEL MAPA")
    print("=" * 60)

    print(f"Dimensiones            : {info['width']:,} x {info['height']:,} px")

    print(f"Píxeles totales        : {info['pixels']:,}")

    print()

    print(f"Bandas                 : {info['bands']}")

    print(f"Canal Alpha            : {'SI' if info['alpha'] else 'NO'}")

    print()

    print(f"Resolución X           : {info['pixel_size_x']}")

    print(f"Resolución Y           : {info['pixel_size_y']}")

    print()

    print(f"EPSG                   : {info['epsg']}")

    print()

    print(f"Tiles horizontales     : {info['tiles_x']}")

    print(f"Tiles verticales       : {info['tiles_y']}")

    print(f"Tiles totales          : {info['total_tiles']}")

    print()

    print(f"Memoria estimada       : {info['estimated_size_mb']} MB")

    print()

    print(
        f"Compatible Garmin      : {'SI' if info['compatible'] else 'NO'}"
    )

    print("=" * 60)




## modulo 3

# ==========================================================
# MÓDULO 3
# DIAGNÓSTICO DEL MAPA
# ==========================================================

import math


def analyze_map(dataset):
    """
    Analiza las características del GeoTIFF y determina
    si es compatible con un Garmin Custom Map.

    Parámetros
    ----------
    dataset : rasterio.DatasetReader

    Retorna
    -------
    dict
        Información diagnóstica del raster.
    """

    # ------------------------------------------------------
    # Configuración Garmin
    # ------------------------------------------------------

    TILE_SIZE = 1024
    MAX_TILES = 100

    # ------------------------------------------------------
    # Dimensiones
    # ------------------------------------------------------

    width = dataset.width
    height = dataset.height

    total_pixels = width * height

    # ------------------------------------------------------
    # Bandas
    # ------------------------------------------------------

    bands = dataset.count

    has_alpha = bands == 4

    # ------------------------------------------------------
    # Resolución
    # ------------------------------------------------------

    pixel_size_x = abs(dataset.res[0])
    pixel_size_y = abs(dataset.res[1])

    # ------------------------------------------------------
    # Sistema de coordenadas
    # ------------------------------------------------------

    epsg = dataset.crs.to_epsg()

    # ------------------------------------------------------
    # Número de mosaicos
    # ------------------------------------------------------

    tiles_x = math.ceil(width / TILE_SIZE)
    tiles_y = math.ceil(height / TILE_SIZE)

    total_tiles = tiles_x * tiles_y

    # ------------------------------------------------------
    # Compatibilidad Garmin
    # ------------------------------------------------------

    compatible = total_tiles <= MAX_TILES

    # ------------------------------------------------------
    # Tamaño aproximado en memoria
    # ------------------------------------------------------

    bytes_per_pixel = bands

    estimated_size_bytes = (
        total_pixels * bytes_per_pixel
    )

    estimated_size_mb = (
        estimated_size_bytes / 1024 / 1024
    )

    # ------------------------------------------------------
    # Resultado
    # ------------------------------------------------------

    diagnostics = {

        "width": width,

        "height": height,

        "pixels": total_pixels,

        "bands": bands,

        "alpha": has_alpha,

        "pixel_size_x": pixel_size_x,

        "pixel_size_y": pixel_size_y,

        "epsg": epsg,

        "tiles_x": tiles_x,

        "tiles_y": tiles_y,

        "total_tiles": total_tiles,

        "compatible": compatible,

        "estimated_size_mb": round(
            estimated_size_mb,
            2
        )

    }

    return diagnostics


# ==========================================================
# FUNCIÓN DE REPORTE
# ==========================================================

def print_diagnostics(info):
    """
    Imprime el diagnóstico del mapa.
    """

    print()

    print("=" * 60)
    print("DIAGNÓSTICO DEL MAPA")
    print("=" * 60)

    print(f"Dimensiones            : {info['width']:,} x {info['height']:,} px")

    print(f"Píxeles totales        : {info['pixels']:,}")

    print()

    print(f"Bandas                 : {info['bands']}")

    print(f"Canal Alpha            : {'SI' if info['alpha'] else 'NO'}")

    print()

    print(f"Resolución X           : {info['pixel_size_x']}")

    print(f"Resolución Y           : {info['pixel_size_y']}")

    print()

    print(f"EPSG                   : {info['epsg']}")

    print()

    print(f"Tiles horizontales     : {info['tiles_x']}")

    print(f"Tiles verticales       : {info['tiles_y']}")

    print(f"Tiles totales          : {info['total_tiles']}")

    print()

    print(f"Memoria estimada       : {info['estimated_size_mb']} MB")

    print()

    print(
        f"Compatible Garmin      : {'SI' if info['compatible'] else 'NO'}"
    )

    print("=" * 60)


# ==========================================================
# PRUEBA DEL MÓDULO 3
# ==========================================================

if __name__ == "__main__":

    dataset = read_geotiff()

    diagnostico = analyze_map(dataset)

    print_diagnostics(diagnostico)

    dataset.close()

## modulo 4

# ==========================================================
# MÓDULO 4
# NORMALIZACIÓN DEL RASTER
# ==========================================================

import numpy as np


def normalize_raster(dataset):
    """
    Normaliza el GeoTIFF para prepararlo para Garmin.

    Funciones
    ---------
    - Lee únicamente las bandas RGB.
    - Elimina el canal Alpha si existe.
    - Convierte la imagen a un arreglo NumPy.
    - Verifica la resolución del raster.
    - No modifica el GeoTIFF original.

    Parámetros
    ----------
    dataset : rasterio.DatasetReader

    Retorna
    -------
    dict
    """

    # ------------------------------------------------------
    # Leer bandas
    # ------------------------------------------------------

    bands = dataset.count

    if bands >= 3:

        red = dataset.read(1)
        green = dataset.read(2)
        blue = dataset.read(3)

        image = np.dstack((red, green, blue))

        alpha_removed = bands == 4

    elif bands == 1:

        gray = dataset.read(1)

        image = np.dstack((gray, gray, gray))

        alpha_removed = False

    else:

        raise ValueError(
            "Número de bandas no compatible."
        )

    # ------------------------------------------------------
    # Información de la imagen
    # ------------------------------------------------------

    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]

    # ------------------------------------------------------
    # Resolución espacial
    # ------------------------------------------------------

    pixel_size_x = abs(dataset.res[0])
    pixel_size_y = abs(dataset.res[1])

    # ------------------------------------------------------
    # Validación resolución
    # ------------------------------------------------------

    resolution_ok = True

    if pixel_size_x <= 0 or pixel_size_y <= 0:

        resolution_ok = False

    # ------------------------------------------------------
    # Resultado
    # ------------------------------------------------------

    raster = {

        "image": image,

        "width": width,

        "height": height,

        "channels": channels,

        "alpha_removed": alpha_removed,

        "pixel_size_x": pixel_size_x,

        "pixel_size_y": pixel_size_y,

        "resolution_ok": resolution_ok

    }

    return raster


# ==========================================================
# REPORTE DEL MÓDULO 4
# ==========================================================

def print_normalization(raster):

    print()

    print("=" * 60)
    print("NORMALIZACIÓN DEL RASTER")
    print("=" * 60)

    print(f"Ancho                 : {raster['width']:,} px")

    print(f"Alto                  : {raster['height']:,} px")

    print()

    print(f"Canales               : {raster['channels']}")

    print(
        f"Alpha eliminado       : {'SI' if raster['alpha_removed'] else 'NO'}"
    )

    print()

    print(
        f"Resolución X          : {raster['pixel_size_x']}"
    )

    print(
        f"Resolución Y          : {raster['pixel_size_y']}"
    )

    print()

    print(
        f"Resolución válida     : {'SI' if raster['resolution_ok'] else 'NO'}"
    )

    print()

    print(
        f"Tipo NumPy            : {raster['image'].dtype}"
    )

    print(
        f"Shape                 : {raster['image'].shape}"
    )

    print("=" * 60)

# ==========================================================
# TEST DEL MÓDULO 4
# ==========================================================

def test_module_4():

    print()
    print("=" * 60)
    print("EJECUTANDO TEST DEL MÓDULO 4")
    print("=" * 60)

    dataset = read_geotiff()

    raster = normalize_raster(dataset)

    print_normalization(raster)

    dataset.close()

    print()
    print("✓ Test del Módulo 4 finalizado correctamente.")

if __name__ == "__main__":

    test_module_4()


#Modulo 5

# ==========================================================
# MÓDULO 5
# GENERACIÓN DE TESELAS
# ==========================================================

import math


def create_tile_grid(dataset, tile_size=1024):
    """
    Calcula la malla de mosaicos (tiles) que cubrirán
    completamente el GeoTIFF.

    No genera archivos.

    Retorna
    -------
    list(dict)
    """

    width = dataset.width
    height = dataset.height

    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)

    tiles = []

    tile_id = 1

    for row in range(tiles_y):

        for col in range(tiles_x):

            x0 = col * tile_size
            y0 = row * tile_size

            x1 = min(x0 + tile_size, width)
            y1 = min(y0 + tile_size, height)

            tile = {

                "id": tile_id,

                "row": row,

                "col": col,

                "x0": x0,

                "y0": y0,

                "x1": x1,

                "y1": y1,

                "width": x1 - x0,

                "height": y1 - y0

            }

            tiles.append(tile)

            tile_id += 1

    return tiles


# ==========================================================
# REPORTE
# ==========================================================

def print_tiles(tiles):

    print()

    print("=" * 60)
    print("TESELAS GENERADAS")
    print("=" * 60)

    print(f"Número de tiles : {len(tiles)}")

    print()

    for tile in tiles:

        print(
            f"Tile {tile['id']:03d}"
            f" | fila {tile['row']}"
            f" | columna {tile['col']}"
            f" | {tile['width']} x {tile['height']} px"
        )

    print("=" * 60)


# ==========================================================
# TEST DEL MÓDULO 5
# ==========================================================

def test_module_5():

    print()
    print("=" * 60)
    print("EJECUTANDO TEST DEL MÓDULO 5")
    print("=" * 60)

    dataset = read_geotiff()

    tiles = create_tile_grid(dataset)

    print_tiles(tiles)

    dataset.close()

    print()
    print("✓ Test del Módulo 5 finalizado correctamente.")

if __name__ == "__main__":

    test_module_5()


## Modulo 6

# ==========================================================
# MÓDULO 6
# COORDENADAS GEOGRÁFICAS DE LOS TILES
# ==========================================================

from rasterio.transform import xy


def calculate_tile_coordinates(dataset, tiles):
    """
    Calcula las coordenadas geográficas de cada tesela.

    Parámetros
    ----------
    dataset : rasterio.DatasetReader

    tiles : list

    Retorna
    -------
    list
    """

    transform = dataset.transform

    for tile in tiles:

        # -------------------------------
        # Esquinas del tile
        # -------------------------------

        west, north = xy(
            transform,
            tile["y0"],
            tile["x0"],
            offset="ul"
        )

        east, south = xy(
            transform,
            tile["y1"],
            tile["x1"],
            offset="lr"
        )

        tile["west"] = west
        tile["east"] = east
        tile["north"] = north
        tile["south"] = south

    return tiles


# ==========================================================
# REPORTE
# ==========================================================

def print_tile_coordinates(tiles):

    print()
    print("=" * 60)
    print("COORDENADAS DE LAS TESELAS")
    print("=" * 60)

    for tile in tiles:

        print()

        print(f"Tile {tile['id']:03d}")

        print(f"West  : {tile['west']}")
        print(f"East  : {tile['east']}")
        print(f"North : {tile['north']}")
        print(f"South : {tile['south']}")

    print()
    print("=" * 60)


# ==========================================================
# TEST DEL MÓDULO 6
# ==========================================================

def test_module_6():

    print()
    print("=" * 60)
    print("EJECUTANDO TEST DEL MÓDULO 6")
    print("=" * 60)

    dataset = read_geotiff()

    tiles = create_tile_grid(dataset)

    tiles = calculate_tile_coordinates(
        dataset,
        tiles
    )

    print_tile_coordinates(tiles)

    dataset.close()

    print()
    print("✓ Test del Módulo 6 finalizado correctamente.")

if __name__ == "__main__":

    test_module_6()

## Modulo 7

# ==========================================================
# MÓDULO 7
# CREACIÓN DEL DOC.KML
# ==========================================================

import simplekml


def create_kml(tiles, output_folder, image_folder="files"):
    """
    Crea el archivo doc.kml utilizado por Garmin.

    Parámetros
    ----------
    tiles : list

    output_folder : Path

    image_folder : str

    Retorna
    -------
    Path
    """

    kml = simplekml.Kml()

    for tile in tiles:

        overlay = kml.newgroundoverlay(
            name=f"Tile_{tile['id']:03d}"
        )

        overlay.icon.href = (
            f"{image_folder}/tile_{tile['id']:03d}.jpg"
        )

        overlay.latlonbox.north = tile["north"]
        overlay.latlonbox.south = tile["south"]
        overlay.latlonbox.east = tile["east"]
        overlay.latlonbox.west = tile["west"]

    output_file = output_folder / "doc.kml"

    kml.save(str(output_file))

    return output_file


# ==========================================================
# REPORTE
# ==========================================================

def print_kml_info(kml_file):

    print()

    print("=" * 60)
    print("ARCHIVO KML")
    print("=" * 60)

    print(f"Archivo generado : {kml_file}")

    print("=" * 60)


# ==========================================================
# TEST DEL MÓDULO 7
# ==========================================================

def test_module_7():

    print()
    print("=" * 60)
    print("EJECUTANDO TEST DEL MÓDULO 7")
    print("=" * 60)

    dataset = read_geotiff()

    tiles = create_tile_grid(dataset)

    tiles = calculate_tile_coordinates(
        dataset,
        tiles
    )

    kml_file = create_kml(
        tiles,
        OUTPUT_FOLDER
    )

    print_kml_info(kml_file)

    dataset.close()

    print()

    print("✓ Test del Módulo 7 finalizado correctamente.")

if __name__ == "__main__":

    test_module_7()

## Modulo 8

# ==========================================================
# MÓDULO 8
# EXPORTACIÓN DE TESELAS JPG
# ==========================================================

from pathlib import Path
from PIL import Image


def export_tiles(raster, tiles, output_folder):
    """
    Exporta cada tesela como una imagen JPG.

    Parámetros
    ----------
    raster : dict
        Resultado del módulo de normalización.

    tiles : list
        Resultado del módulo de teselas.

    output_folder : Path

    Retorna
    -------
    Path
        Carpeta donde quedaron almacenadas las imágenes.
    """

    files_folder = output_folder / "files"

    files_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    image = raster["image"]

    for tile in tiles:

        tile_array = image[
            tile["y0"]:tile["y1"],
            tile["x0"]:tile["x1"]
        ]

        img = Image.fromarray(tile_array)

        filename = (
            files_folder /
            f"tile_{tile['id']:03d}.jpg"
        )

        img.save(
            filename,
            format="JPEG",
            quality=95
        )

    return files_folder


# ==========================================================
# REPORTE
# ==========================================================

def print_exported_tiles(folder):

    print()

    print("=" * 60)
    print("TESELAS EXPORTADAS")
    print("=" * 60)

    images = sorted(folder.glob("*.jpg"))

    print(f"Carpeta : {folder}")

    print()

    print(f"Total imágenes : {len(images)}")

    print()

    for image in images:

        print(image.name)

    print("=" * 60)

# ==========================================================
# TEST DEL MÓDULO 8
# ==========================================================

def test_module_8():

    print()
    print("=" * 60)
    print("EJECUTANDO TEST DEL MÓDULO 8")
    print("=" * 60)

    dataset = read_geotiff()

    raster = normalize_raster(dataset)

    tiles = create_tile_grid(dataset)

    folder = export_tiles(
        raster,
        tiles,
        OUTPUT_FOLDER
    )

    print_exported_tiles(folder)

    dataset.close()

    print()

    print("✓ Test del Módulo 8 finalizado correctamente.")

if __name__ == "__main__":

    test_module_8()

## Modulo 9

# ==========================================================
# MÓDULO 9
# CREACIÓN DEL KMZ
# ==========================================================

import zipfile


def create_kmz(output_folder, kmz_name):
    """
    Empaqueta el doc.kml y la carpeta files en un KMZ
    compatible con Garmin.

    Parámetros
    ----------
    output_folder : Path

    kmz_name : str

    Retorna
    -------
    Path
    """

    kmz_file = output_folder / kmz_name

    doc_kml = output_folder / "doc.kml"

    files_folder = output_folder / "files"

    if not doc_kml.exists():

        raise FileNotFoundError(
            "No existe doc.kml"
        )

    if not files_folder.exists():

        raise FileNotFoundError(
            "No existe la carpeta files."
        )

    with zipfile.ZipFile(
        kmz_file,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as kmz:

        # -------------------------
        # Agregar doc.kml
        # -------------------------

        kmz.write(
            doc_kml,
            arcname="doc.kml"
        )

        # -------------------------
        # Agregar imágenes
        # -------------------------

        for image in sorted(files_folder.glob("*.jpg")):

            kmz.write(

                image,

                arcname=f"files/{image.name}"

            )

    return kmz_file


# ==========================================================
# REPORTE
# ==========================================================

def print_kmz_info(kmz_file):

    print()

    print("=" * 60)
    print("KMZ GENERADO")
    print("=" * 60)

    print(f"Archivo : {kmz_file}")

    print(f"Tamaño  : {kmz_file.stat().st_size/1024:.2f} KB")

    print("=" * 60)


# ==========================================================
# TEST DEL MÓDULO 9
# ==========================================================

def test_module_9():

    print()

    print("=" * 60)
    print("EJECUTANDO TEST DEL MÓDULO 9")
    print("=" * 60)

    kmz = create_kmz(

        OUTPUT_FOLDER,

        f"{INPUT_TIFF.stem}.kmz"

    )

    print_kmz_info(kmz)

    print()

    print("✓ Test del Módulo 9 finalizado correctamente.")

if __name__ == "__main__":

    test_module_9()

## Modulo 10

# ==========================================================
# MÓDULO 10
# VALIDACIÓN FINAL Y COPIA AL GARMIN
# ==========================================================

import zipfile
import shutil
from pathlib import Path


def validate_kmz(kmz_file):
    """
    Verifica que el KMZ tenga la estructura esperada.
    """

    if not kmz_file.exists():
        raise FileNotFoundError("No existe el KMZ.")

    with zipfile.ZipFile(kmz_file, "r") as kmz:

        files = kmz.namelist()

    if "doc.kml" not in files:
        raise RuntimeError("El KMZ no contiene doc.kml")

    jpgs = [f for f in files if f.startswith("files/")]

    if len(jpgs) == 0:
        raise RuntimeError("El KMZ no contiene imágenes.")

    return {

        "total_files": len(files),

        "tiles": len(jpgs),

        "size_mb": round(
            kmz_file.stat().st_size / 1024 / 1024,
            2
        )

    }


# ----------------------------------------------------------
# Buscar Garmin conectado
# ----------------------------------------------------------

def find_garmin():

    posibles = []

    for letra in "DEFGHIJKLMNOPQRSTUVWXYZ":

        unidad = Path(f"{letra}:/")

        if not unidad.exists():
            continue

        carpeta = unidad / "Garmin"

        if carpeta.exists():

            posibles.append(carpeta)

    if len(posibles) == 0:
        return None

    return posibles[0]


# ----------------------------------------------------------
# Copiar KMZ
# ----------------------------------------------------------

def copy_to_garmin(kmz_file):

    garmin = find_garmin()

    if garmin is None:

        print()
        print("Garmin no encontrado.")
        print("El KMZ queda listo para copiar manualmente.")
        return None

    custom = garmin / "CustomMaps"

    custom.mkdir(exist_ok=True)

    destino = custom / kmz_file.name

    shutil.copy2(kmz_file, destino)

    return destino


# ----------------------------------------------------------
# REPORTE
# ----------------------------------------------------------

def print_final_report(info, destino):

    print()

    print("=" * 60)
    print("REPORTE FINAL")
    print("=" * 60)

    print(f"Tiles              : {info['tiles']}")

    print(f"Archivos internos  : {info['total_files']}")

    print(f"Tamaño KMZ         : {info['size_mb']} MB")

    print()

    if destino is None:

        print("GPS Garmin         : NO DETECTADO")

    else:

        print("GPS Garmin         : DETECTADO")

        print(f"Archivo copiado en : {destino}")

    print()

    print("Estado             : LISTO PARA USAR")

    print("=" * 60)


# ==========================================================
# TEST DEL MÓDULO 10
# ==========================================================

def test_module_10():

    print()

    print("=" * 60)
    print("EJECUTANDO TEST DEL MÓDULO 10")
    print("=" * 60)

    kmz = OUTPUT_FOLDER / f"{INPUT_TIFF.stem}.kmz"

    info = validate_kmz(kmz)

    destino = copy_to_garmin(kmz)

    print_final_report(info, destino)

    print()

    print("✓ Proyecto finalizado correctamente.")

if __name__ == "__main__":

    test_module_10()

