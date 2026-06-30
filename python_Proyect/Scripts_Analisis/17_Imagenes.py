from pathlib import Path
from PIL import Image
import cv2
import imagehash
import shutil

ruta_base = Path(r"D:\CORPONOR 2025\FOTOS")
ruta_salida = ruta_base / "SELECCION_3_MEJORES"
ruta_salida.mkdir(exist_ok=True)

extensiones_img = [".jpg", ".jpeg", ".png"]
extensiones_video = [".mp4", ".avi", ".mov", ".mkv"]

carpetas_ignoradas = {"HERPETOS", "METODOS"}


# ------------------------------------------------------
# 1. Identificar MUNICIPIOS (carpetas MAYÚSCULAS)
# ------------------------------------------------------
def es_municipio(carpeta: Path):
    if not carpeta.is_dir():
        return False

    nombre = carpeta.name.upper()

    if nombre in carpetas_ignoradas:
        return False  # ignorar HERPETOS y METODOS

    return carpeta.name.isupper()


# ------------------------------------------------------
# 2. Medición de nitidez (Laplacian Variance)
# ------------------------------------------------------
def calcular_nitidez(path):
    try:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0
        return cv2.Laplacian(img, cv2.CV_64F).var()
    except:
        return 0


# ------------------------------------------------------
# 3. Hash perceptual (duplicados)
# ------------------------------------------------------
def obtener_hash(path):
    try:
        with Image.open(path) as img:
            return imagehash.phash(img)
    except:
        return None


# ------------------------------------------------------
# 4. Extraer especie = primeras dos palabras del archivo
# ------------------------------------------------------
def obtener_especie(path):
    nombre = path.stem.replace("_", " ").replace("-", " ")
    partes = nombre.split()

    if len(partes) < 2:  # No tiene formato de especie
        return None

    return partes[0] + " " + partes[1]  # Género + especie


# ------------------------------------------------------
# 5. Recopilar imágenes organizadas
# ------------------------------------------------------
dicc = {}

for municipio in ruta_base.iterdir():
    if not es_municipio(municipio):
        continue

    print(f"\n=== Analizando municipio: {municipio.name} ===")

    for archivo in municipio.rglob("*"):

        # ----------------------------------
        # 1. IGNORAR CARPETAS HERPETOS / METODOS
        # ----------------------------------
        if any(p.upper() in carpetas_ignoradas for p in archivo.parts):
            continue

        # -------------------------------
        # 2. SI NO ES IMAGEN NI VIDEO → ignorar
        # -------------------------------
        suf = archivo.suffix.lower()

        if suf not in extensiones_img and suf not in extensiones_video:
            continue

        # -------------------------------
        # 3. ORGANIZAR VIDEOS
        # -------------------------------
        if suf in extensiones_video:
            carpeta_dest = ruta_salida / municipio.name / "VIDEOS"
            carpeta_dest.mkdir(parents=True, exist_ok=True)
            shutil.copy(archivo, carpeta_dest / archivo.name)
            continue

        # -------------------------------
        # 4. IGNORAR IMÁGENES TIPO “IMG_7458.jpg”
        # -------------------------------
        nombre = archivo.stem.upper()

        if (
            nombre.startswith("IMG_") or
            nombre.startswith("DSC") or
            nombre.startswith("PXL_") or
            nombre.startswith("WHATSAPP") or
            nombre.startswith("CAMERA") or
            nombre.startswith("PHOTO")
        ):
            continue

        # Ignorar nombres como IMG1234, DSC0001, P1010456
        if nombre.replace("_", "").replace("-", "").isalnum() and len(nombre.split()) < 2:
            continue

        # -------------------------------
        # 5. PROCESAR IMAGEN VÁLIDA
        # -------------------------------
        especie = obtener_especie(archivo)
        if especie is None:
            continue

        nitidez = calcular_nitidez(archivo)
        hash_img = obtener_hash(archivo)

        try:
            with Image.open(archivo) as im:
                res = im.size[0] * im.size[1]
        except:
            res = 0

        peso = archivo.stat().st_size

        info = {
            "path": archivo,
            "nitidez": float(nitidez),
            "res": res,
            "peso": peso,
            "hash": hash_img
        }

        dicc.setdefault(municipio.name, {})
        dicc[municipio.name].setdefault(especie, [])
        dicc[municipio.name][especie].append(info)


# ------------------------------------------------------
# 6. Eliminar duplicados
# ------------------------------------------------------
def eliminar_duplicados(lista):
    unicos = []
    hashes = set()

    for entry in lista:
        if entry["hash"] is None:
            continue
        if entry["hash"] in hashes:
            continue
        hashes.add(entry["hash"])
        unicos.append(entry)

    return unicos


# ------------------------------------------------------
# 7. Seleccionar 3 mejores por especie (sin renombrar)
# ------------------------------------------------------
for municipio, especies in dicc.items():

    carpeta_dest = ruta_salida / municipio
    carpeta_dest.mkdir(parents=True, exist_ok=True)

    for especie, fotos in especies.items():

        fotos = eliminar_duplicados(fotos)

        fotos = sorted(
            fotos,
            key=lambda x: (x["nitidez"], x["res"], x["peso"]),
            reverse=True
        )

        mejores = fotos[:1]

        for img in mejores:
            destino = carpeta_dest / img["path"].name
            shutil.copy(img["path"], destino)

        print(f"{municipio} - {especie}: {len(mejores)} imágenes guardadas.")
