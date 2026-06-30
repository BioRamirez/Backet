import os
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip


# ---------------------------------------------------------
# CONFIGURACIÓN "E:\DCIM\FOTOS_INFORMES"
# ---------------------------------------------------------
CARPETA_BASE = Path(r"D:\Forestal Consultores\2026\FAUNA\BD\REPTILES\FotosInformeR")
CARPETA_COPIAS = Path(r"D:\Forestal Consultores\2026\FAUNA\BD\REPTILES\FotosInformeRCopia")

marca_texto = "Ramirez_Juan"

IMAGENES_EXT = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
VIDEOS_EXT = [".mp4", ".mov", ".avi", ".AVI",  ".mkv"]


# ---------------------------------------------------------
# CREAR COPIA SEGURA
# ---------------------------------------------------------
def copiar_original(ruta_original):
    """Copia la foto/video a la carpeta de backup manteniendo las subcarpetas."""
    ruta_original = Path(ruta_original)
    ruta_relativa = ruta_original.relative_to(CARPETA_BASE)

    destino = CARPETA_COPIAS / ruta_relativa
    destino.parent.mkdir(parents=True, exist_ok=True)

    if not destino.exists():
        shutil.copy2(ruta_original, destino)
        print(f"📁 Copia creada: {destino}")
    else:
        print(f"✓ Copia YA existía: {destino}")

    return destino  # ← devolvemos la ruta a la copia para marcar esa


# ---------------------------------------------------------
# FUNCIONES PARA IMÁGENES
# ---------------------------------------------------------
def poner_marca_imagen(ruta_img):
    try:
        img = Image.open(ruta_img).convert("RGBA")
        w, h = img.size

        tamanio_fuente_img = int(w * 0.025)

        try:
            font = ImageFont.truetype("arial.ttf", tamanio_fuente_img)
        except:
            font = ImageFont.load_default()

        capa = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(capa)

        bbox = draw.textbbox((0, 0), marca_texto, font=font)
        w_texto = bbox[2] - bbox[0]

        x = (w - w_texto) // 2 + 1600
        y = int(h * 0.65)

        draw.text((x, y), marca_texto, fill=(255, 255, 255, 150), font=font)

        resultado = Image.alpha_composite(img, capa).convert("RGB")
        resultado.save(ruta_img)

        print(f"✔ Imagen marcada: {ruta_img}")

    except Exception as e:
        print(f"⚠ ERROR en imagen {ruta_img}: {e}")


# ---------------------------------------------------------
# FUNCIONES PARA VIDEOS (SIN IMAGEMAGICK)
# ---------------------------------------------------------
# ---------------------------------------------------------
# FUNCIONES PARA VIDEOS (SIN IMAGEMAGICK)
# ---------------------------------------------------------
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip

def medir_texto(draw, texto, font):
    """Compatibilidad para Pillow: usa textbbox() si existe, si no textsize()."""
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0,0), texto, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    else:
        return draw.textsize(texto, font=font)

def poner_marca_video(ruta_video):
    try:
        ruta_video = Path(ruta_video)
        ruta_temp = ruta_video.with_suffix(".temp" + ruta_video.suffix)

        clip = VideoFileClip(str(ruta_video))
        W, H = clip.size

        # Tamaño del texto proporcional
        tamanio_fuente = int(W * 0.025)

        # Generar marca como imagen
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Fuente
        try:
            font = ImageFont.truetype("arial.ttf", tamanio_fuente)
        except:
            font = ImageFont.load_default()

        # Medir texto (compatible con todas las versiones)
        text_w, text_h = medir_texto(draw, marca_texto, font)

        # MISMA POSICIÓN QUE LAS FOTOS → abajo derecha
        x = (W - text_w) // 2 + 300
        y = int(H * 0.65)

        # Pintar texto en semitransparente
        draw.text((x, y), marca_texto, font=font, fill=(255, 255, 255, 180))

        # Convertir marca a clip de video
        marca_clip = ImageClip(np.array(img)).set_duration(clip.duration)

        # Componer
        final = CompositeVideoClip([clip, marca_clip])

        # Exportar
        final.write_videofile(
            str(ruta_temp),
            codec="libx264",
            audio_codec="aac",
            preset="medium",
            threads=4,
            verbose=False
        )

        os.replace(ruta_temp, ruta_video)
        print(f"🎬 Video marcado: {ruta_video}")

    except Exception as e:
        print(f"⚠ ERROR en video {ruta_video}: {e}")

# ---------------------------------------------------------
# PROCESAR CARPETAS
# ---------------------------------------------------------
def procesar_carpeta(carpeta):
    carpeta = Path(carpeta)

    for root, _, files in os.walk(carpeta):
        for f in files:
            ruta_original = Path(root) / f
            ext = ruta_original.suffix.lower()

            # 1. Crear copia segura
            ruta_copia = copiar_original(ruta_original)

            # 2. Marcar la copia (NUNCA el original)
            if ext in IMAGENES_EXT:
                poner_marca_imagen(ruta_copia)

            elif ext in VIDEOS_EXT:
                poner_marca_video(ruta_copia)


# ---------------------------------------------------------
# EJECUCIÓN
# ---------------------------------------------------------
if __name__ == "__main__":
    print("=== COPIANDO ARCHIVOS Y APLICANDO MARCAS ===\n")
    procesar_carpeta(CARPETA_BASE)
    print("\n=== PROCESO COMPLETADO ===")


































# =========================================================
# IMPORTS
# =========================================================
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
CARPETA_BASE = Path(r"D:\Forestal Consultores\2026\FAUNA\BD\AVES\FotosInforme")
CARPETA_COPIAS = Path(r"D:\Forestal Consultores\2026\FAUNA\BD\AVES\FotosInformeCopia")

MARCA_FIRMA = "Ramírez_Juan/Carreño_Jair"
FUENTE_FIRMA = r"C:\fonts\Signature.ttf"

IMAGENES_EXT = [".jpg", ".jpeg", ".png"]
VIDEOS_EXT = [".mp4", ".mov", ".avi", ".mkv"]


# =========================================================
# COPIA SEGURA (NO TOCA ORIGINALES)
# =========================================================
def copiar_original(ruta_original):
    ruta_original = Path(ruta_original)
    ruta_relativa = ruta_original.relative_to(CARPETA_BASE)
    destino = CARPETA_COPIAS / ruta_relativa

    destino.parent.mkdir(parents=True, exist_ok=True)

    if not destino.exists():
        shutil.copy2(ruta_original, destino)
        print(f"📁 Copia creada: {destino}")
    else:
        print(f"✓ Copia existente: {destino}")

    return destino


# =========================================================
# MARCA TIPO FIRMA EN IMÁGENES
# =========================================================
def poner_firma_imagen(ruta_img):
    try:
        img = Image.open(ruta_img).convert("RGBA")
        w, h = img.size

        tamanio_fuente = max(40, int(w * 0.04))

        try:
            font = ImageFont.truetype(FUENTE_FIRMA, tamanio_fuente)
        except:
            font = ImageFont.load_default()

        firma = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(firma)

        bbox = draw.textbbox((0, 0), MARCA_FIRMA, font=font)
        fw = bbox[2] - bbox[0]
        fh = bbox[3] - bbox[1]

        x = w - fw - int(w * 0.05)
        y = h - fh - int(h * 0.08)

        draw.text(
            (x, y),
            MARCA_FIRMA,
            fill=(255, 255, 255, 110),
            font=font
        )

        firma = firma.rotate(
            -6,
            resample=Image.BICUBIC,
            center=(x + fw // 2, y + fh // 2)
        )

        resultado = Image.alpha_composite(img, firma).convert("RGB")
        resultado.save(ruta_img)

        print(f"✔ Imagen firmada: {ruta_img}")

    except Exception as e:
        print(f"⚠ Error en imagen {ruta_img}: {e}")


# =========================================================
# MARCA TIPO FIRMA EN VIDEOS
# =========================================================
def poner_firma_video(ruta_video):
    try:
        ruta_video = Path(ruta_video)
        ruta_temp = ruta_video.with_suffix(".temp" + ruta_video.suffix)

        clip = VideoFileClip(str(ruta_video))
        W, H = clip.size

        tamanio_fuente = int(W * 0.035)

        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype(FUENTE_FIRMA, tamanio_fuente)
        except:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), MARCA_FIRMA, font=font)
        fw = bbox[2] - bbox[0]
        fh = bbox[3] - bbox[1]

        x = W - fw - int(W * 0.05)
        y = H - fh - int(H * 0.08)

        draw.text(
            (x, y),
            MARCA_FIRMA,
            fill=(255, 255, 255, 120),
            font=font
        )

        img = img.rotate(
            -6,
            resample=Image.BICUBIC,
            center=(x + fw // 2, y + fh // 2)
        )

        firma_clip = ImageClip(np.array(img)).set_duration(clip.duration)
        final = CompositeVideoClip([clip, firma_clip])

        final.write_videofile(
            str(ruta_temp),
            codec="libx264",
            audio_codec="aac",
            preset="medium",
            threads=4,
            verbose=False
        )

        os.replace(ruta_temp, ruta_video)
        print(f"🎬 Video firmado: {ruta_video}")

    except Exception as e:
        print(f"⚠ Error en video {ruta_video}: {e}")


# =========================================================
# PROCESAMIENTO DE CARPETAS
# =========================================================
def procesar_carpeta(carpeta):
    carpeta = Path(carpeta)

    for root, _, files in os.walk(carpeta):
        for f in files:
            ruta_original = Path(root) / f
            ext = ruta_original.suffix.lower()

            ruta_copia = copiar_original(ruta_original)

            if ext in IMAGENES_EXT:
                poner_firma_imagen(ruta_copia)
            elif ext in VIDEOS_EXT:
                poner_firma_video(ruta_copia)


# =========================================================
# EJECUCIÓN
# =========================================================
if __name__ == "__main__":
    print("=== COPIANDO ARCHIVOS Y APLICANDO FIRMA ===\n")
    procesar_carpeta(CARPETA_BASE)
    print("\n=== PROCESO COMPLETADO ===")
