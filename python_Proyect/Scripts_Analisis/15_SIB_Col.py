import os
import re

# Carpeta que contiene todos tus scripts
CARPETA = r"D:\CORPONOR 2025\Backet\python_Proyect\Scripts_Analisis"

# Expresión regular que detecta emojis unicode
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002700-\U000027BF"  # dingbats
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA70-\U0001FAFF"  # chess, symbols
    "]+",
    flags=re.UNICODE
)

def limpiar_archivo(ruta):
    with open(ruta, "r", encoding="utf-8", errors="ignore") as f:
        contenido = f.read()

    # Eliminar emojis
    contenido_limpio = EMOJI_PATTERN.sub("", contenido)

    with open(ruta, "w", encoding="utf-8") as f:
        f.write(contenido_limpio)

    print(f"Archivo limpiado: {os.path.basename(ruta)}")

for root, dirs, files in os.walk(CARPETA):
    for file in files:
        if file.lower().endswith((".py", ".r")):
            limpiar_archivo(os.path.join(root, file))
