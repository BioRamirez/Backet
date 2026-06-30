
import os
import re
from pathlib import Path

ruta_scripts = Path(r"D:\CORPONOR 2025\Backet\python_Proyect\Scripts_Analisis")
ruta_maestro = ruta_scripts / "16_Script_Maestro.py"

print("\n🔍 ANALIZANDO Script_Maestro.py...\n")

# 1. Leer Script Maestro
with open(ruta_maestro, "r", encoding="utf-8") as f:
    maestro = f.read()

# 2. Extraer scripts
patron = r'python_scripts\s*=\s*\[(.*?)\]'
bloque = re.search(patron, maestro, re.S).group(1)
scripts_llamados = re.findall(r'"(.+?\.py)"', bloque)

print("📌 SCRIPTS DETECTADOS EN ORDEN:\n")
for s in scripts_llamados:
    print(" →", s)


# -------------------------------------------------------------
# 3. Patrón mejorado para detectar *cualquier archivo generado*
# -------------------------------------------------------------
# Detecta: .xlsx .csv .png .jpg .jpeg .pdf .txt .json .md .shp .tif .xml .html .docx .zip etc.
patron_archivos = r'["\']([^"\']+\.[A-Za-z0-9]{2,5})["\']'


def buscar_archivos(script_path):
    """Detecta cualquier archivo mencionado dentro del script."""
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            cont = f.read()

        # Encuentra cualquier cosa que termine con .EXTENSION
        encontrados = re.findall(patron_archivos, cont)

        # (Opcional) Filtrar falsos positivos si salen rutas internas
        # encontrados = [e for e in encontrados if "." in Path(e).name]

        return encontrados

    except:
        return []


# -------------------------------------------------------------
# 4. Mostrar resultados finales
# -------------------------------------------------------------
print("\n\n📌 MAPA REAL: Script → Archivos generados\n")

for script in scripts_llamados:
    path_script = ruta_scripts / script

    if not path_script.exists():
        print(f"❗ Script no encontrado: {script}")
        continue

    archivos = buscar_archivos(path_script)

    if archivos:
        print(f"🐍 {script}")
        for a in archivos:
            print(f"   → 📄 {a}")
    else:
        print(f"🐍 {script} (no genera archivos detectables)")




























































