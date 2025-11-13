# --------------------------------------------------------
# run_voila_app.py
# Ejecuta una app Voila directamente desde Python
# --------------------------------------------------------

import subprocess
import os
import sys

# Ruta del archivo Jupyter Notebook (.ipynb)
notebook_path = r"D:\CORPONOR 2025\Backet\python_Proyect\Flujo_Trabajo\Flujo_Trabajo.ipynb"

# Verificar que el archivo existe
if not os.path.exists(notebook_path):
    sys.exit(f"❌ No se encontró el notebook: {notebook_path}")

# Obtener la ruta absoluta de Voila dentro del entorno virtual
voila_executable = os.path.join(os.path.dirname(sys.executable), "voila")

# Comprobar si Voila está instalado
if not os.path.exists(voila_executable + ".exe") and not os.path.exists(voila_executable):
    sys.exit("❌ Voila no está instalado en este entorno. Instálalo con:\n\npip install voila")

# Ejecutar Voila
print("🚀 Iniciando Voila...")
subprocess.run([
    voila_executable,
    notebook_path,
    "--theme=dark",
    "--port=8866",
    "--no-browser"
])

