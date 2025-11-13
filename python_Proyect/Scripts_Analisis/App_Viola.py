# --------------------------------------------------------
# run_voila_app.py
# Ejecuta una app Voila directamente desde Python
# --------------------------------------------------------

import subprocess
import os

# Ruta del archivo Jupyter Notebook (.ipynb)
notebook_path = r"D:\CORPONOR 2025\Dashboards\mi_app_voila.ipynb"

# Verificar que el archivo existe
if not os.path.exists(notebook_path):
    raise FileNotFoundError(f"No se encontró el notebook: {notebook_path}")

# Ejecutar Voila (abre el navegador automáticamente)
subprocess.run(["voila", notebook_path, "--theme=dark", "--port=8866", "--no-browser"])
