

#-----------Crear entorno nuevo python estbale
#py -3.12 -m venv .venv

#----------Activar entorno

#.\.venv\Scripts\activate


import os
import ast
import pkgutil

def buscar_imports_en_archivo(ruta_archivo):
    """Extrae los módulos importados en un archivo .py"""
    try:
        with open(ruta_archivo, "r", encoding="utf-8") as file:
            node = ast.parse(file.read(), filename=ruta_archivo)
    except Exception:
        return set()

    imports = set()

    for n in ast.walk(node):
        # import x
        if isinstance(n, ast.Import):
            for alias in n.names:
                imports.add(alias.name.split('.')[0])

        # from x import y
        elif isinstance(n, ast.ImportFrom):
            if n.module:
                imports.add(n.module.split('.')[0])

    return imports


def buscar_imports_en_proyecto(ruta_proyecto):
    """Recorre todo el proyecto y reúne todos los imports encontrados."""
    imports_total = set()

    for root, dirs, files in os.walk(ruta_proyecto):
        for file in files:
            if file.endswith(".py"):
                ruta = os.path.join(root, file)
                imports_total |= buscar_imports_en_archivo(ruta)

    return imports_total


def detectar_paquetes_faltantes(ruta_proyecto):
    print("Buscando imports en el proyecto...\n")

    imports = buscar_imports_en_proyecto(ruta_proyecto)
    instalados = {mod.name for mod in pkgutil.iter_modules()}
    faltantes = sorted(list(imports - instalados))

    print("📦 Paquetes encontrados en el código:", len(imports))
    print("📦 Paquetes instalados:", len(instalados))
    print("\n❌ Paquetes faltantes:\n")
    for p in faltantes:
        print("-", p)

    return faltantes


# --------------------
# EJECUCIÓN
# --------------------
if __name__ == "__main__":
    ruta = r"D:\CORPONOR 2025\Backet\python_Proyect"   # <- AJUSTA TU RUTA
    detectar_paquetes_faltantes(ruta)








































