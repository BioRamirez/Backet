import subprocess
from datetime import datetime


# Carpetas y archivos que SÍ se subirán
RUTAS = [
    "python_Proyect",
    "R_STUDIO",
    "README.md",
    ".gitignore",
    "requirements.txt"
]


def ejecutar(comando):
    try:
        resultado = subprocess.run(
            comando,
            check=True,
            text=True,
            capture_output=True
        )
        return True, resultado.stdout

    except subprocess.CalledProcessError as e:
        print(e.stderr)
        return False, e.stderr


def auto_commit():

    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")
    mensaje = f"Actualización automática {fecha}"

    print("Agregando únicamente el código...")

    ok, _ = ejecutar(["git", "add"] + RUTAS)

    if not ok:
        return

    ok, estado = ejecutar(["git", "status", "--porcelain"])

    if not ok:
        return

    if estado.strip() == "":
        print("No hay cambios.")
        return

    ok, _ = ejecutar(["git", "commit", "-m", mensaje])

    if not ok:
        return

    ok, rama = ejecutar(["git", "branch", "--show-current"])

    if not ok:
        return

    ok, salida = ejecutar(["git", "push", "origin", rama.strip()])

    if ok:
        print("✅ Proyecto actualizado correctamente.")
    else:
        print("❌ Error al subir el proyecto.")
        print(salida)


if __name__ == "__main__":
    auto_commit()