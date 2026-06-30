import subprocess
from datetime import datetime
import os
import sys


def ejecutar_comando(comando):
    """Ejecuta un comando y devuelve (éxito, salida)."""
    try:
        resultado = subprocess.run(
            comando,
            check=True,
            text=True,
            capture_output=True
        )
        return True, resultado.stdout.strip()

    except subprocess.CalledProcessError as e:
        mensaje = e.stderr.strip() if e.stderr else e.stdout.strip()
        return False, mensaje


def auto_commit():

    print("=" * 60)
    print("🚀 BACKET AUTO COMMIT")
    print("=" * 60)

    # Verificar que estamos en un repositorio Git
    ok, _ = ejecutar_comando(["git", "rev-parse", "--is-inside-work-tree"])

    if not ok:
        print("❌ Esta carpeta no pertenece a un repositorio Git.")
        return

    # Verificar cambios
    print("🔍 Verificando cambios...")

    ok, estado = ejecutar_comando(["git", "status", "--porcelain"])

    if not ok:
        print("❌ Error obteniendo el estado del repositorio.")
        print(estado)
        return

    if estado == "":
        print("✅ No hay cambios para guardar.")
        return

    # Agregar archivos
    print("➕ Agregando archivos...")

    ok, salida = ejecutar_comando(["git", "add", "."])

    if not ok:
        print("❌ Error en git add")
        print(salida)
        return

    # Commit
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mensaje = f"Auto-commit: actualización {fecha}"

    print("💾 Creando commit...")

    ok, salida = ejecutar_comando(
        ["git", "commit", "-m", mensaje]
    )

    if not ok:
        print("❌ Error realizando el commit.")
        print(salida)
        return

    print(salida)

    # Obtener rama
    ok, rama = ejecutar_comando(
        ["git", "branch", "--show-current"]
    )

    if not ok:
        print("❌ No fue posible obtener la rama.")
        return

    print(f"🌿 Rama actual: {rama}")

    # Push
    print("☁️ Subiendo cambios a GitHub...")

    ok, salida = ejecutar_comando(
        ["git", "push", "origin", rama]
    )

    if not ok:

        print("\n❌ ERROR DURANTE EL PUSH\n")
        print(salida)

        if "exceeds GitHub's file size limit" in salida:
            print("\n⚠️ GitHub rechazó el push porque existen archivos mayores de 100 MB.")
            print("Revise el .gitignore o elimine esos archivos del historial.")

        return

    print(salida)

    print("\n✅ Todo salió correctamente.")
    print("=" * 60)


if __name__ == "__main__":
    auto_commit()