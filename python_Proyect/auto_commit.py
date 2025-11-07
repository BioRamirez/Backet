import subprocess
from datetime import datetime

def ejecutar_comando(comando):
    """Ejecuta un comando del sistema y muestra la salida en consola."""
    try:
        resultado = subprocess.run(
            comando,
            check=True,
            text=True,
            capture_output=True
        )
        print(resultado.stdout)
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Error ejecutando comando git: {e}")
        print(e.stderr)

def auto_commit():
    """Agrega, commitea y sube los cambios automáticamente."""
    # 1️⃣ Mensaje con fecha y hora
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mensaje = f"Auto-commit: actualización {fecha}"

    print("🔍 Verificando cambios...")
    ejecutar_comando(["git", "status"])

    print("➕ Agregando cambios al área de preparación...")
    ejecutar_comando(["git", "add", "."])

    print(f"💾 Realizando commit con mensaje: '{mensaje}'")
    ejecutar_comando(["git", "commit", "-m", mensaje])

    print("🚀 Subiendo cambios al repositorio remoto...")
    ejecutar_comando(["git", "push", "origin", "main"])

    print("✅ Cambios subidos correctamente a GitHub.")

if __name__ == "__main__":
    auto_commit()

