

import subprocess
from datetime import datetime

def ejecutar_comando(comando):
    """Ejecuta un comando del sistema y devuelve salida o error."""
    try:
        resultado = subprocess.run(
            comando,
            check=True,
            text=True,
            capture_output=True
        )
        return resultado.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Error ejecutando comando: {' '.join(comando)}")
        print(e.stderr)
        return None

def auto_commit():
    """Agrega, commitea y sube los cambios automáticamente al repositorio."""
    # 🕒 1️⃣ Generar mensaje con fecha y hora
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mensaje = f"Auto-commit: actualización {fecha}"

    print("🔍 Verificando estado del repositorio...")
    estado = ejecutar_comando(["git", "status", "--porcelain"])

    if not estado:
        print("✅ No hay cambios para commitear.")
        return

    print("➕ Agregando todos los cambios al área de preparación...")
    ejecutar_comando(["git", "add", "."])

    print(f"💾 Realizando commit con mensaje: '{mensaje}'")
    ejecutar_comando(["git", "commit", "-m", mensaje])

    # 🔎 Detectar la rama actual automáticamente
    rama_actual = ejecutar_comando(["git", "branch", "--show-current"]) or "master"
    print(f"🚀 Subiendo cambios al repositorio remoto (rama {rama_actual})...")
    ejecutar_comando(["git", "push", "origin", rama_actual])

    print("✅ Cambios subidos correctamente a GitHub.")

# Punto de entrada del script
if __name__ == "__main__":
    auto_commit()
