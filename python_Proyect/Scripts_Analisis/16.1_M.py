# Script_Maestro.py — Versión con CONTROL DE PROGRESO

import os
import subprocess
import sys
import time
import traceback

BASE = r"D:\CORPONOR 2025\Backet\python_Proyect\Scripts_Analisis"
R_BIN = r"C:\Program Files\R\R-4.5.2\bin\Rscript.exe"
PYTHON_CMD = sys.executable
PROGRESO = os.path.join(BASE, "progreso.txt")

python_scripts = [
    "1_Esfuerzo_muestreo.py",
    "2_Tabla_taxonomica.py",
    "2.1_Figura_orden_familia.py",
    "2.2_Ord_Fam_Esp_Rep.py",
    "3_Tabla_UnidMuestreo.py",
    "4.1_Informe_pdf.py",
    "5.1_Curvas_Abundancias.py",
    "6.1_Curvas_Frecuencias.py",
    "7_Justificacion_diversidad.py",
    "7.1_Indices_Diversidad.py",
    "8_Similitud_Cluster.py",
    "8.1_Rango_Abundancia.py",
    "9_Diversidad_Funcional.py",
    "10_Uso_Habitat.py",
    "11_Gremio_Trofico.py",
    "12_Tabla_Sensibilidad.py",
    "13_Uso.py",
    "14_Punto_M_Fauna.py",
    "14.1_Transec_M_Fauna.py",
    "14.2_Muestreo_Fauna_TB.py",
    "14.3_M_Fauna_Result_TB.py",
]

r_scripts_after_3 = [
    "3.1_Coeficiente_Variacion.r",
    "4_Analisis_Curva_iNEXT.r",
    "5_Tabla_Estimadores_Abundancia.r",
    "6_Tabla_Estimadores_Frequencia.r",
]

def cargar_progreso():
    if not os.path.exists(PROGRESO):
        return set()
    with open(PROGRESO, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f.readlines())

def guardar_progreso(nombre):
    with open(PROGRESO, "a", encoding="utf-8") as f:
        f.write(nombre + "\n")

def ejecutar_comando(cmd_list, cwd=None):
    print("    > Ejecutando:", " ".join(cmd_list))
    try:
        proc = subprocess.run(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            text=True,
            encoding="utf-8",
            errors="replace"
        )

        if proc.stdout:
            print("    > STDOUT:\n", proc.stdout.strip())
        if proc.stderr:
            print("    > STDERR:\n", proc.stderr.strip())

        if proc.returncode != 0:
            print("\n❌ ERROR FATAL en comando:", " ".join(cmd_list))
            print(proc.stderr)
            sys.exit(proc.returncode)

        return proc.returncode

    except Exception as e:
        print("\n🔥 Error al ejecutar comando:")
        print(e)
        traceback.print_exc()
        sys.exit(1)

print("\n========================================")
print("INICIO Script Maestro con Control de Progreso")
print(time.strftime("%Y-%m-%d %H:%M:%S"))
print("========================================\n")

if not os.path.isdir(BASE):
    print(f"ERROR: no existe carpeta base: {BASE}")
    sys.exit(1)

progreso = cargar_progreso()

for script in python_scripts:

    # SALTAR scripts ya ejecutados
    if script in progreso:
        print(f"✔ Script ya ejecutado anteriormente, se omite: {script}")
        continue

    ruta_script = os.path.join(BASE, script)
    print(f"\n--- Ejecutando: {script} ---")

    if not os.path.exists(ruta_script):
        print(f"❌ ERROR: No existe {ruta_script}")
        sys.exit(1)

    ejecutar_comando([PYTHON_CMD, ruta_script], cwd=BASE)

    # MARCAR COMO COMPLETADO
    guardar_progreso(script)

    # Ejecutar los scripts R después del 3
    if script == "3_Tabla_UnidMuestreo.py":
        print("\n>>> Ejecutando scripts R asociados <<<")
        for rscript in r_scripts_after_3:

            if rscript in progreso:
                print(f"✔ Script R ya ejecutado, se omite: {rscript}")
                continue

            ruta_r = os.path.join(BASE, rscript)
            print(f"   → {rscript}")

            if not os.path.exists(ruta_r):
                print(f"❌ ERROR: Script R no encontrado {ruta_r}")
                sys.exit(1)

            ejecutar_comando([R_BIN, ruta_r], cwd=BASE)
            guardar_progreso(rscript)

        print(">>> Fin scripts R <<<")

print("\n========================================")
print("FIN sin errores ✓")
print(time.strftime("%Y-%m-%d %H:%M:%S"))
print("========================================\n")























