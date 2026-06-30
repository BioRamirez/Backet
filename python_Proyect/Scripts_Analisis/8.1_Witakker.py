# ==========================================
#  ANÁLISIS DE DIVERSIDAD BETA – WHITTAKER
#  Autor: Juan C. Ramírez Gil
# ==========================================

import pandas as pd
import numpy as np
import os

# --------------------------------------------------
# 1️⃣ Cargar datos
# --------------------------------------------------
ruta = r"D:\Aeropuerto Aguachica\OSO_PARDO_2025\Muestreo\GDB\ANALISIS\MAMIFEROS\MAMIFEROS_OSO_PARDO_2025.xlsx"
df = pd.read_excel(ruta)

# --------------------------------------------------
# 2️⃣ Crear tabla de abundancia (Cobertura × Especie)
# --------------------------------------------------
tabla = (
    df.groupby(['COBERTURA', 'ESPECIE'])['INDIVIDUOS']
    .sum()
    .unstack(fill_value=0)
)

print("\nTabla de abundancia:")
print(tabla.head())

# --------------------------------------------------
# 3️⃣ Cálculo de componentes de Whittaker
# --------------------------------------------------

# Riqueza total (gamma)
gamma = (tabla.sum(axis=0) > 0).sum()

# Riqueza por cobertura (alpha)
alpha_por_cobertura = (tabla > 0).sum(axis=1)

# Alfa promedio
alpha_media = alpha_por_cobertura.mean()

# Índice de Whittaker
whittaker = (gamma / alpha_media) - 1

# --------------------------------------------------
# 4️⃣ Tabla resumen
# --------------------------------------------------
resultado = pd.DataFrame({
    "Riqueza_por_cobertura": alpha_por_cobertura,
})

resultado.loc["PROMEDIO"] = alpha_media

resumen = pd.DataFrame({
    "Gamma (riqueza total)": [gamma],
    "Alpha promedio": [alpha_media],
    "Whittaker (β)": [whittaker]
})

# --------------------------------------------------
# 5️⃣ Guardar resultados
# --------------------------------------------------
salida = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\9_Whittaker_Beta.xlsx"

with pd.ExcelWriter(salida, engine="openpyxl") as writer:
    tabla.to_excel(writer, sheet_name="Matriz_Abundancia")
    resultado.to_excel(writer, sheet_name="Alpha_por_cobertura")
    resumen.to_excel(writer, sheet_name="Whittaker")

print("\n✔ Análisis de Whittaker finalizado")
print("✔ Archivo guardado en:")
print(salida)

# --------------------------------------------------
# 6️⃣ Interpretación automática
# --------------------------------------------------
if whittaker < 1:
    interpretacion = "baja diferenciación entre coberturas (comunidades similares)."
elif 1 <= whittaker <= 2:
    interpretacion = "recambio moderado de especies entre coberturas."
else:
    interpretacion = "alto recambio de especies y fuerte heterogeneidad espacial."

print("\nInterpretación ecológica:")
print(f"El índice de Whittaker fue {whittaker:.2f}, lo que indica {interpretacion}")




























































# ==========================================
#  WHITTAKER ENTRE COBERTURAS
#  Autor: Juan C. Ramírez Gil
# ==========================================

import pandas as pd
import os

# --------------------------------------------------
# 1️⃣ Cargar datos
# --------------------------------------------------
ruta = r"D:\Aeropuerto Aguachica\OSO_PARDO_2025\Muestreo\GDB\ANALISIS\ANFIBIOS\ANFIBIOS_OSO_PARDO_2025.xlsx"
df = pd.read_excel(ruta)

# --------------------------------------------------
# 2️⃣ Matriz Cobertura × Especie
# --------------------------------------------------
tabla = (
    df.groupby(['COBERTURA', 'ESPECIE'])['INDIVIDUOS']
    .sum()
    .unstack(fill_value=0)
)

# --------------------------------------------------
# 3️⃣ Cálculo de Whittaker
# --------------------------------------------------

# Gamma = riqueza total del sistema
gamma = (tabla.sum(axis=0) > 0).sum()

# Alfa = riqueza por cobertura
alpha_por_cobertura = (tabla > 0).sum(axis=1)

# Alfa promedio
alpha_media = alpha_por_cobertura.mean()

# Whittaker
beta_whittaker = (gamma / alpha_media) - 1

# --------------------------------------------------
# 4️⃣ Resultados
# --------------------------------------------------
tabla_alpha = pd.DataFrame({
    "Riqueza_por_cobertura": alpha_por_cobertura
})

resumen = pd.DataFrame({
    "Gamma_total": [gamma],
    "Alpha_promedio": [alpha_media],
    "Whittaker_beta": [beta_whittaker]
})

# --------------------------------------------------
# 5️⃣ Exportar resultados
# --------------------------------------------------
salida = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\9_Whittaker_Coberturas.xlsx"

with pd.ExcelWriter(salida, engine="openpyxl") as writer:
    tabla.to_excel(writer, sheet_name="Matriz_Abundancia")
    tabla_alpha.to_excel(writer, sheet_name="Alpha_por_cobertura")
    resumen.to_excel(writer, sheet_name="Resumen_Whittaker")

print("✔ Análisis de Whittaker completado")
print("✔ Archivo guardado en:")
print(salida)

# --------------------------------------------------
# 6️⃣ Interpretación automática
# --------------------------------------------------
if beta_whittaker < 1:
    interpretacion = "bajo recambio de especies entre coberturas"
elif beta_whittaker <= 2:
    interpretacion = "recambio moderado de especies entre coberturas"
else:
    interpretacion = "alto recambio de especies entre coberturas"

print("\nInterpretación ecológica:")
print(f"Whittaker = {beta_whittaker:.2f} → {interpretacion}")





import matplotlib.pyplot as plt

# -----------------------------------------
# GRÁFICO DE BARRAS – RIQUEZA POR COBERTURA
# -----------------------------------------

plt.figure(figsize=(8, 5))

alpha_por_cobertura.sort_values().plot(
    kind='bar',
    edgecolor='black'
)

plt.title("Riqueza de especies por cobertura\n(Componente del recambio β de Whittaker)")
plt.xlabel("Cobertura")
plt.ylabel("Número de especies")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()

# Guardar figura
ruta_figura = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\Whittaker_Riqueza_por_Cobertura.png"
plt.savefig(ruta_figura, dpi=300)
plt.show()

print("Gráfico guardado en:")
print(ruta_figura)
