#-------------------------Dar formato al archivo Estimadores_Abundancia.xlsx-------------------------#
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Alignment, Font, Border, Side
from openpyxl.utils import get_column_letter
import os

# --- Rutas ---
ruta_original = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\Estimadores_Abundancia.xlsx"
ruta_limpia = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\Estimadores_Abundancia.xlsx"

# --- Verificar existencia ---
if not os.path.exists(ruta_original):
    raise FileNotFoundError(f"⚠️ No se encontró el archivo: {ruta_original}")

# --- Leer archivo dañado con pandas ---
try:
    df = pd.read_excel(ruta_original)
    print("✅ Archivo leído correctamente con pandas.")
except Exception as e:
    raise RuntimeError(f"❌ No se pudo leer el archivo: {e}")

# --- Reescribir el archivo limpio ---
df.to_excel(ruta_limpia, index=False)
print(f"🧹 Archivo reparado y guardado como:\n{ruta_limpia}")

# --- Aplicar formato con openpyxl ---
from openpyxl import load_workbook

wb = load_workbook(ruta_limpia)
ws = wb.active

# --- Estilos base ---
header_fill = PatternFill(start_color='BFD8B8', end_color='BFD8B8', fill_type='solid')
header_font = Font(bold=True, color='000000', name='Calibri')
center_align = Alignment(horizontal='center', vertical='center', wrap_text=True)
thin_border = Border(
    left=Side(style='thin', color='000000'),
    right=Side(style='thin', color='000000'),
    top=Side(style='thin', color='000000'),
    bottom=Side(style='thin', color='000000')
)

# --- Aplicar formato y reemplazar vacíos ---
for row in ws.iter_rows():
    for cell in row:
        if cell.value is None or str(cell.value).strip() == '':
            cell.value = '-'
        cell.alignment = center_align
        cell.border = thin_border

# --- Encabezado ---
for cell in ws[1]:
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = center_align

# --- Ajustar ancho de columnas ---
for col in ws.columns:
    max_length = 0
    column = get_column_letter(col[0].column)
    for cell in col:
        if cell.value:
            length = len(str(cell.value))
            if length > max_length:
                max_length = length
    ws.column_dimensions[column].width = max_length + 3

# --- Ajustar altura de filas ---
for row in ws.iter_rows():
    ws.row_dimensions[row[0].row].height = 18

# --- Guardar cambios ---
wb.save(ruta_limpia)
print(f'📘 Archivo formateado y reparado correctamente:\n{ruta_limpia}')

#-------------------------Fin Dar formato al archivo Estimadores_Abundancia.xlsx-------------------------#

#-----------------------leer el archivo formateado-----------------------#
import pandas as pd
tabla_Abund = pd.read_excel(r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\Estimadores_Abundancia.xlsx")

tabla_Abund

names = tabla_Abund.columns.tolist()
names


#-----------------------Calcular efectividad de los estimadores de abundancia-----------------------#
import pandas as pd

# --- Cargar datos ---
ruta = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\Estimadores_Abundancia.xlsx"
tabla_Abund = pd.read_excel(ruta)

# --- Calcular efectividad para cada estimador ---
estimadores = ['Chao1_Mean', 'Chao1_Chao_1984_Mean', 'Chao1_bc_Mean', 'iChao1_Chiu_et_al_2014_Mean',
               'ACE_Chao_Lee_1992_Mean', 'ACE_1_Chao_Lee_1992_Mean', '1st_order_jackknife_Mean', '2nd_order_jackknife_Mean', 
               'Bootstrap_Mean',]  # ajusta según tus columnas reales

efectividad = pd.DataFrame()
efectividad['Unidad'] = tabla_Abund['Unidad']
efectividad['Observadas_Mean'] = tabla_Abund['Observadas_Mean']

for est in estimadores:
    if est in tabla_Abund.columns:
        efectividad[est.replace('_Mean', '_Efectividad_%')] = (
            (tabla_Abund['Observadas_Mean'] / tabla_Abund[est]) * 100
        )

# --- Calcular promedio total de efectividad por estimador ---
resumen = (
    efectividad.drop(columns=['Unidad', 'Observadas_Mean'])
    .mean()
    .reset_index()
    .rename(columns={'index': 'Estimador', 0: 'Efectividad_Promedio_%'})
)

resumen = resumen.sort_values(by='Efectividad_Promedio_%', ascending=False)
resumen

#-----------------------Fin Calcular efectividad de los estimadores de abundancia-----------------------#
#-----------------------Guardar tabla de efectividad-----------------------#

ruta_salida = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\Efectividad_Estimadores.xlsx"
with pd.ExcelWriter(ruta_salida, engine='openpyxl') as writer:
    efectividad.to_excel(writer, sheet_name='Por_Unidad', index=False)
    resumen.to_excel(writer, sheet_name='Resumen_Efectividad', index=False)

print("✅ Tabla de efectividad exportada correctamente.")

#-----------------------Fin Guardar tabla de efectividad-----------------------#
#-----------------------Graficar curvas de acumulacion de especies-----------------------#


import matplotlib.pyplot as plt
import numpy as np

# --- Escoger los dos mejores estimadores ---
top2 = resumen['Estimador'].head(3).str.replace('_Efectividad_%', '_Mean').tolist()
print("📊 Mejores estimadores:", top2)

# --- Crear figura ---
fig, ax = plt.subplots(figsize=(10, 6))

# Eje X dinámico según número de unidades
x = np.arange(1, len(tabla_Abund) + 1)

# --- Función automática de etiquetado sin solapamientos ---
etiquetas_previas = []

def colocar_etiqueta_automatica(x_val, y_val, texto, ax):
    """Ubica etiquetas sin solaparse y dentro de los límites del gráfico."""
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * 0.05  # desplazamiento dinámico: 3% del rango

    for y_prev in etiquetas_previas:
        if abs(y_prev - y_val) < offset:
            y_val += offset
    etiquetas_previas.append(y_val)

    # Limitar dentro del eje
    y_val = np.clip(y_val, ymin + offset, ymax - offset)
    x_val = min(x_val, ax.get_xlim()[1] - 0.5)

    # Texto sin fondo, ligeramente a la derecha del punto
    ax.text(x_val + 0.2, y_val, f"{float(texto):.1f}",
            fontsize=9, ha='left', va='center', color='black')


# --- Dibujar observadas ---
ax.plot(x, tabla_Abund['Observadas_Mean'], 'o-', color='black', label='Observadas')
colocar_etiqueta_automatica(x[-1], tabla_Abund['Observadas_Mean'].iloc[-1],
                            tabla_Abund['Observadas_Mean'].iloc[-1], ax)

# --- Dibujar los dos mejores estimadores ---
for est in top2:
    ax.plot(x, tabla_Abund[est], 'o--', label=est.replace('_Mean', ''))
    colocar_etiqueta_automatica(x[-1], tabla_Abund[est].iloc[-1],
                                tabla_Abund[est].iloc[-1], ax)

# --- Dibujar desviación estándar de Singletons ---
if 'Singletons_SD' in tabla_Abund.columns:
    ax.plot(x, tabla_Abund['Singletons_SD'], 'o--', color='gray', linewidth=2,
            label='Singletons_SD')
    colocar_etiqueta_automatica(x[-1], tabla_Abund['Singletons_SD'].iloc[-1],
                                tabla_Abund['Singletons_SD'].iloc[-1], ax)

# --- Ajustes automáticos del gráfico ---
ax.set_xlim(0.5, len(x) + 0.8)  # deja espacio a la derecha para etiquetas
ax.margins(y=0.1)               # ajusta el alto para evitar cortes
plt.title("Curva de acumulación de especies")
plt.xlabel("Unidades de muestreo")
plt.ylabel("Riqueza estimada")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# --- Guardar la gráfica en formato PNG ---
fig.savefig("D:/CORPONOR 2025/Backet/python_Proyect/Resultados/estimadores_riqueza.png",
            dpi=300, bbox_inches='tight', transparent=False)
print("✅ Gráfica guardada correctamente.")

#-----------------------Fin Graficar curvas de acumulacion de especies-----------------------#

#-----------------------Crear tabla resumen de estimadores-----------------------#
import pandas as pd

# --- Crear tabla resumen de efectividad ---
datos_tabla = []

# Valor observado final
obs_final = tabla_Abund['Observadas_Mean'].iloc[-1]
datos_tabla.append({
    "Estimador": "Observadas",
    "Individuos_estimados": obs_final,
    "Efectividad_%": None  # sin porcentaje
})

# Los estimadores del gráfico (automático según top2)
efectividades = []
for est in top2:
    valor_final = tabla_Abund[est].iloc[-1]
    # 🔹 Porcentaje de representatividad del observado respecto al estimado
    efectividad = (obs_final / valor_final) * 100
    efectividades.append(efectividad)
    datos_tabla.append({
        "Estimador": est.replace('_Mean', ''),
        "Individuos_estimados": valor_final,
        "Efectividad_%": efectividad
    })

# Agregar fila de promedio de efectividad
promedio_efectividad = sum(efectividades) / len(efectividades)
datos_tabla.append({
    "Estimador": "Promedio efectividad",
    "Individuos_estimados": None,
    "Efectividad_%": promedio_efectividad
})

# Convertir a DataFrame
tabla_resumen = pd.DataFrame(datos_tabla)

# --- Mostrar con formato redondeado ---
print("\n📋 Resumen de estimadores (valores finales):")
print(tabla_resumen.round(2).to_string(index=False))

# --- Guardar en Excel ---
ruta_salida = "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/Resumen_estimadores.xlsx"
tabla_resumen.to_excel(ruta_salida, index=False)

print(f"\n✅ Archivo Excel guardado en:\n{ruta_salida}")
