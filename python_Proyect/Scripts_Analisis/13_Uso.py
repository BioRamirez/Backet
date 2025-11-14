#--------------## Cargar librerias necesarias------------------------------

# Si no las tienes instaladas, ejecuta esta celda una vez:
# Salir del interprete con: exit() exit() python   pip install tabulate pandas numpy scipy scikit-bio openpyxl
#
# !pip install pandas numpy matplotlib tabulate openpyxl

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import openpyxl


# Carpeta donde guardarás los gráficos (solo una vez)
output_folder = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados"
os.makedirs(output_folder, exist_ok=True)

#--------------## Leer archivo y revisar columnas------------------------------
# Ruta del archivo
ruta = r"D:\CORPONOR 2025\Backet\python_Proyect\data\POF_ZULIA_2025_BD_AVES_MAMIFEROS.xlsx"

# Leer el archivo Excel
Registros = pd.read_excel(ruta)

# Mostrar las primeras filas
print("📄 Primeras filas del archivo:")
print(Registros.head())

# Mostrar nombres de las columnas
print("\n📋 Columnas del DataFrame:")
print(Registros.columns)

Registros['Uso'].unique()


# Copia de trabajo
df = Registros.copy()

# Normalizar texto
df['Uso'] = df['Uso'].astype(str).str.strip().str.lower()

# Unificación de categorías
df['Uso'] = df['Uso'].replace({
    'sin uso conocido': 'sin_uso',
    'uso cultural': 'cultural',
    'cultural': 'cultural',
    'medicinal': 'medicinal',
    'mascota': 'mascotas',
    'mascotas': 'mascotas',
    'subsistencia': 'subsistencia',
    'otro': 'otro'
})


# Quitar espacios y separar por coma
df['Uso'] = df['Uso'].str.replace(' ', '').str.split(',')

# Expandir usos múltiples a filas individuales
df_uso = df.explode('Uso')


tabla_final = (
    df_uso[['CLASE', 'Orden', 'Familia', 'Especie', 'Uso']]
    .dropna()
    .drop_duplicates()
    .sort_values(['Orden', 'Familia', 'Especie'])
    .reset_index(drop=True)
)

tabla_final.rename(columns={'Uso': 'Uso'}, inplace=True)

print(tabla_final)


# 1. Copia del dataframe final
df2 = tabla_final.copy()

# 2. Renombrar la columna a "Clase"
df2.rename(columns={'CLASE': 'Clase'}, inplace=True)

# 3. Eliminar especies SIN USO
df2 = df2[df2['Uso'] != 'sin_uso']

# 4. Ordenar primero Aves luego Mamíferos
df2['Clase'] = df2['Clase'].str.upper()

orden_clases = {'AVES': 1, 'MAMMALIA': 2}
df2['Clase_orden'] = df2['Clase'].map(orden_clases)

df2 = df2.sort_values(['Clase_orden', 'Orden', 'Familia', 'Especie'])

# 5. Reiniciar índices por clase y agregar columna "N°"
df2['N°'] = df2.groupby('Clase').cumcount() + 1

# 6. Ordenar columnas como solicitaste
tabla_ordenada = df2[['N°', 'Clase', 'Orden', 'Familia', 'Especie', 'Uso']]

# 7. Eliminar la columna auxiliar
tabla_ordenada = tabla_ordenada.reset_index(drop=True)

print(tabla_ordenada)

# --- 1. Definir el orden deseado para 'Uso' ---
orden_uso = ['cultural', 'medicinal', 'mascotas', 'subsistencia', 'otro', 'sin_uso']

tabla_ordenada['Uso'] = pd.Categorical(tabla_ordenada['Uso'], categories=orden_uso, ordered=True)

# --- 2. Ordenar por Clase y luego por Uso ---
tabla_ordenada = tabla_ordenada.sort_values(
    by=['Clase', 'Uso', 'Orden', 'Familia', 'Especie']
).reset_index(drop=True)

# --- 3. Volver a crear N° independiente por Clase ---
tabla_ordenada['N°'] = tabla_ordenada.groupby('Clase').cumcount() + 1

# --- 4. Reordenar columnas para que N° quede primero ---
tabla_ordenada = tabla_ordenada[['N°', 'Clase', 'Orden', 'Familia', 'Especie', 'Uso']]

tabla_ordenada['Clase'] = tabla_ordenada['Clase'].str.capitalize()
tabla_ordenada['Uso'] = tabla_ordenada['Uso'].str.capitalize()


print(tabla_ordenada)

# Guardar el resultado en un nuevo archivo Excel
output_folder = os.path.join(output_folder, "tabla_Uso_Cultural.xlsx")
tabla_ordenada.to_excel(output_folder, index=False)
print(f"\n✅ Archivo guardado en: {output_folder}")
#--------------## Fin del análisis------------------------------
#----------------------------------------------------------------------
#---------------------------------- Reparar y formatear archivo de tabla_Uso_Cultural -----------------------
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Alignment, Font, Border, Side
from openpyxl.utils import get_column_letter
import os

# --- Rutas ---
ruta_original = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\tabla_Uso_Cultural.xlsx"
ruta_limpia = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\tabla_Uso_Cultural.xlsx"

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
#----------------------------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd

# --- 1. Filtrar usos válidos ---
usos_validos = ['Cultural', 'Medicinal', 'Mascotas', 'Subsistencia', 'Otro']
df_plot = tabla_ordenada[tabla_ordenada['Uso'].isin(usos_validos)]

# --- 2. Contar especies por Clase y Uso ---
conteos = df_plot.groupby(['Clase', 'Uso'])['Especie'].nunique().reset_index()

# --- 3. Convertir en tabla pivote para graficar ---
pivot = conteos.pivot(index='Clase', columns='Uso', values='Especie').fillna(0)

# --- 4. Crear gráfico ---
ax = pivot.plot(kind='bar', figsize=(12, 7))

plt.title("Tipos de Uso por Especies", fontsize=14)
plt.xlabel("Clase", fontsize=12)
plt.ylabel("Número de especies", fontsize=12)
plt.xticks(rotation=0)
plt.legend(title="Uso", fontsize=10)

# --- 5. Agregar etiquetas de datos ---
for container in ax.containers:
    ax.bar_label(container, fmt='%d', padding=3)

plt.tight_layout()

# --- 6. Guardar el gráfico en PNG ---
output_path = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\Grafico_Uso_Cultural.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')


plt.show()

print(f"Gráfico guardado en: {output_path}")

