#--------------## Cargar librerias necesarias------------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# --------------------------------------------------
# 2️⃣ Funciones para abreviar nombres de coberturas
# --------------------------------------------------

def generar_abreviacion(nombre):
    """
    Genera abreviaciones automáticas a partir de nombres de coberturas.
    Ejemplo: 'Bosque de galería y ripario' → 'Bgr'
    """
    # Convertir a minúsculas y dividir en palabras
    palabras = nombre.lower().split()

    # Eliminar conectores comunes
    palabras = [p for p in palabras if p not in ['de', 'del', 'la', 'el', 'y', 'con', 'en', 'los', 'las']]

    # Tomar la primera letra de cada palabra
    abreviacion = ''.join([p[0] for p in palabras])

    # Asegurar que tenga al menos 3 caracteres
    if len(abreviacion) < 3:
        abreviacion = abreviacion.ljust(3, '_')

    return abreviacion.capitalize()


def abreviar_coberturas(df, columna='COBERTURA'):
    """
    Crea un diccionario de abreviaciones y reemplaza los nombres en el DataFrame.
    """
    coberturas_unicas = df[columna].dropna().unique()
    abreviaciones = {c: generar_abreviacion(c) for c in coberturas_unicas}

    print("\n🔤 Abreviaciones generadas automáticamente:")
    for original, abrev in abreviaciones.items():
        print(f"  {original} → {abrev}")

    # Reemplazar en el DataFrame
    df[columna] = df[columna].replace(abreviaciones)

    return df, abreviaciones


# --- Aplicar las abreviaciones ---
Registros, abreviaciones_cobertura = abreviar_coberturas(Registros, columna='COBERTURA')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# --- Verificar columnas clave ---
print(Registros.columns)

Registros


# ============================================
# 📊 Análisis del uso de hábitat por cobertura
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Cargar los datos (ya los tienes en Registros)
# Asegurar que no haya valores nulos en columnas clave
df = Registros.dropna(subset=['ESPECIE', 'COBERTURA']).copy()

# ===============================
# 1️⃣ Riqueza total de especies por cobertura
# ===============================
riqueza_por_cobertura = (
    df.groupby('COBERTURA')['ESPECIE']
    .nunique()
    .reset_index()
    .rename(columns={'ESPECIE': 'Riqueza_total'})
)

# ===============================
# 2️⃣ Especies exclusivas por cobertura
# ===============================
# Calcular cuántas coberturas tiene cada especie
coberturas_por_especie = (
    df[['ESPECIE', 'COBERTURA']].drop_duplicates()
    .groupby('ESPECIE')['COBERTURA']
    .nunique()
    .reset_index()
    .rename(columns={'COBERTURA': 'Num_coberturas'})
)

# Filtrar especies presentes en una sola cobertura
especies_exclusivas = coberturas_por_especie.loc[
    coberturas_por_especie['Num_coberturas'] == 1, 'ESPECIE'
]

# Contar cuántas exclusivas hay por cobertura
exclusivas_por_cobertura = (
    df[df['ESPECIE'].isin(especies_exclusivas)]
    .groupby('COBERTURA')['ESPECIE']
    .nunique()
    .reset_index()
    .rename(columns={'ESPECIE': 'Especies_exclusivas'})
)

# ===============================
# 3️⃣ Combinar ambos resultados
# ===============================
uso_habitat = pd.merge(
    riqueza_por_cobertura,
    exclusivas_por_cobertura,
    on='COBERTURA',
    how='left'
).fillna(0)

# Calcular el porcentaje de especies exclusivas
uso_habitat['%_Exclusivas'] = (
    uso_habitat['Especies_exclusivas'] / uso_habitat['Riqueza_total'] * 100
).round(2)

# ===============================
# 4️⃣ Mostrar tabla resumen
# ===============================
print("\n📋 Resumen del uso de hábitat por cobertura:")
print(uso_habitat.sort_values(by='Riqueza_total', ascending=False))

# ===============================
# 💾 GUARDAR RESULTADOS
# ===============================

# 1️⃣ Guardar resumen como archivo Excel
resumen_path = os.path.join(output_folder, "Resumen_Uso_Habitat.xlsx")
uso_habitat.to_excel(resumen_path, index=False)
print(f"✅ Resumen guardado en: {resumen_path}")

#---------------------------------- Reparar y formatear archivo de Resumen_Uso_Habitat -----------------------
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Alignment, Font, Border, Side
from openpyxl.utils import get_column_letter
import os

# --- Rutas ---
ruta_original = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\Resumen_Uso_Habitat.xlsx"
ruta_limpia = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\Resumen_Uso_Habitat.xlsx"

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



# ===============================
# 5️⃣ Gráfico comparativo
# ===============================
plt.figure(figsize=(10,6))
sns.barplot(
    data=uso_habitat.sort_values('Riqueza_total', ascending=False),
    x='COBERTURA',
    y='Riqueza_total',
    color='skyblue',
    label='Riqueza total'
)
sns.barplot(
    data=uso_habitat.sort_values('Riqueza_total', ascending=False),
    x='COBERTURA',
    y='Especies_exclusivas',
    color='steelblue',
    label='Exclusivas'
)

plt.title('Uso de hábitat: riqueza y especies exclusivas por cobertura')
plt.ylabel('Número de especies')
plt.xlabel('Cobertura')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()



#----------------------------------------------
# 6️⃣ Gráfico de barras apiladas


import matplotlib.pyplot as plt
import numpy as np

# --- Datos ordenados ---
data_plot = uso_habitat.sort_values('Riqueza_total', ascending=False)
x = np.arange(len(data_plot))
width = 0.6

# --- Colores ---
color_excl = '#E76F51'  # naranja
color_comp = '#457B9D'  # azul

# --- Calcular compartidas ---
data_plot['Especies_compartidas'] = data_plot['Riqueza_total'] - data_plot['Especies_exclusivas']

# --- Crear barras apiladas ---
plt.figure(figsize=(10,6))
plt.bar(x, data_plot['Especies_exclusivas'], color=color_excl, label='Exclusivas')
plt.bar(x, data_plot['Especies_compartidas'], 
        bottom=data_plot['Especies_exclusivas'], color=color_comp, label='Compartidas')

# --- Etiquetas ---
for i, row in data_plot.iterrows():
    exclusivas = row['Especies_exclusivas']
    compartidas = row['Especies_compartidas']
    total = row['Riqueza_total']
    
    # Exclusivas (centro de la barra naranja)
    plt.text(i, exclusivas/2, f"{int(exclusivas)}", ha='center', va='center', color='white', fontsize=9)
    
    # Compartidas (centro de la barra azul)
    plt.text(i, exclusivas + compartidas/2, f"{int(compartidas)}", ha='center', va='center', color='white', fontsize=9)
    
    # Total (encima)
    plt.text(i, total + 1, f"Total: {int(total)}", ha='center', va='bottom', fontsize=9, color='black')


# --- Personalización ---
plt.xticks(x, data_plot['COBERTURA'], rotation=45, ha='right')
plt.ylabel('Número de especies')
plt.xlabel('Cobertura')
plt.title('Uso de hábitat', fontsize=13, weight='bold')
plt.legend(
    title="Categorías",
    fontsize=11,       # 🔹 Tamaño del texto de la leyenda
    title_fontsize=12, # 🔹 Tamaño del título de la leyenda (opcional)
    loc='upper right'  # 🔹 Puedes moverla (ej: 'upper left', 'lower right', etc.)
)
plt.tight_layout()
plt.grid(
    True,           # activa la grilla
    axis='y',       # solo líneas horizontales (eje Y)
    linestyle='--', # tipo de línea (puede ser '-', '--', ':', '-.')
    alpha=0.4,      # transparencia
    zorder=0        # asegura que quede detrás de las barras
)


# 2️⃣ Guardar la figura apilada como imagen
fig_path = os.path.join(output_folder, "Grafico_Uso_Habitat_Apilado.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"✅ Gráfico guardado en: {fig_path}")

plt.show()