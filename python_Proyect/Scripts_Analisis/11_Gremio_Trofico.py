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
# --------------------------------------------------

# ==============================================
# 🧩 Análisis de gremios tróficos
# ==============================================

# Filtrar registros válidos
df_gremios = Registros.dropna(subset=['Gremio', 'COBERTURA', 'INDIVIDUOS']).copy()

# Agrupar por gremio y cobertura
gremio_cobertura = (
    df_gremios.groupby(['COBERTURA', 'Gremio'])['INDIVIDUOS']
    .sum()
    .reset_index()
)

# Calcular abundancia total por cobertura
total_por_cobertura = (
    gremio_cobertura.groupby('COBERTURA')['INDIVIDUOS']
    .sum()
    .reset_index()
    .rename(columns={'INDIVIDUOS': 'Total_individuos'})
)

# Unir y calcular abundancia relativa (%)
gremio_cobertura = gremio_cobertura.merge(total_por_cobertura, on='COBERTURA')
gremio_cobertura['Abund_relativa_%'] = (
    gremio_cobertura['INDIVIDUOS'] / gremio_cobertura['Total_individuos'] * 100
).round(2)

# Mostrar tabla resumen
print("\n📋 Abundancia relativa por gremio y cobertura:")
print(gremio_cobertura.sort_values(['COBERTURA', 'Abund_relativa_%'], ascending=[True, False]))

# Guardar tabla en Excel
output_path = os.path.join(output_folder, "Resumen_Gremios_Troficos.xlsx")
gremio_cobertura.to_excel(output_path, index=False)
print(f"\n✅ Archivo guardado en: {output_path}")

#---------------------------------- Reparar y formatear archivo de Resumen_Uso_Habitat -----------------------
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Alignment, Font, Border, Side
from openpyxl.utils import get_column_letter
import os

# --- Rutas ---
ruta_original = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\Resumen_Gremios_Troficos.xlsx"
ruta_limpia = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\Resumen_Gremios_Troficos.xlsx"

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




# 📊 Gráfico 1: barras apiladas por cobertura
# ===============================
fig, ax = plt.subplots(figsize=(10,6))

pivot_data = gremio_cobertura.pivot(index='COBERTURA', columns='Gremio', values='Abund_relativa_%').fillna(0)

pivot_data.plot(kind='bar', stacked=True, colormap='tab10', ax=ax)

ax.set_ylabel('Abundancia relativa (%)')
ax.set_title('Distribución de gremios tróficos por cobertura', weight='bold')
ax.legend(title='Gremio trófico', fontsize=8, title_fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()

# Guardar gráfico
grafico_path = os.path.join(output_folder, "Grafico_Gremios_Troficos.png")
plt.savefig(grafico_path, dpi=300)
plt.show()

print(f"✅ Gráfico guardado en: {grafico_path}")



import matplotlib.pyplot as plt
import numpy as np
import os





# ===============================
# 📊 Gráfico de torta con etiquetas internas ajustadas
# ===============================

# --- Calcular abundancia total por gremio ---
gremio_total = gremio_cobertura.groupby("Gremio")["Abund_relativa_%"].sum().reset_index()

# --- Ordenar de mayor a menor ---
gremio_total = gremio_total.sort_values(by="Abund_relativa_%", ascending=False)

# --- Generar colores automáticamente según número de gremios ---
num_gremios = len(gremio_total)
colors = plt.cm.tab20(np.linspace(0, 1, num_gremios))

# --- Crear figura y gráfico ---
fig, ax = plt.subplots(figsize=(9, 8))

# --- Crear gráfico de torta ---
wedges, texts = ax.pie(
    gremio_total["Abund_relativa_%"],
    startangle=90,
    colors=colors,
    wedgeprops={'edgecolor': 'white'}
)

# ===============================
# 🏷️ Etiquetas ajustadas dentro del gráfico
# ===============================
prev_positions = []  # almacenará posiciones previas para evitar solapamiento

for i, w in enumerate(wedges):
    porcentaje = gremio_total["Abund_relativa_%"].iloc[i]
    ang = (w.theta2 - w.theta1) / 2.0 + w.theta1
    x = np.cos(np.deg2rad(ang))
    y = np.sin(np.deg2rad(ang))

    # posición inicial del texto
    text_x, text_y = 0.6 * x, 0.6 * y

    # ajustar si se solapa con otro texto
    for (px, py) in prev_positions:
        if abs(text_y - py) < 0.08:  # margen mínimo vertical
            text_y += 0.1 if text_y > py else -0.1

    prev_positions.append((text_x, text_y))

    # --- Solo dibujar línea si la etiqueta quedó fuera del radio 0.75 ---
    if abs(text_x) > 0.75 or abs(text_y) > 0.75:
        ax.plot([0.8 * x, text_x], [0.8 * y, text_y], color='gray', lw=0.8)

    # texto del porcentaje dentro (o ligeramente fuera) del gráfico
    ax.text(
        text_x, text_y,
        f"{porcentaje:.1f}%",
        ha='center', va='center',
        fontsize=10, fontweight='bold', color='black'
    )

# --- Leyenda ---
ax.legend(
    wedges,
    gremio_total["Gremio"],
    title="Gremio trófico",
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    fontsize=12,
    title_fontsize=14,
    frameon=True
)

# --- Título y formato ---
ax.set_title("Distribución general de gremios tróficos", fontsize=14, fontweight='bold', pad=20)
ax.axis('equal')  # mantiene forma circular
plt.tight_layout()

# --- Guardar gráfico ---
grafico_torta_path = os.path.join(output_folder, "Grafico_Torta_Gremios_Troficos.png")
plt.savefig(grafico_torta_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Gráfico de torta guardado en: {grafico_torta_path}")




