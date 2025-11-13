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

#------------------------Tabla general grupo taxonomico-------------------------

# Mostrar las primeras filas
print(Registros.info())


import pandas as pd

# --- Copiar el DataFrame base ---
df = Registros.copy()

# --- Normalizar texto ---
for col in ['CLASE', 'Orden', 'Familia', 'Genero', 'Epiteto', 'N. comun', 'Gremio', 'COBERTURA', 'METODOLOGIA']:
    df[col] = df[col].astype(str).str.strip().str.title()

# --- Crear nombre científico completo ---
df['Especie_cientifica'] = df['Genero'] + ' ' + df['Epiteto']

# --- Diccionario de abreviaciones de Metodologia ---
abreviaciones_metodo = {
    'Auditivo': 'Aud',
    'Fotografia': 'Fot',
    'Fotografia ': 'Fot',
    'Marcas De Presencia': 'MP',
    'Avistamiento': 'Obs',
    'Observacion': 'Obs',
    'Entrevista': 'Ent',
    'Captura': 'Cap',
    'Rastros': 'Ras',
    'Huellas': 'Hue',
    'Cueva': 'Cuv',
    'Heces': 'Hec',
    'Video': 'Vid',
    'Informacion Mcnup': 'MCNUP'
}

# --- Diccionario de abreviaciones de cobertura ---
abreviaciones_cobertura = {
    'Bosque De Galería Y Ripario': 'Bgr',
    'Bosque Denso Alto De Tierra Firme': 'Bda',
    'Bosque Denso Bajo De Tierra Firme': 'Bdb',
    'Bosque Fragmentado Con Vegetación Secundaria': 'Bfvs'
}

# --- Diccionario de abreviaciones de gremio ---
abreviaciones_gremio = {
    'Carnívoro': 'Car',
    'Nectarívoro': 'Nec',
    'Carroñero': 'Crr',
    'Granívoro': 'Gra',
    'Frugívoro': 'Fru',
    'Insectívoro': 'Ins',
    'Omnívoro': 'Omn',
    'Herbívoro': 'Her',
    'Herbivoro': 'Her',
    'Nan': 'NA'
}

# --- Reemplazar nombres por abreviaciones ---
df['METODOLOGIA'] = df['METODOLOGIA'].replace(abreviaciones_metodo)
df['COBERTURA'] = df['COBERTURA'].replace(abreviaciones_cobertura)
df['Gremio'] = df['Gremio'].replace(abreviaciones_gremio)


# Mostrar nombres de las columnas
print("\n📋 Columnas del DataFrame:")
print(df.columns)

# ==========================
# 📊 TABLA DE SENSIBILIDAD
# ==========================

# --- Seleccionar columnas relevantes ---
tabla_sensibilidad = df[[
    'CLASE',
    'Familia',
    'Especie_cientifica',
    'N. comun',
    'IUCN',
    'MADS (Resol 0126)',
    'CITES',
    'Dist_Geo',
    'Tipo_Migra'
]].copy()

# --- Renombrar columnas ---
tabla_sensibilidad = tabla_sensibilidad.rename(columns={
    'Especie_cientifica': 'Especie',
    'CLASE': 'Clase',
    'N. comun': 'N. común',
    'MADS (Resol 0126)': 'Res. 0126',
    'Dist_Geo': 'Distribución',
    'Tipo_Migra': 'Migración'
})

# --- Normalizar texto ---
for col in ['Clase','Familia', 'Especie', 'N. común', 'Distribución', 'Migración']:
    tabla_sensibilidad[col] = tabla_sensibilidad[col].astype(str).str.strip().str.title()

# --- Eliminar duplicados (una fila por especie) ---
tabla_sensibilidad = tabla_sensibilidad.drop_duplicates(subset=['Especie']).reset_index(drop=True)

# --- Mostrar tabla final ---
print(tabulate(tabla_sensibilidad.head(10), headers='keys', tablefmt='github', showindex=False))

# Mostrar nombres de las columnas
print("\n📋 Columnas del DataFrame:")
print(tabla_sensibilidad.columns)

# --- Revisar valores únicos en las columnas de interés ---
print("🔎 Valores únicos en la columna 'IUCN':")
print(df['IUCN'].dropna().unique())

print("\n🔎 Valores únicos en la columna 'MADS (Resol 0126)':")
print(df['MADS (Resol 0126)'].dropna().unique())

# --- Diccionario de abreviaciones IUCN ---
abreviaciones_iucn = {
    'Preocupación Menor (LC)': 'LC',
    'Preocupacin Menor (LC)': 'LC',  # error ortográfico corregido
    'Casi Amenazado (NT)': 'NT',
    'Vulnerable (VU)': 'VU',
    'En Peligro (EN)': 'EN',
    'En Peligro Crítico (CR)': 'CR',
    'Extinto En Estado Silvestre (EW)': 'EW',
    'Extinto (EX)': 'EX',
    'Datos Insuficientes (DD)': 'DD',
    'No Evaluado (NE)': 'NE'
}

# --- Diccionario de abreviaciones Resolución 0126 (MADS) ---
abreviaciones_mads = {
    'Preocupación Menor (LC)': 'LC',
    'Casi Amenazado (NT)': 'NT',
    'Vulnerable (VU)': 'VU',
    'En Peligro (EN)': 'EN',
    'En Peligro Crítico (CR)': 'CR',
    'Extinto En Estado Silvestre (EW)': 'EW',
    'Extinto (EX)': 'EX',
    'No Listada': 'NL',
    'NL': 'NL',
    'No aplica': 'NA'
}
# --- Aplicar abreviaciones ---


tabla_sensibilidad['IUCN'] = tabla_sensibilidad['IUCN'].replace(abreviaciones_iucn)
tabla_sensibilidad['Res. 0126'] = tabla_sensibilidad['Res. 0126'].replace(abreviaciones_mads)

# Mostrar nombres de las columnas
print("\n📋 Columnas del DataFrame:")
print(tabla_sensibilidad.columns)

# ====================================================
# 🧩 Filtrar especies sensibles sin modificar formato original
# ====================================================
# ---------- Filtrado robusto sin alterar formato original ----------
import re

# trabajar sobre copia
temp = tabla_sensibilidad.copy()

# columnas de interés
cols = ['IUCN', 'Res. 0126', 'CITES', 'Distribución', 'Migración']

# preparar columna normalizada para cada campo (solo para comparar)
def norm_iucn(v):
    if pd.isna(v): return ''
    s = str(v).strip().lower()
    # arreglar errores comunes
    s = s.replace('preocupacin', 'preocupación')
    s = re.sub(r'[^\w\(\) ]', '', s)  # quitar puntuación inusual salvo paréntesis
    # mapear variantes a códigos
    if s in ('lc', 'preocupación menor lc', 'preocupación menor (lc)', 'preocupación menor'):
        return 'lc'
    if s in ('nt', 'casi amenazado (nt)', 'casi amenazado'):
        return 'nt'
    if s in ('vu', 'vulnerable (vu)', 'vulnerable'):
        return 'vu'
    if s in ('en', 'en peligro (en)', 'en peligro'):
        return 'en'
    if s in ('cr', 'en peligro crítico (cr)', 'en peligro crítico'):
        return 'cr'
    if s in ('ew', 'extinto en estado silvestre (ew)', 'extinto en estado silvestre'):
        return 'ew'
    if s in ('ex', 'extinto (ex)', 'extinto'):
        return 'ex'
    if s in ('dd', 'datos insuficientes (dd)', 'datos insuficientes'):
        return 'dd'
    if s in ('ne', 'no evaluado (ne)', 'no evaluado', 'no evaluada (ne)', 'no evaluada'):
        return 'ne'
    # Si ya es la frase larga
    return s

def norm_res126(v):
    if pd.isna(v): return ''
    s = str(v).strip().lower()
    s = s.replace('preocupacin', 'preocupación')
    s = re.sub(r'[^\w\(\) ]', '', s)
    if s in ('nl', 'no listada', 'no listada (nl)'):
        return 'nl'
    if s in ('no aplica', 'na', ''):
        return ''
    # mapear igual que iucn si vienen clasificaciones iguales
    mapped = norm_iucn(s)
    return mapped if mapped in ('lc','nt','vu','en','cr','ew','ex','dd','ne') else s

def norm_cites(v):
    if pd.isna(v): return ''
    s = str(v).strip().lower()
    s = s.replace('á','a')
    s = s.replace('apendice', 'apendice')  # mantener ortografía
    # mapear apéndices
    if 'apendice i' in s or 'apendice i' == s or 'apendicei'==s:
        return 'apendice i'
    if 'apendice ii' in s or 'apendice ii' == s or 'apendiceii'==s:
        return 'apendice ii'
    if 'apendice iii' in s or 'apendice iii' == s or 'apendiceiii'==s:
        return 'apendice iii'
    if 'no aplica' in s or s in ('', 'na'):
        return ''
    return s

def norm_dist(v):
    if pd.isna(v): return ''
    s = str(v).strip().lower()
    # considerar listas/comas: si contiene 'neotropical' marcar como 'neotropical'
    if 'neotropical' in s:
        return 'neotropical'
    if 'cosmopolita' in s:
        return 'cosmopolita'
    if 'nearctica' in s or 'nearctic' in s:
        return 'nearctica'
    return s

def norm_mig(v):
    if pd.isna(v): return ''
    s = str(v).strip().lower()
    s = s.replace('.', '')
    if s in ('res', 'residente', 'resident'):
        return 'res'
    if 'lat' in s or 'lat-trans' in s or 'lat trans' in s:
        return 'lat-trans'
    if 'nomad' in s:
        return 'nomad'
    if 'residente' in s:
        return 'res'
    return s

# crear columnas normalizadas
temp['_iucn_norm'] = temp['IUCN'].apply(norm_iucn)
temp['_res126_norm'] = temp['Res. 0126'].apply(norm_res126)
temp['_cites_norm'] = temp['CITES'].apply(norm_cites)
temp['_dist_norm'] = temp['Distribución'].apply(norm_dist)
temp['_mig_norm'] = temp['Migración'].apply(norm_mig)

# ahora definir neutros usando los códigos canónicos (todo en minúscula)
iucn_neutro = {'lc', 'ne', ''}            # LC y NO EVALUADO consideramos neutro
res126_neutro = {'lc', '', 'nl'}                # NL o vacío = neutro
cites_neutro = {'', 'no aplica'}                       # vacío = sin CITES
dist_neutro = {'neotropical', 'cosmopolita','nomadismo', '','nearctica, neotropical'}         # neotropical se considera neutro
mig_neutro = {'res', ''}                  # residente se considera neutro

condicion_neutra = (
    temp['_iucn_norm'].isin(iucn_neutro) &
    temp['_res126_norm'].isin(res126_neutro) &
    temp['_cites_norm'].isin(cites_neutro) &
    temp['_dist_norm'].isin(dist_neutro) &
    temp['_mig_norm'].isin(mig_neutro)
)

# Filtrar sobre la tabla original (sin modificar su formato)
tabla_sensible = temp.loc[~condicion_neutra, tabla_sensibilidad.columns].copy()

print(f"✅ Especies sensibles detectadas: {len(tabla_sensible)} de {len(tabla_sensibilidad)}")
print(tabla_sensible.head(20))


# --- 🔹 Ordenar clases: primero Aves, luego Mammalia, luego el resto ---
orden_clase = ['Aves', 'Mammalia']
tabla_sensible['Clase'] = pd.Categorical(
    tabla_sensible['Clase'],
    categories=orden_clase + sorted(set(tabla_sensible['Clase']) - set(orden_clase)),
    ordered=True
)

# --- 🔹 Ordenar la tabla por Clase, Orden, Familia y Especie ---
tabla_sensible = tabla_sensible.sort_values(['Clase', 'Familia', 'Especie']).reset_index(drop=True)

# --- 🔹 Agregar numeración reiniciada por Clase ---
tabla_sensible['N°'] = tabla_sensible.groupby('Clase').cumcount() + 1

# Mover la columna "N°" al inicio
cols = ['N°'] + [col for col in tabla_sensible.columns if col != 'N°']
tabla_sensible = tabla_sensible[cols]

# --- 🔹 Insertar fila con nombres de columnas justo antes de 'Mammalia' ---
idx_mam = tabla_sensible.index[tabla_sensible['Clase'] == 'Mammalia']
if len(idx_mam) > 0:
    insert_pos = idx_mam[0]
    fila_header = pd.DataFrame([{col: str(col) for col in tabla_sensible.columns}])  # mantiene los nombres de columnas como texto
    tabla_sensible = pd.concat(
        [tabla_sensible.iloc[:insert_pos], fila_header, tabla_sensible.iloc[insert_pos:]],
        ignore_index=True
    )

# --- 🔹 Mostrar resultado ---
tabla_sensible.head(20)



# Guardar tabla en Excel
output_path = os.path.join(output_folder, "tabla_sensibilidad.xlsx")
tabla_sensible.to_excel(output_path, index=False)
print(f"\n✅ Archivo guardado en: {output_path}")


#---------------------------------- Reparar y formatear archivo de tabla_sensibilidad -----------------------
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Alignment, Font, Border, Side
from openpyxl.utils import get_column_letter
import os

# --- Rutas ---
ruta_original = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\tabla_sensibilidad.xlsx"
ruta_limpia = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\tabla_sensibilidad.xlsx"

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




