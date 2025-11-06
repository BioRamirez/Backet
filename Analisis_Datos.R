#------------------------------------------Trabajar en python-----------------------------


library(reticulate)

# Crea un entorno virtual permanente llamado "r-reticulate"
virtualenv_create("C:/Users/Ramirez Juan/.python_envs/r-reticulate")

# Usar este entorno como el principal de reticulate
use_virtualenv("C:/Users/Ramirez Juan/.python_envs/r-reticulate", required = TRUE)

# Instala los paquetes básicos de ciencia de datos
py_install(c("pandas", "numpy", "matplotlib", "tabulate", "openpyxl"))




library(reticulate)

# Instalar pandas y openpyxl si aún no los tienes
py_install(c("pandas", "openpyxl"))

# Ejecutar código Python desde R
py_run_string("
import pandas as pd

# Leer el archivo Excel
Registros = pd.read_excel('D:/CORPONOR 2025/FOTOS/POF_ZULIA_2025_BD_AVES_MAMIFEROS.xlsx')

# Mostrar las primeras filas
print(Registros.head())
")

# También puedes traer el DataFrame a R si lo necesitas:
Registros <- py$Registros
View(Registros)

#--------------## Esfuerzo de Muestreo------------------------------

py_run_string("# Mostrar las primeras filas
print(Registros.head())
")

py_run_string("print(Registros.columns)")


py_run_string("print(Registros['METODOLOGIA'].unique())")

py_run_string("print(Registros['METODO'].unique())")

py_run_string("print(Registros['ID'].unique())")

py_run_string("print(Registros['Gremio'].unique())")

#--------------Crear tabla de esfuerzo de muestreo---------------------------

library(reticulate)
py_install(c("pandas", "numpy"))

py_install(c("tabulate", "openpyxl"))




py_run_string("
import pandas as pd
from tabulate import tabulate

# --- Copiar dataframe base ---
df = Registros.copy()

# --- Normalizar texto ---
df['METODO'] = df['METODO'].astype(str).str.strip()
df['COBERTURA'] = df['COBERTURA'].astype(str).str.strip()
df['ID'] = df['ID'].astype(str).str.strip()

# --- Diccionario de abreviaciones de coberturas ---
abreviaciones_cobertura = {
    'Bosque De Galería Y Ripario': 'Bgr',
    'Bosque De Galeria Y Ripario': 'Bgr',
    'Bosque Denso Alto De Tierra Firme': 'Bda',
    'Bosque Denso Bajo De Tierra Firme': 'Bdb',
    'Bosque Fragmentado Con Vegetación Secundaria': 'Bfvs',
    'Bosque Fragmentado Con Vegetacion Secundaria': 'Bfvs',
    'Sin Dato': 'NA'
}

# --- Aplicar reemplazos (con control de mayúsculas y tildes) ---
df['COBERTURA'] = df['COBERTURA'].apply(
    lambda x: abreviaciones_cobertura.get(x.strip().title(), x)
)

# --- Reemplazar vacíos y nulos por 'Sin dato' ---
df['METODO'] = df['METODO'].replace('', 'Sin dato').fillna('Sin dato')
df['COBERTURA'] = df['COBERTURA'].replace('', 'Sin dato').fillna('Sin dato')


# --- Validar que exista la nueva columna de horas ---
if 'Hora_Hombre' not in df.columns:
    raise ValueError('❌ La columna Hora_Hombre no existe en el dataframe Registros.')

# --- Calcular totales de individuos (sin perder registros) ---
individuos = (
    df.groupby(['METODO', 'COBERTURA'], dropna=False, as_index=False)['INDIVIDUOS']
      .sum(min_count=1)
)

# --- Calcular esfuerzo total único por ID ---
# (Evitamos duplicar horas si un ID aparece varias veces)
esfuerzo_unico = df[['ID', 'METODO', 'COBERTURA', 'Hora_Hombre']].drop_duplicates()

# --- Calcular esfuerzo total (solo una vez por ID) ---
esfuerzo = (
    esfuerzo_unico.groupby(['METODO', 'COBERTURA'], dropna=False, as_index=False)['Hora_Hombre']
    .sum(min_count=1)
    .rename(columns={'Hora_Hombre': 'Esfuerzo_horas'})
)


# --- Unir tablas ---
tabla = individuos.merge(esfuerzo, on=['METODO', 'COBERTURA'], how='outer')
tabla['Exito_captura'] = tabla['INDIVIDUOS'] / tabla['Esfuerzo_horas']

# --- Calcular totales por método ---
totales = tabla.groupby('METODO', as_index=False).agg({
    'INDIVIDUOS': 'sum',
    'Esfuerzo_horas': 'sum'
})
totales['Exito_captura'] = totales['INDIVIDUOS'] / totales['Esfuerzo_horas']
totales['COBERTURA'] = 'Total'

# --- Unir con la tabla principal ---
tabla_final = pd.concat([tabla, totales], ignore_index=True)

# --- Reestructurar para salida ---
tabla_melt = pd.melt(
    tabla_final,
    id_vars=['METODO', 'COBERTURA'],
    value_vars=['INDIVIDUOS', 'Esfuerzo_horas', 'Exito_captura'],
    var_name='Indice',
    value_name='Valor'
)

# --- Cambiar nombres de los índices ---
tabla_melt['Indice'] = tabla_melt['Indice'].replace({
    'INDIVIDUOS': 'Número de individuos',
    'Esfuerzo_horas': 'Esfuerzo captura (horas-hombre)',
    'Exito_captura': 'Éxito de captura (individuos/horas-hombre)'
})


# --- Orden lógico de los índices ---
orden_indices = [
    'Número de individuos',
    'Esfuerzo captura (horas-hombre)',
    'Éxito de captura (individuos/horas-hombre)'
]
tabla_melt['Indice'] = pd.Categorical(tabla_melt['Indice'], categories=orden_indices, ordered=True)

# --- Renombrar columna ---
tabla_melt = tabla_melt.rename(columns={'METODO': 'Metodología'})

# --- Orden personalizado de metodologías ---
orden_metodologia = [
    'Transecto',
    'Punto de observacion',
    'Red de niebla',
    'Camara Trampa',
    'Informacion Secundaria'
]

tabla_melt['Metodología'] = pd.Categorical(tabla_melt['Metodología'], categories=orden_metodologia, ordered=True)

# --- Orden personalizado de metodologías ---
orden_COBERTURA = [
    'Bgr',
    'Bfvs',
    'Bda',
    'Bdb',
    'Total'
]

tabla_melt['COBERTURA'] = pd.Categorical(tabla_melt['COBERTURA'], categories=orden_COBERTURA, ordered=True)

# --- Pivotar ---
tabla_pivot = tabla_melt.pivot_table(
    index=['Metodología', 'Indice'],
    columns='COBERTURA',
    values='Valor',
    aggfunc='first'
).reset_index()

# --- Redondear ---
tabla_pivot = tabla_pivot.round(3)

# --- Mostrar resumen en consola ---
print(tabulate(tabla_pivot, headers='keys', tablefmt='fancy_grid', floatfmt='.3f'))

# --- Exportar a Excel ---
output_file = 'Esfuerzo_Muestreo.xlsx'
tabla_pivot.to_excel(output_file, index=False)
print(f'\\n✅ Archivo exportado correctamente como {output_file}')
")


#-----------Dar formato------------


py_run_string("
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Alignment, Font, Border, Side
from openpyxl.utils import get_column_letter

# --- Nombre del archivo a formatear ---
output_file = 'Esfuerzo_Muestreo.xlsx'

# --- Cargar el archivo ---
wb = load_workbook(output_file)
ws = wb.active

# --- Estilos base ---
header_fill = PatternFill(start_color='BFD8B8', end_color='BFD8B8', fill_type='solid')
header_font = Font(bold=True, color='000000', name='Calibri')
center_align = Alignment(horizontal='center', vertical='center', wrap_text=True)

# --- Bordes finos para toda la tabla ---
thin_border = Border(
    left=Side(style='thin', color='000000'),
    right=Side(style='thin', color='000000'),
    top=Side(style='thin', color='000000'),
    bottom=Side(style='thin', color='000000')
)

# --- Aplicar formato y reemplazar vacíos ---
for row in ws.iter_rows():
    for cell in row:
        # Reemplazar vacíos o None por guion
        if cell.value is None or str(cell.value).strip() == '':
            cell.value = '-'
        # Aplicar formato general
        cell.border = thin_border
        cell.alignment = center_align

# --- Aplicar formato al encabezado ---
for cell in ws[1]:
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = center_align

# --- Ajustar ancho de columnas automáticamente ---
for col in ws.columns:
    max_length = 0
    column = get_column_letter(col[0].column)
    for cell in col:
        try:
            if cell.value:
                length = len(str(cell.value))
                if length > max_length:
                    max_length = length
        except:
            pass
    adjusted_width = max_length + 3
    ws.column_dimensions[column].width = adjusted_width

# --- Ajustar altura de filas automáticamente ---
for row in ws.iter_rows():
    max_height = 15
    for cell in row:
        if cell.value:
            lines = str(cell.value).count('\\n') + 1
            if lines > 1:
                max_height = 15 * lines
    ws.row_dimensions[cell.row].height = max_height

# --- Guardar cambios ---
wb.save(output_file)
print(f'📘 Archivo {output_file} formateado con éxito: celdas centradas, bordes finos y guiones en vacíos.')
")








# También puedes traer el DataFrame a R si lo necesitas:
esfuerzo_unico <- py$esfuerzo_unico
View(esfuerzo_unico)
py_run_string("print(esfuerzo_unico)")


#------------------------Tabla general grupo taxonomico-------------------------

py_run_string("# Mostrar las primeras filas
print(Registros.info())
")


py_run_string("
import pandas as pd

# --- Copiar el DataFrame base ---
df = Registros.copy()

# --- Normalizar texto ---
for col in ['CLASE', 'Orden', 'Familia', 'Genero', 'Epiteto', 'N. comun', 'Gremio', 'COBERTURA', 'METODOLOGIA']:
    df[col] = df[col].astype(str).str.strip().str.title()

# --- Crear nombre científico completo ---
df['Especie_cientifica'] = df['Genero'] + ' ' + df['Epiteto']

# --- Diccionario de abreviaciones de metodología ---
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

# --- Agrupar registros únicos por especie ---
tabla = (
    df.groupby(['CLASE', 'Orden', 'Familia', 'ESPECIE', 'N. comun', 'Gremio'], dropna=False)
      .agg({
          'COBERTURA': lambda x: ', '.join(sorted(set(x.dropna()))),
          'INDIVIDUOS': 'sum',
          'METODOLOGIA': lambda x: ', '.join(sorted(set(x.dropna())))
      })
      .reset_index()
)

# --- Crear tabla pivote con coberturas como columnas ---
pivot = (
    df.groupby(['ESPECIE', 'COBERTURA'], as_index=False)['INDIVIDUOS'].sum()
      .pivot(index='ESPECIE', columns='COBERTURA', values='INDIVIDUOS')
      .fillna(0)
      .reset_index()
)

# --- Unir tabla pivote con la tabla principal ---
tabla = tabla.merge(pivot, on='ESPECIE', how='left')

# --- Renombrar columnas ---
tabla = tabla.rename(columns={
    'CLASE': 'Clase',
    'Orden': 'Orden',
    'Familia': 'Familia',
    'ESPECIE': 'Especie',
    'N. comun': 'Nombre comun',
    'Gremio': 'Gremio trófico',
    'COBERTURA': 'Cobertura(s)',
    'INDIVIDUOS': 'Abundancia',
    'METODOLOGIA': 'Tipo de registro'
})

# --- Ordenar clases ---
orden_clase = ['Aves', 'Mammalia']
tabla['Clase'] = pd.Categorical(tabla['Clase'], categories=orden_clase + sorted(set(tabla['Clase']) - set(orden_clase)), ordered=True)

# --- Ordenar por Clase, Orden y Familia ---
tabla = tabla.sort_values(['Clase', 'Orden', 'Familia', 'Especie']).reset_index(drop=True)

# --- 🔹 Agregar conteo reiniciado por Clase ---
tabla['N°'] = tabla.groupby('Clase').cumcount() + 1

# --- 🔹 Insertar fila con nombres de columnas justo antes de Mammalia ---
# --- 🔹 Insertar fila con nombres de columnas justo antes de Mammalia ---
idx_mam = tabla.index[tabla['Clase'] == 'Mammalia']
if len(idx_mam) > 0:
    insert_pos = idx_mam[0]
    fila_header = pd.DataFrame([{col: str(col) for col in tabla.columns}])  # ✅ mantiene texto
    tabla = pd.concat([tabla.iloc[:insert_pos], fila_header, tabla.iloc[insert_pos:]], ignore_index=True)


# --- 🔹 Eliminar columnas duplicadas ---
tabla = tabla.loc[:, ~tabla.columns.duplicated()]

# --- 🔹 Reordenar columnas ---
columnas_orden = ['N°', 'Clase', 'Orden', 'Familia', 'Especie', 'Nombre comun',
                  'Gremio trófico', 'Bda', 'Bdb', 'Bfvs', 'Bgr',
                  'Abundancia', 'Tipo de registro']
tabla = tabla[[col for col in columnas_orden if col in tabla.columns]]

# --- Exportar a Excel ---
output_file = 'tabla_composicion_taxonomica.xlsx'
tabla.to_excel(output_file, index=False)

print(tabla.head(10))
print('\\n✅ Archivo exportado correctamente como', output_file)
")

#-----------Dar formato------------


py_run_string("
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Alignment, Font, Border, Side
from openpyxl.utils import get_column_letter

# --- Nombre del archivo a formatear ---
output_file = 'Esfuerzo_Muestreo.xlsx'

# --- Cargar el archivo ---
wb = load_workbook(output_file)
ws = wb.active

# --- Estilos base ---
header_fill = PatternFill(start_color='BFD8B8', end_color='BFD8B8', fill_type='solid')
header_font = Font(bold=True, color='000000', name='Calibri')
center_align = Alignment(horizontal='center', vertical='center', wrap_text=True)

# --- Bordes finos para toda la tabla ---
thin_border = Border(
    left=Side(style='thin', color='000000'),
    right=Side(style='thin', color='000000'),
    top=Side(style='thin', color='000000'),
    bottom=Side(style='thin', color='000000')
)

# --- Aplicar formato y reemplazar vacíos ---
for row in ws.iter_rows():
    for cell in row:
        # Reemplazar vacíos o None por guion
        if cell.value is None or str(cell.value).strip() == '':
            cell.value = '-'
        # Aplicar formato general
        cell.border = thin_border
        cell.alignment = center_align

# --- Aplicar formato al encabezado ---
for cell in ws[1]:
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = center_align

# --- Ajustar ancho de columnas automáticamente ---
for col in ws.columns:
    max_length = 0
    column = get_column_letter(col[0].column)
    for cell in col:
        try:
            if cell.value:
                length = len(str(cell.value))
                if length > max_length:
                    max_length = length
        except:
            pass
    adjusted_width = max_length + 3
    ws.column_dimensions[column].width = adjusted_width

# --- Ajustar altura de filas automáticamente ---
for row in ws.iter_rows():
    max_height = 15
    for cell in row:
        if cell.value:
            lines = str(cell.value).count('\\n') + 1
            if lines > 1:
                max_height = 15 * lines
    ws.row_dimensions[cell.row].height = max_height

# --- Guardar cambios ---
wb.save(output_file)
print(f'📘 Archivo {output_file} formateado con éxito: celdas centradas, bordes finos y guiones en vacíos.')
")


#------------------Figura de Ordenes familias---------------

# --- Instalar e importar reticulate ---
install.packages("reticulate")
library(reticulate)

# --- (Opcional) Configurar el entorno Python si no lo tienes ---
# use_python("/ruta/a/tu/python", required = TRUE)
# o usa el entorno por defecto de RStudio

# --- Instalar dependencias de Python si hace falta ---
py_install(c("pandas", "matplotlib", "seaborn", "openpyxl"))

# --- Ejecutar código Python ---
py_run_string("
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Leer el archivo Excel ---
tabla = pd.read_excel('especies.xlsx')

# --- Transformar a formato largo ---
tabla_larga = tabla.melt(id_vars='Orden', var_name='Familia', value_name='Num_especies')

# --- Filtrar ceros ---
tabla_larga = tabla_larga[tabla_larga['Num_especies'] > 0]

# --- Crear gráfico ---
plt.figure(figsize=(10,6))
sns.set(style='whitegrid')

# --- Gráfico de barras apiladas ---
tabla_pivot = tabla_larga.pivot(index='Orden', columns='Familia', values='Num_especies').fillna(0)
tabla_pivot.plot(kind='bar', stacked=True, colormap='tab20', edgecolor='black', figsize=(12,7))

# --- Etiquetas numéricas ---
for i, (idx, row) in enumerate(tabla_pivot.iterrows()):
    cumulative = 0
    for familia, valor in row.items():
        if valor > 0:
            plt.text(i, cumulative + valor/2, str(int(valor)), ha='center', va='center', fontsize=9)
            cumulative += valor

# --- Títulos y ejes ---
plt.title('Riqueza de especies por Orden y Familia', fontsize=14, fontweight='bold')
plt.xlabel('Orden')
plt.ylabel('Numero de especies')  # sin acento para evitar error
plt.legend(title='Familia', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
")
