import pandas as pd
import openpyxl

# Ruta del archivo D:\CORPONOR 2025\Backet\python_Proyect\data\SRF_LAM_5235_AVES_SAMORE_AVES _BF.xlsx
ruta = r"D:\Aeropuerto Aguachica\OSO_PARDO_2025\Muestreo\GDB\ANALISIS\ANFIBIOS\ANFIBIOS_OSO_PARDO_2025.xlsx"

# Leer el archivo Excel
Registros = pd.read_excel(ruta)

# =========================================================
# FILTRAR UNA SOLA CLASE
# =========================================================

grupo = "MAMIFEROS"   # AVES, MAMIFEROS, REPTILES, etc.

Registros = Registros[
    Registros["CLASE"].astype(str).str.upper() == grupo.upper()
].copy()

# Reiniciar índice
Registros.reset_index(drop=True, inplace=True)

# =========================================================
# VERIFICACIÓN
# =========================================================

print(f"\nGrupo seleccionado: {grupo}")
print(f"Número de registros: {len(Registros)}")

print("\nPrimeras filas:")
print(Registros.head())

# --- Paso 1. Asegurar formato de fecha (opcional, se mantiene por trazabilidad) ---
Registros['FECHA'] = pd.to_datetime(Registros['FECHA'])

# --- Paso 2. Crear tabla de abundancia por unidad de muestreo (ID) ---
tabla_abundancia = (
    Registros
    .groupby(['ESPECIE', 'ID'])['INDIVIDUOS']
    .sum()
    .unstack(fill_value=0)   # Filas = especies, columnas = ID (unidades de muestreo)
)

# --- Paso 3. Exportar a Excel ---
ruta_salida = r'D:\CORPONOR 2025\Backet\python_Proyect\Resultados\3_Tabla_Abundancia_Semanal.xlsx'

with pd.ExcelWriter(ruta_salida, engine='openpyxl') as writer:
    tabla_abundancia.to_excel(writer, sheet_name='Abundancia_Por_ID')

print(' Tabla de abundancia por unidad de muestreo (ID) creada en:')
print(ruta_salida)
print('\nVista previa:')
print(tabla_abundancia.head())



















# ============================
# TABLA DE ABUNDANCIA POR UNIDAD DE MUESTREO (ID + FECHA)
# ============================

import pandas as pd
import openpyxl

# --------------------------------------------------
# 1. RUTA DEL ARCHIVO DE ENTRADA
# --------------------------------------------------
ruta = r"D:\Aeropuerto Aguachica\OSO_PARDO_2025\Muestreo\GDB\ANALISIS\ANFIBIOS\ANFIBIOS_OSO_PARDO_2025.xlsx"

# --------------------------------------------------
# 2. LEER ARCHIVO EXCEL
# --------------------------------------------------
Registros = pd.read_excel(ruta)

# --------------------------------------------------
# 3. ASEGURAR FORMATO DE FECHA
# --------------------------------------------------
Registros['FECHA'] = pd.to_datetime(Registros['FECHA'])

# --------------------------------------------------
# 4. CREAR UNIDAD DE MUESTREO (ID + FECHA)
#    Cada combinación ID–FECHA es una unidad independiente
# --------------------------------------------------
Registros['UM'] = (
    Registros['ID'].astype(str) + "_" +
    Registros['FECHA'].dt.strftime('%Y%m%d')
)

# --------------------------------------------------
# 5. CREAR TABLA DE ABUNDANCIA
#    Filas: ESPECIE
#    Columnas: UM (ID_FECHA)
#    Valores: suma de INDIVIDUOS
# --------------------------------------------------
tabla_abundancia = (
    Registros
    .groupby(['ESPECIE', 'UM'])['INDIVIDUOS']
    .sum()
    .unstack(fill_value=0)
)

# --------------------------------------------------
# 6. EXPORTAR A EXCEL
# --------------------------------------------------
ruta_salida = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\3_Tabla_Abundancia_Semanal.xlsx"

with pd.ExcelWriter(ruta_salida, engine='openpyxl') as writer:
    tabla_abundancia.to_excel(writer, sheet_name='Abundancia_Por_UM')

# --------------------------------------------------
# 7. MENSAJES DE CONFIRMACIÓN
# --------------------------------------------------
print(" Tabla de abundancia creada correctamente")
print(" Unidad de muestreo definida como: ID + FECHA")
print(" Archivo guardado en:")
print(ruta_salida)

print("\n Vista previa de la tabla:")
print(tabla_abundancia.head())

# ============================
# FIN DEL SCRIPT
# ============================









#ID REAL POR FECHA

import pandas as pd
import openpyxl

# Ruta del archivo
ruta = r"D:\Forestal Consultores\2026\FAUNA\BD\BD_SANROQUE.xlsx"
# Leer el archivo Excel
Registros = pd.read_excel(ruta)



# =========================================================
# FILTRAR UNA SOLA CLASE
# =========================================================

grupo = "MAMIFEROS"   # AVES, MAMIFEROS, REPTILES, etc.

Registros = Registros[
    Registros["CLASE"].astype(str).str.upper() == grupo.upper()
].copy()

# Reiniciar índice
Registros.reset_index(drop=True, inplace=True)

# =========================================================
# VERIFICACIÓN
# =========================================================

print(f"\nGrupo seleccionado: {grupo}")
print(f"Número de registros: {len(Registros)}")

print("\nPrimeras filas:")
print(Registros.head())
# --- Paso 1. Asegurar formato de fecha ---
Registros['FECHA'] = pd.to_datetime(Registros['FECHA'])

# --- Paso 2. Definir el orden cronológico de los ID ---
# Se toma la fecha más antigua asociada a cada ID
orden_id = (
    Registros
    .groupby('ID')['FECHA']
    .min()
    .sort_values()          # De la fecha más antigua a la más reciente
    .index
    .tolist()
)

# --- Paso 3. Crear tabla de abundancia por especie y unidad de muestreo (ID) ---
tabla_abundancia = (
    Registros
    .groupby(['ESPECIE', 'ID'])['INDIVIDUOS']
    .sum()
    .unstack(fill_value=0)
)

# --- Paso 4. Reordenar las columnas según el orden cronológico de los ID ---
tabla_abundancia = tabla_abundancia[orden_id]

# --- Paso 5. Exportar a Excel ---
ruta_salida = r'D:\CORPONOR 2025\Backet\python_Proyect\Resultados\3_Tabla_Abundancia_Semanal.xlsx'

with pd.ExcelWriter(ruta_salida, engine='openpyxl') as writer:
    tabla_abundancia.to_excel(writer, sheet_name='Abundancia_Por_ID')

print('Tabla de abundancia por unidad de muestreo (ID) creada en:')
print(ruta_salida)
print('\nOrden cronológico de los ID:')
print(orden_id)
print('\nVista previa:')
print(tabla_abundancia.head())






#------------------Fin Tabla de Abundancia por Unidad de Muestreo (FECHA-Día)------------------#

import pandas as pd
import openpyxl

## Ruta del archivo
ruta = r"D:\Forestal Consultores\2026\FAUNA\BD\BD_SANROQUE.xlsx"
# Leer el archivo Excel
Registros = pd.read_excel(ruta)



# =========================================================
# FILTRAR UNA SOLA CLASE
# =========================================================

grupo = "MAMIFEROS"   # AVES, MAMIFEROS, REPTILES, etc.

Registros = Registros[
    Registros["CLASE"].astype(str).str.upper() == grupo.upper()
].copy()

# Reiniciar índice
Registros.reset_index(drop=True, inplace=True)

# =========================================================
# VERIFICACIÓN
# =========================================================

print(f"\nGrupo seleccionado: {grupo}")
print(f"Número de registros: {len(Registros)}")

print("\nPrimeras filas:")
print(Registros.head())
# -----------------------------
# 1. Asegurar formato de fecha
# -----------------------------
Registros['FECHA'] = pd.to_datetime(Registros['FECHA']).dt.date

# -----------------------------
# 2. Orden cronológico real
# -----------------------------
orden_fechas = (
    Registros['FECHA']
    .sort_values()
    .unique()
)

# -----------------------------
# 3. Tabla de abundancia por especie y fecha
# -----------------------------
tabla_abundancia = (
    Registros
    .groupby(['ESPECIE', 'FECHA'])['INDIVIDUOS']
    .sum()
    .unstack(fill_value=0)
)

# -----------------------------
# 4. Reordenar columnas por fecha
# -----------------------------
tabla_abundancia = tabla_abundancia[orden_fechas]

# -----------------------------
# 5. Exportar a Excel
# -----------------------------
ruta_salida = r'D:\CORPONOR 2025\Backet\python_Proyect\Resultados\3_Tabla_Abundancia_Semanal.xlsx'

with pd.ExcelWriter(ruta_salida, engine='openpyxl') as writer:
    tabla_abundancia.to_excel(writer, sheet_name='Abundancia_Por_Fecha')

print("✔ Tabla de abundancia por FECHA creada correctamente")
print("Fechas usadas como unidades de muestreo:")
print(orden_fechas)
