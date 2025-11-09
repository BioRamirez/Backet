import pandas as pd
import openpyxl

# --- Paso 1. Asegurar formato de fecha ---
Registros['FECHA'] = pd.to_datetime(Registros['FECHA'])

# --- Paso 2. Crear rangos semanales ---
Registros['RANGO_FECHA'] = Registros['FECHA'].dt.to_period('W')

# --- Paso 3. Crear tabla de abundancia ---
# Agrupamos por especie y rango, sumando el número de individuos
tabla_abundancia = (
    Registros
    .groupby(['ESPECIE', 'RANGO_FECHA'])['INDIVIDUOS']
    .sum()
    .unstack(fill_value=0)   # Filas = especies, columnas = rangos
)

# --- Paso 4. Exportar a Excel ---
ruta_salida = 'D:/CORPONOR 2025/Backet/python_Proyect/Resultados/Tabla_Abundancia_Semanal.xlsx'
with pd.ExcelWriter(ruta_salida, engine='openpyxl') as writer:
    tabla_abundancia.to_excel(writer, sheet_name='Abundancia_Semanal')

print('✅ Tabla de abundancia creada y guardada en:', ruta_salida)
print('\\nVista previa:')
print(tabla_abundancia.head())


