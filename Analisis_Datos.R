#------------------------------------------Trabajar en python-----------------------------
library(reticulate)
use_python("C:/Users/Ramirez Juan/AppData/Local/Microsoft/WindowsApps/python.exe", required = TRUE)


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
py_run_string("print(Registros['METODOLOGIA'].unique())")

py_run_string("print(Registros['METODO'].unique())")




py_run_string("
import pandas as pd
import numpy as np

# Copia del DataFrame original
df = Registros.copy()

# Normalizar texto para evitar problemas de mayúsculas o espacios
df['METODO'] = df['METODO'].str.strip()

# --- Definir esfuerzo base por tipo de método ---
esfuerzo_dict = {
  'Transecto': 240,           # horas-hombre
  'Punto de observacion': 2,  # horas-punto
  'Red de niebla': 24,        # horas-red
  'Camara Trampa': 216,       # horas-cámara
  'Informacion Secundaria': 1 # sin esfuerzo directo
}

# Filtrar solo métodos que están en el diccionario
df = df[df['METODO'].isin(esfuerzo_dict.keys())]

# Asignar el esfuerzo según el método
df['Esfuerzo'] = df['METODO'].map(esfuerzo_dict)

# Agrupar por método y cobertura
tabla = (
  df.groupby(['METODO', 'COBERTURA'], dropna=False)
  .agg({'INDIVIDUOS': 'sum', 'Esfuerzo': 'first'})
  .reset_index()
)

# Calcular éxito de captura
tabla['Exito_Captura'] = tabla['INDIVIDUOS'] / tabla['Esfuerzo']

# Calcular totales por método
totales = (
  tabla.groupby('METODO')
  .agg({'INDIVIDUOS': 'sum', 'Esfuerzo': 'sum'})
  .reset_index()
)
totales['COBERTURA'] = 'Total'
totales['Exito_Captura'] = totales['INDIVIDUOS'] / totales['Esfuerzo']

# Unir totales con la tabla principal
tabla_final = pd.concat([tabla, totales], ignore_index=True)

# Reorganizar para formato ancho (tipo matriz resumen)
tabla_pivot = tabla_final.pivot(index='METODO', columns='COBERTURA', values=['INDIVIDUOS','Esfuerzo','Exito_Captura'])
tabla_pivot = tabla_pivot.round(2)

# Mostrar resultado final
print(tabla_pivot)
")
# También puedes traer el DataFrame a R si lo necesitas:
tabla_pivot <- py$tabla_pivot
View(tabla_pivot)



reticulate::py_run_string("
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tabulate'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'openpyxl'])
")







py_run_string("
import pandas as pd
import numpy as np
from tabulate import tabulate   # para mostrar tabla bonita
import openpyxl                 # para exportar a Excel

# Copiar dataframe base
df = Registros.copy()

# Normalizar texto para evitar errores por mayúsculas o espacios
df['METODO'] = df['METODO'].astype(str).str.strip()
df['COBERTURA'] = df['COBERTURA'].astype(str).str.strip()

# --- Definir esfuerzo base por tipo de método ---
esfuerzo_dict = {
    'Transecto': 240,           # horas-hombre
    'Punto de observacion': 2,  # horas-punto
    'Red de niebla': 24,        # horas-red
    'Camara Trampa': 216,       # horas-cámara
    'Informacion Secundaria': 1 # sin esfuerzo directo
}

# Filtrar solo métodos definidos
df = df[df['METODO'].isin(esfuerzo_dict.keys())]

# Asignar esfuerzo
df['Esfuerzo'] = df['METODO'].map(esfuerzo_dict)

# Agrupar por método y cobertura
tabla = (
    df.groupby(['METODO', 'COBERTURA'], dropna=False)
      .agg({'INDIVIDUOS': 'sum', 'Esfuerzo': 'first'})
      .reset_index()
)

# Calcular éxito de captura
tabla['Exito_Captura'] = tabla['INDIVIDUOS'] / tabla['Esfuerzo']

# Calcular totales por método
totales = (
    tabla.groupby('METODO')
         .agg({'INDIVIDUOS': 'sum', 'Esfuerzo': 'sum'})
         .reset_index()
)
totales['COBERTURA'] = 'Total'
totales['Exito_Captura'] = totales['INDIVIDUOS'] / totales['Esfuerzo']

# Unir totales
tabla_final = pd.concat([tabla, totales], ignore_index=True)

# Redondear valores
tabla_final[['INDIVIDUOS', 'Esfuerzo', 'Exito_Captura']] = tabla_final[['INDIVIDUOS', 'Esfuerzo', 'Exito_Captura']].round(2)

# Mostrar tabla formateada
print(tabulate(tabla_final, headers='keys', tablefmt='fancy_grid', showindex=False))

# Exportar a Excel
tabla_final.to_excel('tabla_esfuerzo_captura.xlsx', index=False)

print('\\n✅ Archivo exportado como tabla_esfuerzo_captura.xlsx en el directorio de trabajo.')
")
