#--------------## Cargar librerias necesarias------------------------------

py_install(c("pandas", "numpy", "matplotlib", "tabulate", "openpyxl"))

#--------------## Leer archivo y revisar columnas------------------------------
# Mostrar las primeras filas
import pandas as pd

# Leer el archivo Excel
Registros = pd.read_excel('D:\CORPONOR 2025\Backet\python_Proyect\data\POF_ZULIA_2025_BD_AVES_MAMIFEROS.xlsx')

# Mostrar las primeras filas
print(Registros.head())


print(Registros.head())

print(Registros.columns)

print(Registros['METODOLOGIA'].unique())

print(Registros['METODO'].unique())

print(Registros['ID'].unique())

print(Registros['Gremio'].unique())

#--------------## Esfuerzo de Muestreo------------------------------