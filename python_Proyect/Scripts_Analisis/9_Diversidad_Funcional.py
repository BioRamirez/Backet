
# El objetivo del código es calcular un índice de valor funcional (FVI) para cada tipo de cobertura 
# (bosque, cultivo, potrero, etc.), usando rasgos ecológicos y biogeográficos de las especies observadas
#  y ponderándolos por su abundancia (número de individuos).



#--------------## Cargar librerias necesarias------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import openpyxl

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

# Revisar los valores únicos de las variables clave
for col in ['Gremio', 'Tipo_Migra', 'Uso', 'Dist_Geo', 'Dist_Alt']:
    print(f"\n📌 {col}:")
    print(Registros[col].value_counts(dropna=False))

#-----------------------------#
# --- Definir valores funcionales numéricos ---
peso_gremio = {
    'Insectívoro': 2,
    'Frugívoro': 1,
    'Granívoro': 2,
    'Carnívoro': 1,
    'Herbivoro': 3,
    'Carroñero': 3,
    'Omnívoro': 2,
    'Nectarívoro': 1
}

peso_migra = {
    'Res': 4,
    'Lat-Trans': 2,
    'Alt-Loc': 2,
    'Loc': 2,
    'Lat': 2,
    'Lat-Alt-Trans-Loc': 1,
    'Nomadismo': 3,
    'Estacional': 2,
    'Residentes': 4,
    'Latitudinal': 2
}

peso_uso = {
    'Uso Cultural': 1,
    'Sin uso conocido': 4,
    'Mascotas': 2,
    'Subsistencia': 1,
    'Medicinal': 1,
    'Cultural': 3,
    'Medicinal, Cultural': 1,
    'Mascotas, Subsistencia': 1,
    'Subsistencia, Mascotas': 1,
    'Otro': 3,
    'Mascota': 2,
    'Cultural, Mascotas': 2
}

orden_geo = {
    'Endémica': 1,
    'Casi endémica': 2,
    'Restringida': 3,
    'Neotropical': 5,
    'Nearctica, Neotropical': 4,
    'Cosmopolita': 6,
    'Introducida': 7
}

#-----------------------------#
# --- Crear nuevas columnas numéricas ---
Registros['Gremio_valor'] = Registros['Gremio'].map(peso_gremio).fillna(1)
Registros['Tipo_Migra_valor'] = Registros['Tipo_Migra'].map(peso_migra).fillna(1)
Registros['Uso_valor'] = Registros['Uso'].map(peso_uso).fillna(1)
Registros['Dist_Geo_valor'] = Registros['Dist_Geo'].map(orden_geo).fillna(1)

#-----------------------------#
# --- Calcular Diversidad Funcional (FVI) ---
Registros['Valor_funcional_especie'] = (
    Registros[['Gremio_valor', 'Tipo_Migra_valor', 'Uso_valor', 'Dist_Geo_valor']].mean(axis=1)
)

FVI = (
    Registros.groupby(['COBERTURA', 'ESPECIE'])
    .apply(lambda x: (x['INDIVIDUOS'].sum() * x['Valor_funcional_especie'].mean()))
    .reset_index(name='Valor_funcional_ponderado')
)

# --- Sumar por cobertura ---
FVI_total = FVI.groupby('COBERTURA')['Valor_funcional_ponderado'].sum().reset_index()

print("\n🌿 Índice de Valor Funcional por cobertura:")
print(tabulate(FVI_total.sort_values('Valor_funcional_ponderado', ascending=False), headers='keys', tablefmt='fancy_grid'))

#-----------------------------#
# --- Gráfico ---
plt.figure(figsize=(8, 5))
sns.barplot(
    data=FVI_total.sort_values('Valor_funcional_ponderado', ascending=False),
    x='Valor_funcional_ponderado',
    y='COBERTURA',
    palette='YlGn'
)
plt.title("Índice de Valor Funcional por Cobertura", fontsize=14, fontweight='bold')
plt.xlabel("Valor funcional total (ponderado por abundancia)")
plt.ylabel("Cobertura")
plt.tight_layout()
plt.show()





# análisis completo de diversidad funcional (FD) 
# basado en los rasgos biológicos de las especies y su abundancia en diferentes coberturas.





import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from scipy.spatial import distance
from skbio.stats.ordination import pcoa
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1️⃣  Seleccionar las columnas funcionales relevantes
# ============================================================
rasgos = ['Gremio', 'Tipo_Migra', 'Uso', 'Dist_Geo']

# Filtrar filas sin información funcional
datos_funcionales = Registros[['ESPECIE', 'COBERTURA'] + rasgos].dropna()

# Codificar variables categóricas a numéricas
encoder = OrdinalEncoder()
datos_funcionales[rasgos] = encoder.fit_transform(datos_funcionales[rasgos])

print("✅ Variables funcionales codificadas:")
print(datos_funcionales.head())

# ============================================================
# 2️⃣  Calcular matriz funcional y abundancia por cobertura
# ============================================================

# Promediar rasgos por especie
rasgos_medios = datos_funcionales.groupby('ESPECIE')[rasgos].mean()

# Calcular matriz de distancias funcionales (Euclidiana)
dist_funcional = distance.squareform(distance.pdist(rasgos_medios, metric='euclidean'))
dist_matrix_funcional = pd.DataFrame(dist_funcional, index=rasgos_medios.index, columns=rasgos_medios.index)

print("\n📏 Matriz de distancias funcionales (primeras filas):")
print(dist_matrix_funcional.head())

# Crear tabla de abundancia (filas=cobertura, columnas=especies)
tabla_abundancia = (
    Registros.groupby(['COBERTURA', 'ESPECIE'])['INDIVIDUOS']
    .sum()
    .unstack(fill_value=0)
)

# ============================================================
# 3️⃣  Función para calcular Diversidad Funcional (FD)
# ============================================================

def calc_FD(dist_matrix, abundancias):
    """
    Calcula la diversidad funcional (FD promedio) para cada cobertura.
    dist_matrix: DataFrame cuadrado de distancias funcionales entre especies.
    abundancias: DataFrame con coberturas en filas y especies en columnas.
    """
    fd_resultados = {}

    for cobertura in abundancias.index:
        abunds = abundancias.loc[cobertura]
        especies_presentes = abunds[abunds > 0].index

        if len(especies_presentes) > 1:
            sub_dist = dist_matrix.loc[especies_presentes, especies_presentes]
            # Promedio de distancias funcionales entre especies presentes
            fd = sub_dist.values[np.triu_indices_from(sub_dist, k=1)].mean()
        else:
            fd = 0

        fd_resultados[cobertura] = fd

    return pd.DataFrame.from_dict(fd_resultados, orient='index', columns=['FD'])

# ============================================================
# 4️⃣  Calcular FD por cobertura
# ============================================================

FD_resultados = calc_FD(dist_matrix_funcional, tabla_abundancia)
FD_resultados.sort_values('FD', ascending=False, inplace=True)

print("\n🌱 Diversidad funcional (FD) por cobertura:")
print(FD_resultados)

# ============================================================
# 5️⃣  Visualización: Gráfico de barras FD
# ============================================================

plt.figure(figsize=(9, 5))
sns.barplot(
    x='FD',
    y=FD_resultados.index,
    data=FD_resultados,
    palette='viridis'
)
plt.title("🌿 Diversidad Funcional (FD) por Cobertura", fontsize=14, fontweight='bold')
plt.xlabel("Índice de Diversidad Funcional (FD)")
plt.ylabel("Cobertura")
plt.tight_layout()
plt.show()

# ============================================================
# 6️⃣  Visualización: Espacio funcional de especies (PCoA)
# ============================================================

coords = pcoa(distance.squareform(dist_funcional)).samples
coords.index = rasgos_medios.index

plt.figure(figsize=(7, 6))
sns.scatterplot(x=coords.iloc[:, 0], y=coords.iloc[:, 1], alpha=0.8)

plt.title("🌾 Espacio Funcional de Especies (PCoA)", fontsize=14, fontweight='bold')
plt.xlabel("Eje funcional 1")
plt.ylabel("Eje funcional 2")
plt.tight_layout()
plt.show()








