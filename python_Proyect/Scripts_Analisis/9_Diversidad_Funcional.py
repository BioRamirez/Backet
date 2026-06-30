
# El objetivo del código es calcular un índice de valor funcional (FVI) para cada tipo de cobertura 
# (bosque, cultivo, potrero, etc.), usando rasgos ecológicos y biogeográficos de las especies observadas
#  y ponderándolos por su abundancia (número de individuos).



#--------------## Cargar librerias necesarias----------D:\CORPONOR 2025\Backet\python_Proyect\data\POF_PAMPLONITA_2023_BD_AVES_MAMIFEROS.xlsx--------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import openpyxl

#--------------## Leer archivo y revisar columnas------------------------------
# Ruta del archivo
ruta = r"D:\CORPONOR 2025\Backet\python_Proyect\data\POF_PAMPLONITA_2023_BD_AVES_MAMIFEROS.xlsx"

# Leer el archivo Excel
Registros = pd.read_excel(ruta)

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

    print("\n Abreviaciones generadas automáticamente:")
    for original, abrev in abreviaciones.items():
        print(f"  {original} → {abrev}")

    # Reemplazar en el DataFrame
    df[columna] = df[columna].replace(abreviaciones)

    return df, abreviaciones


# --- Aplicar las abreviaciones ---
Registros, abreviaciones_cobertura = abreviar_coberturas(Registros, columna='COBERTURA')



# Mostrar las primeras filas
print(" Primeras filas del archivo:")
print(Registros.head())

# Mostrar nombres de las columnas
print("\n Columnas del DataFrame:")
print(Registros.columns)

# Revisar los valores únicos de las variables clave
for col in ['Gremio', 'Tipo_Migra', 'Uso', 'Dist_Geo', 'Dist_Alt']:
    print(f"\n {col}:")
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

print("\n Índice de Valor Funcional por cobertura:")
print(tabulate(FVI_total.sort_values('Valor_funcional_ponderado', ascending=False), headers='keys', tablefmt='fancy_grid'))

#-----------------------------#
# 'Valor_funcional_ponderado'   'COBERTURA'
# --- Gráfico ---
plt.figure(figsize=(8, 5))
sns.barplot(
    data=FVI_total.sort_values('Valor_funcional_ponderado', ascending=False),
    x='COBERTURA',
    y='Valor_funcional_ponderado',
    palette='YlGn'
)
plt.title("", fontsize=14, fontweight='bold')
plt.xlabel("Valor funcional total (ponderado por abundancia)")
plt.ylabel("Cobertura")
#Índice de Valor Funcional por Cobertura
# --- Guardar gráfico ---
import os

ruta_fig = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados"
os.makedirs(ruta_fig, exist_ok=True)

plt.tight_layout()  # <-- primero ajustar

plt.savefig(
    os.path.join(ruta_fig, "9_FVI_por_cobertura.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()
























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

print(" Variables funcionales codificadas:")
print(datos_funcionales.head())

# ============================================================
# 2️⃣  Calcular matriz funcional y abundancia por cobertura
# ============================================================

# Promediar rasgos por especie
rasgos_medios = datos_funcionales.groupby('ESPECIE')[rasgos].mean()

# Calcular matriz de distancias funcionales (Euclidiana)
dist_funcional = distance.squareform(distance.pdist(rasgos_medios, metric='euclidean'))
dist_matrix_funcional = pd.DataFrame(dist_funcional, index=rasgos_medios.index, columns=rasgos_medios.index)

print("\n Matriz de distancias funcionales (primeras filas):")
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

print("\n Diversidad funcional (FD) por cobertura:")
print(FD_resultados)

# ============================================================
# 5️⃣  Visualización: Gráfico de barras FD
# ============================================================
#FD_resultados.index   'FD'
plt.figure(figsize=(9, 5))
sns.barplot(
    x=FD_resultados.index,
    y='FD',
    data=FD_resultados,
    palette='viridis'
)
plt.title("", fontsize=14, fontweight='bold')
plt.xlabel("Índice de Diversidad Funcional (FD)")
plt.ylabel("Cobertura")
# Diversidad Funcional (FD) por Cobertura

# --- Guardar gráfico ---
import os

ruta_fig = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados"
os.makedirs(ruta_fig, exist_ok=True)

plt.tight_layout()  # <-- primero ajustar

plt.savefig(
    os.path.join(ruta_fig, "9.1_FVI_por_cobertura.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()






















# ============================================================
# 5️⃣ Visualización: Gráfico de barras FD (Minimalista Profesional)
# ============================================================

import matplotlib.pyplot as plt
import numpy as np
import os

# ---- Estilo minimalista profesional ----
plt.style.use("default")

# Paleta científica suave (8 colores profesionales)
colores = [
    (0.15, 0.35, 0.55),
    (0.65, 0.25, 0.25),
    (0.25, 0.55, 0.25),
    (0.55, 0.45, 0.20),
    (0.35, 0.25, 0.55),
    (0.25, 0.45, 0.55),
    (0.45, 0.45, 0.45),
    (0.20, 0.20, 0.20)
]

# ------ Preparar datos ------
x_labels = FD_resultados.index
y_vals = FD_resultados["FD"].values
x = np.arange(len(x_labels))
colors_used = colores[:len(x_labels)]

# ------ Crear Figura ------
fig, ax = plt.subplots(figsize=(10, 5))

barras = ax.bar(x, y_vals, color=colors_used, width=0.6, edgecolor="black", linewidth=0.8)

# ------ Etiquetas numéricas sobre las barras ------
for bar, val in zip(barras, y_vals):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + max(y_vals)*0.015,
        f"{val:.2f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

# ------ Etiquetas de ejes ------
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=11)

ax.set_ylabel("Índice de Diversidad Funcional (FD)", fontsize=12)
ax.set_xlabel("Cobertura", fontsize=12)

# ------ Grilla sutil ------
ax.grid(axis="y", linestyle="--", alpha=0.35)

# ------ Estilo de espinas ------
for spine in ax.spines.values():
    spine.set_linewidth(0.8)
    spine.set_color("gray")

# ------ Título ------
ax.set_title("", fontsize=14)

plt.tight_layout()

# ------ Guardar figura ------
ruta_fig = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados"
os.makedirs(ruta_fig, exist_ok=True)

fig.savefig(
    os.path.join(ruta_fig, "9.1_FVI_por_coberturaPRO.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 5️⃣  Visualización: Gráfico de barras FD (Estilo Profesional)
# ============================================================
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Estilo Minimalista Profesional (TU ESTILO PREDEFINIDO) ---
plt.style.use("default")
sns.set_theme(
    style="whitegrid",
    rc={
        "axes.edgecolor": "0.3",
        "axes.linewidth": 0.6,
        "grid.color": "0.85",
        "grid.linewidth": 0.6,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 110
    }
)

# --- Paleta profesional (Paired) ---
palette = sns.color_palette("Paired", len(FD_resultados))

# --- Gráfico ---
plt.figure(figsize=(9, 5))

sns.barplot(
    x=FD_resultados.index,
    y='FD',
    data=FD_resultados,
    palette=palette
)

# --- Etiquetas y estilo ---
plt.xlabel("Cobertura", fontsize=11)
plt.ylabel("Índice de Diversidad Funcional (FD)", fontsize=11)
plt.title("", fontsize=12)

plt.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.55)

# --- Guardar gráfico ---
ruta_fig = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados"
os.makedirs(ruta_fig, exist_ok=True)

plt.tight_layout()
plt.savefig(
    os.path.join(ruta_fig, "9.1_FDI_por_cobertura.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()










# ============================================================
# 5️⃣  Visualización: Gráfico de barras FD (Estilo Profesional)
# ============================================================
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Estilo Minimalista Profesional ---
plt.style.use("default")
sns.set_theme(
    style="whitegrid",
    rc={
        "axes.edgecolor": "0.3",
        "axes.linewidth": 0.6,
        "grid.color": "0.85",
        "grid.linewidth": 0.6,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 110
    }
)

# --- Paleta profesional PAIRD ---
palette = sns.color_palette("Paired", len(FD_resultados))

# --- Gráfico ---
plt.figure(figsize=(9, 5))

sns.barplot(
    x=FD_resultados.index,
    y='FD',
    data=FD_resultados,
    palette=palette
)

# --- Etiquetas y estilo ---
plt.xlabel("Cobertura", fontsize=11)
plt.ylabel("Índice de Diversidad Funcional (FD)", fontsize=11)
plt.title("", fontsize=12)

plt.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.55)

# --- Guardar gráfico ---
ruta_fig = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados"
os.makedirs(ruta_fig, exist_ok=True)

plt.tight_layout()
plt.savefig(
    os.path.join(ruta_fig, "9.1_FDI_por_cobertura.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()





































# ============================================================
# 6️⃣  Visualización: Espacio funcional de especies (PCoA)
# ============================================================

coords = pcoa(distance.squareform(dist_funcional)).samples
coords.index = rasgos_medios.index

plt.figure(figsize=(7, 6))
sns.scatterplot(x=coords.iloc[:, 0], y=coords.iloc[:, 1], alpha=0.8)

plt.title(" Espacio Funcional de Especies (PCoA)", fontsize=14, fontweight='bold')
plt.xlabel("Eje funcional 1")
plt.ylabel("Eje funcional 2")


# --- Guardar gráfico ---
import os

ruta_fig = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados"
os.makedirs(ruta_fig, exist_ok=True)

plt.tight_layout()

plt.savefig(
    os.path.join(ruta_fig, "9.1_Espacio_Funcional.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()




#-----------------------INTERPRETACION-------------------------------
#--------------------------------------------------------------------
def interpretar_indice(df, columna_indice, columna_cobertura=None):
    """
    Genera interpretación automática para cualquier índice ecológico.
    
    - Si la cobertura NO está en columna, usa el índice del DataFrame.
    - Si la cobertura está en columna, la usa directamente.
    """

    # Detectar nombre de cobertura
    if columna_cobertura is None:
        coberturas = df.index
    else:
        coberturas = df[columna_cobertura]

    valores = df[columna_indice]

    texto = []
    texto.append("### INTERPRETACIÓN AUTOMÁTICA DEL ÍNDICE\n")

    # Estadísticos generales
    min_val = valores.min()
    max_val = valores.max()
    mean_val = valores.mean()

    texto.append(
        f"El índice '{columna_indice}' presentó valores entre "
        f"{min_val:.3f} y {max_val:.3f}, con un promedio general de {mean_val:.3f}. "
        "Esto refleja diferencias claras en la funcionalidad ecológica entre las coberturas evaluadas."
    )

    # Orden
    orden = valores.sort_values(ascending=False)

    mejor_cobertura = orden.index[0]
    peor_cobertura = orden.index[-1]

    texto.append(
        f"\nLa cobertura con mayor valor fue **{mejor_cobertura}** "
        f"({orden.iloc[0]:.3f}), indicando una mayor importancia ecológica o funcional."
    )

    texto.append(
        f"Por su parte, la cobertura con menor valor fue **{peor_cobertura}** "
        f"({orden.iloc[-1]:.3f}), reflejando un aporte funcional comparativamente menor."
    )

    # Diferencia relativa
    if orden.iloc[-1] != 0:
        diff_pct = ((orden.iloc[0] - orden.iloc[-1]) / orden.iloc[-1]) * 100
        texto.append(
            f"\nLa diferencia relativa entre la cobertura con mayor valor y la de menor valor fue de "
            f"aproximadamente **{diff_pct:.1f}%**, evidenciando un gradiente funcional marcado."
        )

    # Clasificación por percentiles
    texto.append("\n### Interpretación por rangos ecológicos\n")

    p33, p66 = valores.quantile([0.33, 0.66])

    for cov, val in valores.items():

        if val <= p33:
            categoria = "bajo"
            significado = "un aporte funcional reducido dentro del paisaje"
        elif val <= p66:
            categoria = "intermedio"
            significado = "una contribución ecológica moderada y estable"
        else:
            categoria = "alto"
            significado = "una relevancia funcional destacada para el ecosistema"

        texto.append(
            f"- **{cov}** obtuvo un valor de {val:.3f}, ubicándose en el rango **{categoria}**, "
            f"lo que indica {significado}."
        )

    # Explicación general del índice
    texto.append("\n\n### ¿Cómo se genera este índice?\n")
    texto.append(
        "El índice se calcula mediante un proceso automático que comprende los siguientes pasos:\n"
        "1. **Normalización de las variables funcionales**, de modo que todas operen en la misma escala.\n"
        "2. **Cálculo de contribuciones ecológicas** por cobertura, ponderadas según abundancia, riqueza, "
        "diversidad o importancia funcional.\n"
        "3. **Agregación de la información**, generando un valor único que resume la funcionalidad ecológica "
        "o diversidad funcional en cada cobertura.\n"
        "4. Este procedimiento se actualiza automáticamente cada vez que los datos del DataFrame cambian."
    )

    return "\n".join(texto)
#--------------------------------


interpretacion_fvi = interpretar_indice(
    FVI_total,
    columna_indice="Valor_funcional_ponderado",
    columna_cobertura="COBERTURA"
)

print(interpretacion_fvi)

#-------------------------------------------------

interpretacion_fd = interpretar_indice(
    FD_resultados,
    columna_indice="FD",
    columna_cobertura=None
)

print(interpretacion_fd)


#-----------------------------------------------Resumir interpretacion-----------------------
import pandas as pd
import os

def interpretar_fvi_fd(FVI_total, FD_resultados, ruta_salida):

    # ---------------------------------------------------------
    # 1. Convertir valores a float (corrección del error)
    # ---------------------------------------------------------
    try:
        FVI_total = {k: float(v) for k, v in FVI_total.items()}
        FD_resultados = {k: float(v) for k, v in FD_resultados.items()}
    except Exception as e:
        print("❌ ERROR: Existen valores no numéricos en FVI_total o FD_resultados")
        print("Detalle:", e)
        return None

    # ---------------------------------------------------------
    # 2. Ordenar
    # ---------------------------------------------------------
    fvi_sorted = dict(sorted(FVI_total.items(), key=lambda x: x[1], reverse=True))
    fd_sorted = dict(sorted(FD_resultados.items(), key=lambda x: x[1], reverse=True))

    # ---------------------------------------------------------
    # 3. Valores extremos
    # ---------------------------------------------------------
    cobertura_max_fvi = max(FVI_total, key=FVI_total.get)
    valor_max_fvi = FVI_total[cobertura_max_fvi]

    cobertura_min_fvi = min(FVI_total, key=FVI_total.get)
    valor_min_fvi = FVI_total[cobertura_min_fvi]

    cobertura_max_fd = max(FD_resultados, key=FD_resultados.get)
    valor_max_fd = FD_resultados[cobertura_max_fd]

    cobertura_min_fd = min(FD_resultados, key=FD_resultados.get)
    valor_min_fd = FD_resultados[cobertura_min_fd]

    # ---------------------------------------------------------
    # 4. Construcción del informe
    # ---------------------------------------------------------
    informe = []
    informe.append("="*58)
    informe.append("   INFORME AUTOMÁTICO DEL ESTADO FUNCIONAL DEL PAISAJE")
    informe.append("="*58 + "\n")

    informe.append("1. Ranking de coberturas según FVI:\n")
    for k, v in fvi_sorted.items():
        informe.append(f" - {k}: {v}")
    informe.append("")

    informe.append("2. Ranking de coberturas según Diversidad Funcional (FD):\n")
    for k, v in fd_sorted.items():
        informe.append(f" - {k}: {round(v, 3)}")
    informe.append("")

    informe.append("3. Interpretación del FVI por cobertura:\n")
    informe.append(
        f"La cobertura con mayor valor funcional integrado es {cobertura_max_fvi} ({valor_max_fvi})."
    )
    informe.append(
        f"La cobertura con menor FVI es {cobertura_min_fvi} ({valor_min_fvi}).\n"
    )

    informe.append("4. Interpretación del FD por cobertura:\n")
    informe.append(
        f"La mayor diversidad funcional se registró en {cobertura_max_fd} (FD = {round(valor_max_fd, 3)})."
    )
    informe.append(
        f"La menor diversidad funcional se observó en {cobertura_min_fd} (FD = {round(valor_min_fd, 3)}).\n"
    )

    informe.append("5. Síntesis ecológica integrada:\n")
    informe.append(
        "La integración de los índices FVI y FD permite una evaluación global del estado funcional del paisaje..."
    )

    return "\n".join(informe)

# ---------------------------------------------------------
# USO DE LA FUNCIÓN
# ---------------------------------------------------------

ruta_salida = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados"
os.makedirs(ruta_salida, exist_ok=True)

# 🔹 CONVERSIÓN NECESARIA (evita el error)
FVI_total_dict = dict(zip(FVI_total["COBERTURA"], FVI_total["Valor_funcional_ponderado"]))
FD_resultados_dict = FD_resultados["FD"].to_dict()

# 🔹 LLAMADA A LA FUNCIÓN (ya con diccionarios)
informe_resumen = interpretar_fvi_fd(FVI_total_dict, FD_resultados_dict, ruta_salida)

# 🔹 UNIR TODAS LAS INTERPRETACIONES EN UN ÚNICO TEXTO
informe_texto = (
    "=============================================\n"
    "      INFORME COMPLETO DE FUNCIONALIDAD\n"
    "=============================================\n\n"
    "### INTERPRETACIÓN DEL ÍNDICE FVI\n\n"
    + interpretacion_fvi +
    "\n\n---------------------------------------------\n\n"
    "### INTERPRETACIÓN DEL ÍNDICE FD\n\n"
    + interpretacion_fd +
    "\n\n---------------------------------------------\n\n"
    "### RESUMEN INTEGRADO (FVI + FD)\n\n"
    + informe_resumen
)

# 🔹 GUARDAR TXT
ruta_archivo = os.path.join(ruta_salida, "9_Informe_FVI_FD.txt")

try:
    with open(ruta_archivo, "w", encoding="utf-8") as f:
        f.write(informe_texto)

    print("\n=====================================")
    print("  ✅ ARCHIVO TXT GENERADO CORRECTAMENTE")
    print("  Ruta:", ruta_archivo)
    print("=====================================\n")

except Exception as e:
    print("\n❌ ERROR AL GUARDAR TXT:")
    print(e)















































# ============================================================
# 🔁 ANALISIS ROBUSTO POR MUNICIPIO (CON CONTROL DE ERRORES)
# ============================================================

ruta_salida = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\Municipios"
os.makedirs(ruta_salida, exist_ok=True)

municipios = Registros['MUNICIPIO'].dropna().unique()

for muni in municipios:

    print(f"\n==============================")
    print(f"📍 MUNICIPIO: {muni}")
    print(f"==============================")

    try:
        # ====================================================
        # FILTRAR DATOS
        # ====================================================
        df = Registros[Registros['MUNICIPIO'] == muni].copy()

        if df.empty:
            raise ValueError("DataFrame vacío")

        carpeta = os.path.join(ruta_salida, str(muni))
        os.makedirs(carpeta, exist_ok=True)

        # ====================================================
        # 🌿 FVI (SIN WARNING)
        # ====================================================
        FVI = (
            df.groupby(['COBERTURA','ESPECIE'])
            .agg({
                'INDIVIDUOS': 'sum',
                'Valor_funcional_especie': 'mean'
            })
        )

        FVI['Valor_funcional_ponderado'] = (
            FVI['INDIVIDUOS'] * FVI['Valor_funcional_especie']
        )

        FVI = FVI.reset_index()

        FVI_total = FVI.groupby('COBERTURA')['Valor_funcional_ponderado'].sum()

        if len(FVI_total) < 2:
            raise ValueError("No hay suficientes coberturas para FVI")

        print("\nFVI OK")

        # ====================================================
        # 📊 GRAFICO FVI
        # ====================================================
        fvi_plot = FVI_total.sort_values(ascending=False)

        plt.figure(figsize=(10,5))
        bars = plt.bar(fvi_plot.index, fvi_plot.values)

        for bar, val in zip(bars, fvi_plot.values):
            plt.text(
                bar.get_x() + bar.get_width()/2,
                val + max(fvi_plot.values)*0.02,
                f"{val:.1f}",
                ha='center',
                fontsize=9
            )

        plt.title(f"FVI - {muni}")
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()

        plt.savefig(os.path.join(carpeta, f"FVI_{muni}.png"), dpi=300)
        plt.close()

        # ====================================================
        # 🌱 FD
        # ====================================================
        rasgos = ['Gremio','Tipo_Migra','Uso','Dist_Geo']
        datos = df[['ESPECIE','COBERTURA'] + rasgos].dropna()

        if len(datos) < 3:
            raise ValueError("Muy pocos datos para FD")

        encoder = OrdinalEncoder()
        datos[rasgos] = encoder.fit_transform(datos[rasgos])

        rasgos_medios = datos.groupby('ESPECIE')[rasgos].mean()

        dist = distance.squareform(
            distance.pdist(rasgos_medios, metric='euclidean')
        )

        dist_matrix = pd.DataFrame(dist,
            index=rasgos_medios.index,
            columns=rasgos_medios.index
        )

        abund = df.groupby(['COBERTURA','ESPECIE'])['INDIVIDUOS'].sum().unstack(fill_value=0)

        FD = calc_FD(dist_matrix, abund)

        if FD.empty or len(FD) < 2:
            raise ValueError("FD no calculable")

        print("FD OK")

        # ====================================================
        # 📊 GRAFICO FD
        # ====================================================
        fd_plot = FD.sort_values('FD', ascending=False)

        plt.figure(figsize=(10,5))
        bars = plt.bar(fd_plot.index, fd_plot['FD'])

        for bar, val in zip(bars, fd_plot['FD']):
            plt.text(
                bar.get_x() + bar.get_width()/2,
                val + max(fd_plot['FD'])*0.02,
                f"{val:.2f}",
                ha='center',
                fontsize=9
            )

        plt.title(f"FD - {muni}")
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()

        plt.savefig(os.path.join(carpeta, f"FD_{muni}.png"), dpi=300)
        plt.close()

        # ====================================================
        # 🧾 INFORME
        # ====================================================
        texto = []
        texto.append(f"===== MUNICIPIO: {muni} =====\n")

        texto.append("FVI:")
        texto.append(str(fvi_plot))

        texto.append("\nFD:")
        texto.append(str(fd_plot))

        with open(os.path.join(carpeta, f"Informe_{muni}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(texto))

        print("✅ Municipio procesado correctamente")

    except Exception as e:

        print(f"❌ ERROR EN {muni}: {e}")

        # Guardar error en archivo
        with open(os.path.join(ruta_salida, f"ERROR_{muni}.txt"), "w", encoding="utf-8") as f:
            f.write(f"Error en municipio {muni}:\n{str(e)}")

        continue









import os

ruta_salida = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\Municipios"

# 🔹 Municipios originales del dataset
municipios_originales = set(Registros['MUNICIPIO'].dropna().unique())

# 🔹 Municipios que sí generaron carpeta (procesados OK)
municipios_procesados = set([
    f for f in os.listdir(ruta_salida)
    if os.path.isdir(os.path.join(ruta_salida, f))
])

# 🔹 Municipios con error
municipios_error = set([
    f.replace("ERROR_", "").replace(".txt", "")
    for f in os.listdir(ruta_salida)
    if f.startswith("ERROR_")
])

# 🔹 Municipios faltantes reales
municipios_faltantes = municipios_originales - municipios_procesados

# ===================================================
print("\n==============================")
print("📊 RESUMEN DE PROCESAMIENTO")
print("==============================")

print("\n✅ Procesados correctamente:")
for m in sorted(municipios_procesados):
    print("-", m)

print("\n❌ Con error:")
for m in sorted(municipios_error):
    print("-", m)

print("\n⚠️ Faltantes (no generaron salida):")
for m in sorted(municipios_faltantes):
    print("-", m)





















# ============================================================
# 📊 ANALISIS FUNCIONAL COMPLETO POR MUNICIPIO (FVI + FD + FUNCIONES)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import OrdinalEncoder
from scipy.spatial import distance

# ============================================================
# 📁 1. CARGAR DATOS
# ============================================================

ruta = r"D:\CORPONOR 2025\Backet\python_Proyect\data\POF_PAMPLONITA_2023_BD_AVES_MAMIFEROS.xlsx"
Registros = pd.read_excel(ruta)

# ============================================================
# 🔤 2. ABREVIAR COBERTURAS
# ============================================================

def generar_abreviacion(nombre):
    palabras = nombre.lower().split()
    palabras = [p for p in palabras if p not in ['de','del','la','el','y','con','en','los','las']]
    abrev = ''.join([p[0] for p in palabras])
    return abrev.capitalize().ljust(3, '_')

def abreviar_coberturas(df, columna='COBERTURA'):
    unicas = df[columna].dropna().unique()
    dic = {c: generar_abreviacion(c) for c in unicas}
    df[columna] = df[columna].replace(dic)
    return df

Registros = abreviar_coberturas(Registros)

# ============================================================
# ⚖️ 3. PESOS FUNCIONALES
# ============================================================

peso_gremio = {'Insectívoro':2,'Frugívoro':1,'Granívoro':2,'Carnívoro':1,
               'Herbivoro':3,'Carroñero':3,'Omnívoro':2,'Nectarívoro':1}

peso_migra = {'Res':4,'Lat-Trans':2,'Alt-Loc':2,'Loc':2,'Lat':2,
              'Nomadismo':3,'Estacional':2,'Residentes':4}

peso_uso = {'Uso Cultural':1,'Sin uso conocido':4,'Mascotas':2,'Subsistencia':1,
            'Medicinal':1,'Cultural':3,'Otro':3}

orden_geo = {'Endémica':1,'Casi endémica':2,'Restringida':3,
             'Neotropical':5,'Cosmopolita':6,'Introducida':7}

# ============================================================
# 🧮 4. VARIABLES NUMERICAS
# ============================================================

Registros['Gremio_valor'] = Registros['Gremio'].map(peso_gremio).fillna(1)
Registros['Tipo_Migra_valor'] = Registros['Tipo_Migra'].map(peso_migra).fillna(1)
Registros['Uso_valor'] = Registros['Uso'].map(peso_uso).fillna(1)
Registros['Dist_Geo_valor'] = Registros['Dist_Geo'].map(orden_geo).fillna(1)

Registros['Valor_funcional_especie'] = Registros[
    ['Gremio_valor','Tipo_Migra_valor','Uso_valor','Dist_Geo_valor']
].mean(axis=1)

# ============================================================
# 📊 FUNCION FD
# ============================================================

def calc_FD(dist_matrix, abundancias):
    resultados = {}
    for cobertura in abundancias.index:
        abunds = abundancias.loc[cobertura]
        especies = abunds[abunds > 0].index

        if len(especies) > 1:
            sub = dist_matrix.loc[especies, especies]
            fd = sub.values[np.triu_indices_from(sub, k=1)].mean()
        else:
            fd = 0

        resultados[cobertura] = fd

    return pd.DataFrame.from_dict(resultados, orient='index', columns=['FD'])

# ============================================================
# 🔁 ANALISIS POR MUNICIPIO
# ============================================================

ruta_salida = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\Municipios"
os.makedirs(ruta_salida, exist_ok=True)

municipios = Registros['MUNICIPIO'].dropna().unique()

for muni in municipios:

    print(f"\n==============================")
    print(f"📍 MUNICIPIO: {muni}")
    print(f"==============================")

    try:

        df = Registros[Registros['MUNICIPIO'] == muni].copy()

        if df.empty:
            raise ValueError("DataFrame vacío")

        carpeta = os.path.join(ruta_salida, str(muni))
        os.makedirs(carpeta, exist_ok=True)

        # ====================================================
        # 🌿 FVI
        # ====================================================
        FVI = (
            df.groupby(['COBERTURA','ESPECIE'])
            .agg({'INDIVIDUOS':'sum','Valor_funcional_especie':'mean'})
        )

        FVI['Valor_funcional_ponderado'] = (
            FVI['INDIVIDUOS'] * FVI['Valor_funcional_especie']
        )

        FVI = FVI.reset_index()

        FVI_total = FVI.groupby('COBERTURA')['Valor_funcional_ponderado'].sum()

        if len(FVI_total) < 2:
            raise ValueError("No hay suficientes coberturas")

        # ====================================================
        # 📊 GRAFICO FVI
        # ====================================================
        fvi_plot = FVI_total.sort_values(ascending=False)

        plt.figure(figsize=(10,5))
        bars = plt.bar(fvi_plot.index, fvi_plot.values)

        for bar, val in zip(bars, fvi_plot.values):
            plt.text(bar.get_x()+bar.get_width()/2,
                     val + max(fvi_plot.values)*0.02,
                     f"{val:.1f}", ha='center', fontsize=9)

        plt.title(f"FVI - {muni}")
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(carpeta, f"FVI_{muni}.png"), dpi=300)
        plt.close()

        # ====================================================
        # 🌱 FD
        # ====================================================
        rasgos = ['Gremio','Tipo_Migra','Uso','Dist_Geo']
        datos = df[['ESPECIE','COBERTURA'] + rasgos].dropna()

        if len(datos) < 3:
            raise ValueError("Datos insuficientes para FD")

        enc = OrdinalEncoder()
        datos[rasgos] = enc.fit_transform(datos[rasgos])

        rasgos_medios = datos.groupby('ESPECIE')[rasgos].mean()

        dist = distance.squareform(distance.pdist(rasgos_medios))
        dist_matrix = pd.DataFrame(dist, index=rasgos_medios.index, columns=rasgos_medios.index)

        abund = df.groupby(['COBERTURA','ESPECIE'])['INDIVIDUOS'].sum().unstack(fill_value=0)

        FD = calc_FD(dist_matrix, abund)

        if FD.empty or len(FD) < 2:
            raise ValueError("FD no calculable")

        fd_plot = FD.sort_values('FD', ascending=False)

        plt.figure(figsize=(10,5))
        bars = plt.bar(fd_plot.index, fd_plot['FD'])

        for bar, val in zip(bars, fd_plot['FD']):
            plt.text(bar.get_x()+bar.get_width()/2,
                     val + max(fd_plot['FD'])*0.02,
                     f"{val:.2f}", ha='center', fontsize=9)

        plt.title(f"FD - {muni}")
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(carpeta, f"FD_{muni}.png"), dpi=300)
        plt.close()

        # ====================================================
        # 🌿 FUNCIONES ECOLOGICAS
        # ====================================================
        funciones = (
            df.groupby(['COBERTURA','Gremio'])['INDIVIDUOS']
            .sum().reset_index()
        )

        totales = funciones.groupby('COBERTURA')['INDIVIDUOS'].sum().reset_index(name='TOTAL')
        funciones = funciones.merge(totales, on='COBERTURA')

        funciones['PORCENTAJE'] = (funciones['INDIVIDUOS']/funciones['TOTAL'])*100

        funciones_pivot = funciones.pivot_table(
            index='COBERTURA',
            columns='Gremio',
            values='PORCENTAJE',
            fill_value=0
        )

        # Dominante
        funcion_dominante = funciones.loc[
            funciones.groupby('COBERTURA')['PORCENTAJE'].idxmax()
        ][['COBERTURA','Gremio','PORCENTAJE']]

        # Riqueza funcional
        riqueza = df.groupby('COBERTURA')['Gremio'].nunique()

        # ====================================================
        # 📊 GRAFICO FUNCIONES
        # ====================================================
        funciones_pivot.plot(kind='bar', stacked=True, figsize=(10,5))

        plt.title(f"Funciones ecológicas - {muni}")
        plt.ylabel("% individuos")
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05,1))
        plt.tight_layout()
        plt.savefig(os.path.join(carpeta, f"FUNCIONES_{muni}.png"), dpi=300)
        plt.close()

        # ====================================================
        # 🧾 INFORME
        # ====================================================
        texto = []
        texto.append(f"===== MUNICIPIO: {muni} =====\n")

        texto.append("FVI:")
        texto.append(str(fvi_plot))

        texto.append("\nFD:")
        texto.append(str(fd_plot))

        texto.append("\nFUNCIONES (%):")
        texto.append(str(funciones_pivot.round(2)))

        texto.append("\nFUNCION DOMINANTE:")
        texto.append(str(funcion_dominante))

        texto.append("\nRIQUEZA FUNCIONAL:")
        texto.append(str(riqueza))

        with open(os.path.join(carpeta, f"Informe_{muni}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(texto))

        print("✅ Municipio procesado correctamente")

    except Exception as e:

        print(f"❌ ERROR EN {muni}: {e}")

        with open(os.path.join(ruta_salida, f"ERROR_{muni}.txt"), "w", encoding="utf-8") as f:
            f.write(str(e))

        continue

# ============================================================
# 📊 RESUMEN FINAL
# ============================================================

municipios_originales = set(Registros['MUNICIPIO'].dropna().unique())

municipios_procesados = set([
    f for f in os.listdir(ruta_salida)
    if os.path.isdir(os.path.join(ruta_salida, f))
])

municipios_error = set([
    f.replace("ERROR_", "").replace(".txt", "")
    for f in os.listdir(ruta_salida)
    if f.startswith("ERROR_")
])

faltantes = municipios_originales - municipios_procesados

print("\n==============================")
print("📊 RESUMEN FINAL")
print("==============================")

print("\n✅ Procesados:")
print(municipios_procesados)

print("\n❌ Error:")
print(municipios_error)

print("\n⚠️ Faltantes:")
print(faltantes)

print("\n✅ ANALISIS COMPLETO FINALIZADO")














# ============================================================
# 📊 ANALISIS FUNCIONAL COMPLETO POR MUNICIPIO + INTERPRETACIÓN
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import OrdinalEncoder
from scipy.spatial import distance

# ============================================================
# 📁 CARGAR DATOS
# ============================================================

ruta = r"D:\CORPONOR 2025\Backet\python_Proyect\data\POF_PAMPLONITA_2023_BD_AVES_MAMIFEROS.xlsx"
Registros = pd.read_excel(ruta)

# ============================================================
# 🔤 ABREVIAR COBERTURAS
# ============================================================

def generar_abreviacion(nombre):
    palabras = nombre.lower().split()
    palabras = [p for p in palabras if p not in ['de','del','la','el','y','con','en','los','las']]
    return ''.join([p[0] for p in palabras]).capitalize().ljust(3, '_')

Registros['COBERTURA'] = Registros['COBERTURA'].apply(generar_abreviacion)

# ============================================================
# ⚖️ VARIABLES FUNCIONALES
# ============================================================

peso_gremio = {'Insectívoro':2,'Frugívoro':1,'Granívoro':2,'Carnívoro':1,
               'Herbivoro':3,'Carroñero':3,'Omnívoro':2,'Nectarívoro':1}

peso_migra = {'Res':4,'Lat-Trans':2,'Alt-Loc':2,'Loc':2,'Lat':2,
              'Nomadismo':3,'Estacional':2,'Residentes':4}

peso_uso = {'Uso Cultural':1,'Sin uso conocido':4,'Mascotas':2,
            'Subsistencia':1,'Medicinal':1,'Cultural':3,'Otro':3}

orden_geo = {'Endémica':1,'Casi endémica':2,'Restringida':3,
             'Neotropical':5,'Cosmopolita':6,'Introducida':7}

# Mapear valores
Registros['Gremio_valor'] = Registros['Gremio'].map(peso_gremio).fillna(1)
Registros['Tipo_Migra_valor'] = Registros['Tipo_Migra'].map(peso_migra).fillna(1)
Registros['Uso_valor'] = Registros['Uso'].map(peso_uso).fillna(1)
Registros['Dist_Geo_valor'] = Registros['Dist_Geo'].map(orden_geo).fillna(1)

Registros['Valor_funcional_especie'] = Registros[
    ['Gremio_valor','Tipo_Migra_valor','Uso_valor','Dist_Geo_valor']
].mean(axis=1)

# ============================================================
# 📊 FUNCION FD
# ============================================================

def calc_FD(dist_matrix, abundancias):
    resultados = {}
    for cov in abundancias.index:
        especies = abundancias.loc[cov]
        presentes = especies[especies > 0].index

        if len(presentes) > 1:
            sub = dist_matrix.loc[presentes, presentes]
            fd = sub.values[np.triu_indices_from(sub, k=1)].mean()
        else:
            fd = 0

        resultados[cov] = fd

    return pd.DataFrame.from_dict(resultados, orient='index', columns=['FD'])

# ============================================================
# 🧾 FUNCIÓN DE INTERPRETACIÓN COMPLETA (FVI + FD + MULTIFUNCIÓN)
# ============================================================

import numpy as np

def interpretar_municipio(FVI, FD, gremio, migra, uso, geo, muni):

    texto = []
    texto.append(f"===== MUNICIPIO: {muni} =====\n")

    # =====================================================
    # 🔹 FVI
    # =====================================================
    texto.append("🔹 FVI (Importancia ecológica)")
    texto.append(f"Mayor cobertura: {FVI.idxmax()} ({FVI.max():.2f})")
    texto.append(f"Menor cobertura: {FVI.idxmin()} ({FVI.min():.2f})\n")

    # =====================================================
    # 🔹 FD
    # =====================================================
    texto.append("🔹 FD (Diversidad funcional)")
    texto.append(f"Mayor diversidad: {FD['FD'].idxmax()} ({FD['FD'].max():.2f})")
    texto.append(f"Menor diversidad: {FD['FD'].idxmin()} ({FD['FD'].min():.2f})\n")

    # =====================================================
    # 🔧 FUNCIONES AUXILIARES
    # =====================================================
    def shannon(p):
        p = p[p > 0]
        return -np.sum(p * np.log(p)) if len(p) > 0 else 0

    def pielou(H, S):
        return H / np.log(S) if S > 1 else 0

    # =====================================================
    # 🔹 FUNCIONES ECOLÓGICAS COMPLETAS
    # =====================================================
    texto.append("🔹 FUNCIONES ECOLÓGICAS POR COBERTURA\n")

    IMF_resultados = {}

    for cov in gremio.index:

        texto.append(f"--- {cov} ---")

        # ---------------------------
        # GREMIO
        # ---------------------------
        g = gremio.loc[cov]
        g = g[g > 0].sort_values(ascending=False)

        texto.append("Función trófica:")
        texto.append(", ".join([f"{k} ({v:.1f}%)" for k,v in g.items()]))

        p_g = (g / 100).values
        H_g = shannon(p_g)
        S_g = len(p_g)
        J_g = pielou(H_g, S_g)

        # ---------------------------
        # MIGRACIÓN
        # ---------------------------
        m = migra.loc[cov]
        m = m[m > 0].sort_values(ascending=False)

        texto.append("Conectividad (migración):")
        texto.append(", ".join([f"{k} ({v:.1f}%)" for k,v in m.items()]))

        p_m = (m / 100).values
        H_m = shannon(p_m)
        S_m = len(p_m)
        J_m = pielou(H_m, S_m)

        # ---------------------------
        # BIOGEOGRAFÍA
        # ---------------------------
        ge = geo.loc[cov]
        ge = ge[ge > 0].sort_values(ascending=False)

        texto.append("Importancia biogeográfica:")
        texto.append(", ".join([f"{k} ({v:.1f}%)" for k,v in ge.items()]))

        p_ge = (ge / 100).values
        H_ge = shannon(p_ge)
        S_ge = len(p_ge)
        J_ge = pielou(H_ge, S_ge)

        # ---------------------------
        # USO
        # ---------------------------
        u = uso.loc[cov]
        u = u[u > 0].sort_values(ascending=False)

        texto.append("Uso socioecológico:")
        texto.append(", ".join([f"{k} ({v:.1f}%)" for k,v in u.items()]))

        p_u = (u / 100).values
        H_u = shannon(p_u)
        S_u = len(p_u)
        J_u = pielou(H_u, S_u)

        # =====================================================
        # 🔥 ÍNDICE MULTIFUNCIONAL
        # =====================================================
        H_total = np.mean([H_g, H_m, H_ge, H_u])
        S_total = np.mean([S_g, S_m, S_ge, S_u])
        J_total = np.mean([J_g, J_m, J_ge, J_u])

        IMF = (H_total * 0.4) + (S_total * 0.3) + (J_total * 0.3)

        IMF_resultados[cov] = IMF

        texto.append(f"\nÍndice Multifuncional (IMF): {IMF:.3f}")
        texto.append(f" - Diversidad (H): {H_total:.3f}")
        texto.append(f" - Riqueza (S): {S_total:.2f}")
        texto.append(f" - Equidad (J): {J_total:.3f}")

        texto.append("")

    # =====================================================
    # 🔝 RANKING
    # =====================================================
    texto.append("🔹 RANKING MULTIFUNCIONAL\n")

    IMF_ordenado = dict(sorted(IMF_resultados.items(), key=lambda x: x[1], reverse=True))

    for cov, val in IMF_ordenado.items():
        texto.append(f"{cov}: {val:.3f}")

    # =====================================================
    # 🧠 INTERPRETACIÓN ECOLÓGICA
    # =====================================================
    mejor = max(IMF_resultados, key=IMF_resultados.get)
    peor = min(IMF_resultados, key=IMF_resultados.get)

    texto.append("\n🔹 INTERPRETACIÓN ECOLÓGICA\n")

    texto.append(
        f"La cobertura con mayor multifuncionalidad es {mejor} ({IMF_resultados[mejor]:.3f}), "
        "indicando alta diversidad y equilibrio de funciones ecológicas."
    )

    texto.append(
        f"La cobertura con menor multifuncionalidad es {peor} ({IMF_resultados[peor]:.3f}), "
        "lo que sugiere simplificación funcional o dominancia de pocos rasgos."
    )

    return "\n".join(texto)
# ============================================================
# 🔁 ANALISIS POR MUNICIPIO
# ============================================================

ruta_salida = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\Municipios"
os.makedirs(ruta_salida, exist_ok=True)

for muni in Registros['MUNICIPIO'].dropna().unique():

    print(f"\n📍 {muni}")

    try:
        df = Registros[Registros['MUNICIPIO'] == muni]

        carpeta = os.path.join(ruta_salida, muni)
        os.makedirs(carpeta, exist_ok=True)

        # =========================
        # FVI
        # =========================
        FVI = (
            df.groupby(['COBERTURA','ESPECIE'])
            .agg({'INDIVIDUOS':'sum','Valor_funcional_especie':'mean'})
        )

        FVI['Valor_funcional_ponderado'] = FVI['INDIVIDUOS'] * FVI['Valor_funcional_especie']
        FVI = FVI.reset_index()

        FVI_total = FVI.groupby('COBERTURA')['Valor_funcional_ponderado'].sum()

        # =========================
        # FD
        # =========================
        rasgos = ['Gremio','Tipo_Migra','Uso','Dist_Geo']
        datos = df[['ESPECIE','COBERTURA'] + rasgos].dropna()

        enc = OrdinalEncoder()
        datos[rasgos] = enc.fit_transform(datos[rasgos])

        rasgos_medios = datos.groupby('ESPECIE')[rasgos].mean()
        dist = distance.squareform(distance.pdist(rasgos_medios))

        dist_matrix = pd.DataFrame(dist, index=rasgos_medios.index, columns=rasgos_medios.index)

        abund = df.groupby(['COBERTURA','ESPECIE'])['INDIVIDUOS'].sum().unstack(fill_value=0)
        FD = calc_FD(dist_matrix, abund)

        # =========================
        # FUNCIONES MULTIRASGO
        # =========================

        def pivot_func(col):
            tmp = df.groupby(['COBERTURA',col])['INDIVIDUOS'].sum().reset_index()
            tot = tmp.groupby('COBERTURA')['INDIVIDUOS'].sum().reset_index(name='TOTAL')
            tmp = tmp.merge(tot, on='COBERTURA')
            tmp['PORC'] = tmp['INDIVIDUOS']/tmp['TOTAL']*100
            return tmp.pivot_table(index='COBERTURA', columns=col, values='PORC', fill_value=0)

        gremio_p = pivot_func('Gremio')
        migra_p = pivot_func('Tipo_Migra')
        uso_p = pivot_func('Uso')
        geo_p = pivot_func('Dist_Geo')

        # =========================
        # INTERPRETACIÓN
        # =========================

        informe = interpretar_municipio(
            FVI_total, FD, gremio_p, migra_p, uso_p, geo_p, muni
        )

        with open(os.path.join(carpeta, f"Informe_{muni}.txt"), "w", encoding="utf-8") as f:
            f.write(informe)

        print("✅ OK")

    except Exception as e:
        print(f"❌ ERROR {muni}: {e}")

# ============================================================
print("\n✅ PROCESO FINALIZADO")





def interpretar_municipio(FVI, FD, gremio, migra, uso, geo, muni):

    texto = []
    texto.append(f"===== MUNICIPIO: {muni} =====\n")

    # =========================
    # FVI
    # =========================
    texto.append("🔹 FVI (Importancia ecológica)")
    texto.append(f"Mayor cobertura: {FVI.idxmax()} ({FVI.max():.2f})")
    texto.append(f"Menor cobertura: {FVI.idxmin()} ({FVI.min():.2f})\n")

    # =========================
    # FD
    # =========================
    texto.append("🔹 FD (Diversidad funcional)")
    texto.append(f"Mayor diversidad: {FD['FD'].idxmax()} ({FD['FD'].max():.2f})")
    texto.append(f"Menor diversidad: {FD['FD'].idxmin()} ({FD['FD'].min():.2f})\n")

    # =========================
    # FUNCIONES COMPLETAS
    # =========================

    texto.append("🔹 FUNCIONES ECOLÓGICAS POR COBERTURA\n")

    for cov in gremio.index:

        texto.append(f"--- {cov} ---")

        # GREMIO
        g = gremio.loc[cov]
        g = g[g > 0].sort_values(ascending=False)
        texto.append("Función trófica:")
        texto.append(", ".join([f"{k} ({v:.1f}%)" for k,v in g.items()]))

        # MIGRACIÓN
        m = migra.loc[cov]
        m = m[m > 0].sort_values(ascending=False)
        texto.append("Conectividad (migración):")
        texto.append(", ".join([f"{k} ({v:.1f}%)" for k,v in m.items()]))

        # BIOGEOGRAFÍA
        ge = geo.loc[cov]
        ge = ge[ge > 0].sort_values(ascending=False)
        texto.append("Importancia biogeográfica:")
        texto.append(", ".join([f"{k} ({v:.1f}%)" for k,v in ge.items()]))

        # USO
        u = uso.loc[cov]
        u = u[u > 0].sort_values(ascending=False)
        texto.append("Uso socioecológico:")
        texto.append(", ".join([f"{k} ({v:.1f}%)" for k,v in u.items()]))

        texto.append("")

    return "\n".join(texto)




    # ============================================================
# 🌱 ANALISIS DE DISPERSORES DE SEMILLAS POR MUNICIPIO
# (NO MODIFICA EL SCRIPT ORIGINAL)
# ============================================================

import os
import pandas as pd
import numpy as np

ruta_salida_disp = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\Dispersores"
os.makedirs(ruta_salida_disp, exist_ok=True)

# Definir dispersores (ajustable)
dispersores = ['Frugívoro', 'Omnívoro']

for muni in Registros['MUNICIPIO'].dropna().unique():

    print(f"\n🌱 Analizando dispersión: {muni}")

    try:
        df = Registros[Registros['MUNICIPIO'] == muni].copy()

        # =========================
        # FILTRAR DISPERSORES
        # =========================
        df_disp = df[df['Gremio'].isin(dispersores)]

        if df_disp.empty:
            print("⚠️ Sin dispersores")
            continue

        # =========================
        # MÉTRICAS POR COBERTURA
        # =========================

        # Abundancia
        abund_disp = df_disp.groupby('COBERTURA')['INDIVIDUOS'].sum()

        # Riqueza
        riqueza_disp = df_disp.groupby('COBERTURA')['ESPECIE'].nunique()

        # Total individuos (para proporción)
        total_abund = df.groupby('COBERTURA')['INDIVIDUOS'].sum()

        # Proporción
        prop_disp = (abund_disp / total_abund) * 100

        # Índice simple de dispersión (opcional)
        IDF = (prop_disp * 0.6) + (riqueza_disp * 0.4)

        resumen = pd.DataFrame({
            'Abundancia': abund_disp,
            'Riqueza': riqueza_disp,
            'Proporcion_%': prop_disp,
            'IDF': IDF
        }).fillna(0).sort_values(by='IDF', ascending=False)

        # =========================
        # ESPECIES CLAVE
        # =========================
        especies_clave = (
            df_disp.groupby(['COBERTURA','ESPECIE'])['INDIVIDUOS']
            .sum()
            .reset_index()
            .sort_values(['COBERTURA','INDIVIDUOS'], ascending=[True, False])
        )

        top_especies = especies_clave.groupby('COBERTURA').head(5)

        # =========================
        # INTERPRETACIÓN AUTOMÁTICA
        # =========================
        texto = []
        texto.append(f"===== DISPERSIÓN DE SEMILLAS: {muni} =====\n")

        for cov in resumen.index:

            fila = resumen.loc[cov]

            texto.append(f"--- {cov} ---")
            texto.append(f"Abundancia: {fila['Abundancia']:.0f}")
            texto.append(f"Riqueza: {fila['Riqueza']:.0f}")
            texto.append(f"Proporción: {fila['Proporcion_%']:.1f}%")
            texto.append(f"IDF: {fila['IDF']:.2f}")

            # Interpretación ecológica
            if fila['Proporcion_%'] > 50:
                texto.append("Alta capacidad de regeneración natural (dispersión dominante).")
            elif fila['Proporcion_%'] > 25:
                texto.append("Capacidad media de dispersión de semillas.")
            else:
                texto.append("Baja dispersión → posible limitación en regeneración.")

            # Especies clave
            spp = top_especies[top_especies['COBERTURA'] == cov]['ESPECIE'].tolist()
            if spp:
                texto.append("Especies clave: " + ", ".join(spp))

            texto.append("")

        # Ranking
        texto.append("🔝 RANKING DE COBERTURAS (IDF):")
        for cov, val in resumen['IDF'].items():
            texto.append(f"{cov}: {val:.2f}")

        # =========================
        # GUARDAR RESULTADOS
        # =========================
        carpeta = os.path.join(ruta_salida_disp, muni)
        os.makedirs(carpeta, exist_ok=True)

        resumen.to_excel(os.path.join(carpeta, f"Dispersores_{muni}.xlsx"))
        top_especies.to_excel(os.path.join(carpeta, f"TopEspecies_{muni}.xlsx"))

        with open(os.path.join(carpeta, f"Informe_Dispersores_{muni}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(texto))

        print("✅ OK")

    except Exception as e:
        print(f"❌ ERROR {muni}: {e}")

print("\n🌱 ANALISIS DE DISPERSORES FINALIZADO")