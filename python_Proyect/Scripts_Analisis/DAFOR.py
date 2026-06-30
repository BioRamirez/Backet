
import os
import pandas as pd
import numpy as np
import re
from skbio.diversity import alpha_diversity, beta_diversity
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1️⃣ Cargar los datos originales
# --------------------------------------------------
ruta = r"D:\CORPONOR 2025\Backet\python_Proyect\data\POF_ZULIA_2025_BD_AVES_MAMIFEROS.xlsx"
ruta = r"D:\CORPONOR 2025\Backet\python_Proyect\data\SRF_LAM_5235_AVES_SAMORE_AVES.xlsx"
Registros = pd.read_excel(ruta)

print(" Primeras filas del archivo:")
print(Registros.head())
print("\n Columnas del DataFrame:")
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
        abreviacion = abreviacion.ljust(3, ' ')

    return abreviacion.upper()
    #return abreviacion.capitalize()


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


# --------------------------------------------------
import pandas as pd
import numpy as np

# --------------------------------------------------
# 1️⃣ Función para clasificar DAFOR
# --------------------------------------------------
def clasificar_dafor(frecuencia):
    if frecuencia >= 40:
        return 'D'  # Dominante
    elif frecuencia >= 20:
        return 'A'  # Abundante
    elif frecuencia >= 5:
        return 'F'  # Frecuente
    elif frecuencia >= 1:
        return 'O'  # Ocasional
    else:
        return 'R'  # Rara


# --------------------------------------------------
# 2️⃣ Cálculo DAFOR por cobertura (automático)
# --------------------------------------------------
def calcular_dafor_por_cobertura(
    df,
    especie_col='ESPECIE',
    individuos_col='INDIVIDUOS',
    cobertura_col='COBERTURA'
):
    resultados = []

    # Iterar automáticamente por cada cobertura
    for cobertura in df[cobertura_col].dropna().unique():

        df_cov = df[df[cobertura_col] == cobertura]

        # Abundancia por especie
        tabla = (
            df_cov
            .groupby(especie_col)[individuos_col]
            .sum()
            .reset_index()
        )

        total_individuos = tabla[individuos_col].sum()

        # Frecuencia relativa (%)
        tabla['Frecuencia_%'] = (tabla[individuos_col] / total_individuos) * 100

        # Clasificación DAFOR
        tabla['DAFOR'] = tabla['Frecuencia_%'].apply(clasificar_dafor)

        # Agregar metadatos
        tabla['Cobertura'] = cobertura
        tabla['Total_individuos_cobertura'] = total_individuos

        resultados.append(tabla)

    # Unir todas las coberturas
    df_dafor = pd.concat(resultados, ignore_index=True)

    # Ordenar para facilitar interpretación
    df_dafor = df_dafor.sort_values(
        by=['Cobertura', 'Frecuencia_%'],
        ascending=[True, False]
    )

    return df_dafor


# --------------------------------------------------
# 3️⃣ Ejecutar el análisis
# --------------------------------------------------
tabla_dafor = calcular_dafor_por_cobertura(Registros)

print("\n Vista previa de la clasificación DAFOR:")
print(tabla_dafor.head(10))


# --------------------------------------------------
# 4️⃣ Exportar resultados a Excel
# --------------------------------------------------
ruta_salida = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\DAFOR_Por_Cobertura.xlsx"

with pd.ExcelWriter(ruta_salida, engine='openpyxl') as writer:
    tabla_dafor.to_excel(writer, sheet_name='DAFOR', index=False)

print(f"\n Archivo DAFOR generado correctamente en:\n{ruta_salida}")



# ------------------Fin DAFOR por cobertura------------------#
#---------------gráfica de barras DAFOR por cobertura------------------

#---------------------------------------------------
#   GRAFICO DAFOR POR COBERTURA
#   ESTILO MINIMALISTA + PAIRED
#---------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# --------------------------------------------------
# 1️⃣ Preparar datos para el gráfico
# --------------------------------------------------
tabla_plot = (
    tabla_dafor
    .groupby(['Cobertura', 'DAFOR'])
    .size()
    .reset_index(name='Cantidad')
)

# Asegurar orden lógico DAFOR
orden_dafor = ['D', 'A', 'F', 'O', 'R']
tabla_plot['DAFOR'] = pd.Categorical(
    tabla_plot['DAFOR'],
    categories=orden_dafor,
    ordered=True
)

tabla_plot = tabla_plot.sort_values(['Cobertura', 'DAFOR'])

# --------------------------------------------------
# 2️⃣ Tema visual (idéntico al solicitado)
# --------------------------------------------------
sns.set_theme(style="whitegrid")
palette = sns.color_palette("Paired", 5)

fig, ax = plt.subplots(figsize=(14, 7))

# --------------------------------------------------
# 3️⃣ Gráfico de barras
# --------------------------------------------------
sns.barplot(
    data=tabla_plot,
    x="Cobertura",
    y="Cantidad",
    hue="DAFOR",
    palette=palette,
    edgecolor="black",
    linewidth=0.8,
    ax=ax
)

# --------------------------------------------------
# 4️⃣ Etiquetas de valores
# --------------------------------------------------
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.text(
            p.get_x() + p.get_width() / 2,
            height + 0.3,
            f"{int(height)}",
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )

# --------------------------------------------------
# 5️⃣ Estética minimalista profesional
# --------------------------------------------------
ax.set_facecolor("white")
ax.set_title(
    "Clasificación DAFOR por Cobertura",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel("Cobertura", fontsize=12)
ax.set_ylabel("Número de especies", fontsize=12)

plt.xticks(rotation=45, ha="right", fontsize=11)
plt.yticks(fontsize=11)

# Bordes finos
for spine in ax.spines.values():
    spine.set_linewidth(0.8)
    spine.set_color("#444444")

# Grid suave
ax.grid(True, axis="y", linestyle="--", alpha=0.35)
ax.grid(False, axis="x")

# --------------------------------------------------
# 6️⃣ Leyenda minimalista
# --------------------------------------------------
plt.legend(
    title="Categoría DAFOR",
    title_fontsize=12,
    fontsize=11,
    frameon=False,
    loc="upper right"
)

plt.tight_layout()

# --------------------------------------------------
# 7️⃣ Guardar figura
# --------------------------------------------------
ruta_fig = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados"
os.makedirs(ruta_fig, exist_ok=True)

plt.savefig(
    os.path.join(ruta_fig, "8.3_DAFOR_Por_Cobertura_PRO.png"),
    dpi=350,
    bbox_inches="tight"
)

plt.show()
plt.close()
