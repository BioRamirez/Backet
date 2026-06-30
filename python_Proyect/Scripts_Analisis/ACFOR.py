# ==================================================
# CLASIFICACIÓN SEMICUANTITATIVA ACFOR
# Anfibios y Reptiles
# Adaptado de Cunningham et al. (1984) y Strong & Johnson (2020)
# ==================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1️⃣ RUTA DEL ARCHIVO
# --------------------------------------------------
ruta = r"D:\CORPONOR 2025\Backet\python_Proyect\data\SRF_LAM_5235_AVES_SAMORE_HERPETOS.xlsx"
df = pd.read_excel(ruta)

# --------------------------------------------------
# 2️⃣ FUNCIÓN PARA ABREVIAR COBERTURAS
# --------------------------------------------------
def generar_abreviacion(nombre):
    palabras = nombre.lower().split()
    palabras = [p for p in palabras if p not in ['de','del','la','el','y','con','en','los','las']]
    abrev = ''.join([p[0] for p in palabras])[:3]
    return abrev.upper()

def abreviar_coberturas(df, columna='COBERTURA'):
    mapa = {c: generar_abreviacion(c) for c in df[columna].dropna().unique()}
    df[columna] = df[columna].replace(mapa)
    return df, mapa

df, abreviaciones = abreviar_coberturas(df)

# --------------------------------------------------
# 3️⃣ FUNCIÓN DE CLASIFICACIÓN ACFOR (Pi)
# --------------------------------------------------
def clasificar_acfor(pi):
    if pi > 0.50:
        return 'A'  # Abundante
    elif pi > 0.35:
        return 'C'  # Común
    elif pi > 0.10:
        return 'F'  # Frecuente
    elif pi > 0.05:
        return 'O'  # Ocasional
    else:
        return 'R'  # Rara

# --------------------------------------------------
# 4️⃣ CÁLCULO ACFOR POR COBERTURA
# --------------------------------------------------
def calcular_acfor(
    df,
    especie_col='ESPECIE',
    individuos_col='INDIVIDUOS',
    cobertura_col='COBERTURA'
):
    resultados = []

    for cobertura in df[cobertura_col].dropna().unique():

        sub = df[df[cobertura_col] == cobertura]

        tabla = (
            sub
            .groupby(especie_col)[individuos_col]
            .sum()
            .reset_index()
        )

        total = tabla[individuos_col].sum()
        tabla['Pi'] = tabla[individuos_col] / total
        tabla['ACFOR'] = tabla['Pi'].apply(clasificar_acfor)

        tabla['Cobertura'] = cobertura
        tabla['Total_individuos'] = total

        resultados.append(tabla)

    df_acfor = pd.concat(resultados, ignore_index=True)

    df_acfor = df_acfor.sort_values(
        by=['Cobertura', 'Pi'],
        ascending=[True, False]
    )

    return df_acfor

tabla_acfor = calcular_acfor(df)

# --------------------------------------------------
# 5️⃣ EXPORTAR A EXCEL
# --------------------------------------------------
ruta_salida = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados"
os.makedirs(ruta_salida, exist_ok=True)

archivo_excel = os.path.join(ruta_salida, "ACFOR_Por_Cobertura.xlsx")

tabla_acfor.to_excel(archivo_excel, index=False)

print("Archivo ACFOR exportado correctamente:")
print(archivo_excel)

# --------------------------------------------------
# 6️⃣ PREPARAR DATOS PARA GRÁFICO
# --------------------------------------------------
tabla_plot = (
    tabla_acfor
    .groupby(['Cobertura', 'ACFOR'])
    .size()
    .reset_index(name='Numero_especies')
)

orden_acfor = ['A', 'C', 'F', 'O', 'R']
tabla_plot['ACFOR'] = pd.Categorical(
    tabla_plot['ACFOR'],
    categories=orden_acfor,
    ordered=True
)

tabla_plot = tabla_plot.sort_values(['Cobertura', 'ACFOR'])

# --------------------------------------------------
# 7️⃣ GRÁFICO MINIMALISTA PROFESIONAL (EXCEL FRIENDLY)
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 6))

for categoria in orden_acfor:
    sub = tabla_plot[tabla_plot['ACFOR'] == categoria]
    ax.bar(
        sub['Cobertura'],
        sub['Numero_especies'],
        label=categoria
    )

ax.set_title(
    "Clasificación semicuantitativa ACFOR por cobertura",
    fontsize=15,
    fontweight='bold'
)

ax.set_xlabel("Cobertura")
ax.set_ylabel("Número de especies")

ax.legend(
    title="Categoría ACFOR",
    frameon=False
)

ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.grid(axis='x', visible=False)

for spine in ax.spines.values():
    spine.set_linewidth(0.8)

plt.tight_layout()

# --------------------------------------------------
# 8️⃣ GUARDAR FIGURA
# --------------------------------------------------
fig_path = os.path.join(ruta_salida, "ACFOR_Por_Cobertura.png")
plt.savefig(fig_path, dpi=350, bbox_inches='tight')
plt.show()
plt.close()

print("Figura ACFOR generada correctamente:")
print(fig_path)



#---------------gráfica de barras ACFOR por cobertura------------------
#--------------- GRÁFICA DE BARRAS ACFOR POR COBERTURA ------------------

#---------------------------------------------------
#   GRAFICO ACFOR POR COBERTURA
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
    tabla_acfor
    .groupby(['Cobertura', 'ACFOR'])
    .size()
    .reset_index(name='Cantidad')
)

# Asegurar orden lógico ACFOR
orden_acfor = ['A', 'C', 'F', 'O', 'R']
tabla_plot['ACFOR'] = pd.Categorical(
    tabla_plot['ACFOR'],
    categories=orden_acfor,
    ordered=True
)

tabla_plot = tabla_plot.sort_values(['Cobertura', 'ACFOR'])

# --------------------------------------------------
# 2️⃣ Tema visual
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
    hue="ACFOR",
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
            height + 0.1,
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
    "Clasificación semicuantitativa ACFOR por cobertura",
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
    title="Categoría ACFOR",
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
    os.path.join(ruta_fig, "8.3_ACFOR_Por_Cobertura_PRO.png"),
    dpi=350,
    bbox_inches="tight"
)

plt.show()
plt.close()
















# ==================================================
# 9️⃣ GRÁFICO DE TORTA ACFOR
# Todas las especies de anfibios y reptiles
# ==================================================

import matplotlib.pyplot as plt

# --------------------------------------------------
# 1️⃣ Preparar datos (conteo total de especies por ACFOR)
# --------------------------------------------------
tabla_pie = (
    tabla_acfor
    .drop_duplicates(subset='ESPECIE')  # cada especie una sola vez
    .groupby('ACFOR')
    .size()
    .reindex(['A', 'C', 'F', 'O', 'R'])
)

# Eliminar categorías sin datos
tabla_pie = tabla_pie[tabla_pie > 0]

# --------------------------------------------------
# 2️⃣ Gráfico de torta (minimalista profesional)
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 7))

ax.pie(
    tabla_pie.values,
    labels=tabla_pie.index,
    autopct='%1.1f%%',
    startangle=90,
    wedgeprops=dict(linewidth=0.8, edgecolor='black')
)

ax.set_title(
    "Distribución semicuantitativa ACFOR\nAnfibios y Reptiles (Total)",
    fontsize=14,
    fontweight='bold'
)

ax.axis('equal')  # círculo perfecto

plt.tight_layout()

# --------------------------------------------------
# 3️⃣ Guardar figura
# --------------------------------------------------
ruta_fig = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados"
fig_path = os.path.join(ruta_fig, "ACFOR_Torta_Total_Herpetofauna.png")

plt.savefig(fig_path, dpi=350, bbox_inches='tight')
plt.show()
plt.close()

print("Gráfico de torta ACFOR generado correctamente:")
print(fig_path)
