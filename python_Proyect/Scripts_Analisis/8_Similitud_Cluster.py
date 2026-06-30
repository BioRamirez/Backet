# ==========================================
#  Análisis de diversidad y similitud ecológica por coberturas
# Autor: Juan C. Ramírez Gil
# ==========================================

import os
import pandas as pd
import numpy as np
import re
from skbio.diversity import alpha_diversity, beta_diversity
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import seaborn as sns


#--------------## Leer archivo y revisar columnas------------------------------
# Ruta del archivo
ruta = r"D:\Forestal Consultores\2026\FAUNA\BD\BD_SANROQUE.xlsx"
# Leer el archivo Excel
Registros = pd.read_excel(ruta)

# Mostrar las primeras filas
print(" Primeras filas del archivo:")
print(Registros.head())

# Mostrar nombres de las columnas
print("\n Columnas del DataFrame:")
print(Registros.columns)

# =========================================================
# FILTRAR UNA SOLA CLASE
# =========================================================

grupo = "ANFIBIOS"   # AVES, MAMIFEROS, REPTILES, etc.

Registros = Registros[
    Registros["CLASE"].astype(str).str.upper() == grupo.upper()
].copy()

# Reiniciar índice
Registros.reset_index(drop=True, inplace=True)

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
# 3️⃣ Crear tabla de abundancia (cobertura × especie)
# --------------------------------------------------
tabla_abundancia = (
    Registros.groupby(['COBERTURA', 'ESPECIE'])['INDIVIDUOS']
    .sum()
    .unstack(fill_value=0)
)

print("\n Tabla de abundancia (Cobertura x Especie):")
print(tabla_abundancia.head())

# --------------------------------------------------
# 4️⃣ Calcular diversidad alfa
# --------------------------------------------------
indices = ['shannon', 'simpson', 'chao1', 'observed_otus']

diversidad_alpha = pd.DataFrame({
    i: alpha_diversity(i, tabla_abundancia.values, ids=tabla_abundancia.index)
    for i in indices
})

print("\n Diversidad alfa por cobertura:")
print(diversidad_alpha.round(3))

# --------------------------------------------------
# 5️⃣ Calcular disimilitud beta (Bray–Curtis)
# --------------------------------------------------
dist_matrix = beta_diversity('braycurtis', tabla_abundancia.values, ids=tabla_abundancia.index)

print("\n Matriz de disimilitud (Bray–Curtis):")
print(dist_matrix.to_data_frame().round(3))

# --------------------------------------------------
# 6️⃣ Dendrograma jerárquico tipo PAST
# --------------------------------------------------
linkage_matrix = linkage(dist_matrix.condensed_form(), method='average')

plt.figure(figsize=(8, 6))
dendrogram(linkage_matrix, labels=tabla_abundancia.index, leaf_rotation=45)
#Análisis de similitud entre coberturas (Bray–Curtis)
plt.title("")
plt.ylabel("Distancia (Bray–Curtis)")
plt.tight_layout()

# --- Guardar la figura en PNG ---
plt.savefig(r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\8_dendrograma_braycurtis.png", dpi=300, bbox_inches='tight')
plt.show()

# --------------------------------------------------
# 7️⃣ Mapa de calor de similitud
# --------------------------------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(1 - dist_matrix.to_data_frame(), cmap="YlGnBu", annot=True)
plt.title("Matriz de similitud (1 - Bray–Curtis)")
plt.tight_layout()

# --- Guardar la figura en PNG ---
plt.savefig(r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\8.1.1_heatmap_similitud_braycurtis.png", dpi=300, bbox_inches='tight')
plt.show()


#----------------------------Fin del codigo----------------------------
#----------------Interpretación de resultados----------------------------
def interpretar_diversidad(diversidad_alpha, dist_matrix):
    interpretacion = []

    # --- Diversidad Alfa ---
    for cobertura, fila in diversidad_alpha.iterrows():
        shannon = fila['shannon']
        simpson = fila['simpson']
        chao1 = fila['chao1']
        obs = fila['observed_otus']

        # Clasificación Shannon
        if shannon < 2:
            nivel = "baja"
        elif 2 <= shannon <= 3.5:
            nivel = "moderada"
        else:
            nivel = "alta"

        representatividad = "alta" if abs(chao1 - obs) / chao1 < 0.1 else "media" if abs(chao1 - obs) / chao1 < 0.3 else "baja"

        texto = (
            f"En la cobertura {cobertura}, la diversidad de Shannon ({shannon:.2f}) indica una diversidad {nivel}, "
            f"mientras que el índice de Simpson ({simpson:.3f}) sugiere una comunidad con alta equidad. "
            f"La riqueza observada ({obs}) y el estimador Chao1 ({chao1:.1f}) muestran una representatividad {representatividad} "
            f"del muestreo."
        )
        interpretacion.append(texto)

    # --- Diversidad Beta ---
    matriz = dist_matrix.to_data_frame()
    pares = []
    for i in range(len(matriz.columns)):
        for j in range(i+1, len(matriz.columns)):
            a, b = matriz.columns[i], matriz.columns[j]
            valor = matriz.iloc[i, j]
            if valor <= 0.33:
                tipo = "alta similitud"
            elif valor <= 0.66:
                tipo = "similitud moderada"
            else:
                tipo = "baja similitud"
            pares.append(f"{a}–{b} ({valor:.3f}: {tipo})")

    texto_beta = (
        "\nEn cuanto a la disimilitud beta (Bray–Curtis), los valores entre coberturas indican los niveles de similitud ecológica: "
        + "; ".join(pares) + "."
    )

    interpretacion.append(texto_beta)

    return "\n".join(interpretacion)


# ---------------------------------------------------------
# Ejecutar interpretación automática y guardar en Resultados
# ---------------------------------------------------------

# 1. Ejecutar interpretación con tus datos
texto_interpretativo = interpretar_diversidad(diversidad_alpha, dist_matrix)

print("\n Interpretación automática:")
print(texto_interpretativo)

# 2. Definir carpeta de salida Resultados
output_folder = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados"

# 3. Nombre del archivo de salida
output_file = os.path.join(output_folder, "8.1.2_Interpr_Diversidad_Similitud_Auto.txt")

# 4. Guardar el archivo TXT
with open(output_file, "w", encoding="utf-8") as f:
    f.write("INTERPRETACIÓN AUTOMÁTICA DE DIVERSIDAD\n")
    f.write("="*70 + "\n\n")
    f.write(texto_interpretativo)

print(f"\n Archivo interpretativo guardado en:\n {output_file}")

#--------------------------------Fin interpretación----------------------------





# ---------------------------------------Prioridad de conservación según diversidad de Fauna---------------------------------------

# --- Calcular un índice combinado (promedio estandarizado de diversidad) ---


pesos = {'shannon': 0.4, 'simpson': 0.2, 'chao1': 0.3, 'observed_otus': 0.1}

# Normalizar sin ceros
diversidad_norm = (diversidad_alpha - diversidad_alpha.min()) / (diversidad_alpha.max() - diversidad_alpha.min())
diversidad_norm = 0.05 + 0.95 * diversidad_norm

# Índice compuesto ponderado
diversidad_norm['Índice_compuesto'] = sum(diversidad_norm[col] * peso for col, peso in pesos.items())
ranking_div = diversidad_norm.sort_values('Índice_compuesto', ascending=False)

print(ranking_div.round(3))


# --- Visualización tipo barra ---
plt.figure(figsize=(9, 5))
sns.barplot(x=ranking_div['Índice_compuesto'], y=ranking_div.index, palette="viridis")
plt.title("Prioridad de conservación según diversidad de Fauna", fontsize=14, fontweight='bold')
plt.xlabel("Índice compuesto de diversidad (0–1)")
plt.ylabel("Cobertura")
plt.tight_layout()
plt.savefig(r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\8.1.3_Prioridad_Conservacion.png", dpi=300, bbox_inches='tight')
plt.show()

print(" Ranking de coberturas según diversidad:")
print(ranking_div[['Índice_compuesto']].round(3))


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(9, 5))
sns.barplot(
    x=ranking_div.index, 
    y=ranking_div['Índice_compuesto'], 
    palette="viridis"
)

plt.title("Prioridad de conservación según diversidad de Fauna", fontsize=14, fontweight='bold')
plt.xlabel("Cobertura")
plt.ylabel("Índice compuesto de diversidad (0–1)")
plt.xticks(rotation=45, ha='right')  #  Rota etiquetas si son largas
plt.tight_layout()

plt.savefig(
    r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\8.1.3_Prioridad_Conservacion.png", 
    dpi=300, 
    bbox_inches='tight'
)
plt.show()

print(" Ranking de coberturas según diversidad:")
print(ranking_div[['Índice_compuesto']].round(3))



#---------------------------INTERPRETACION--------------------

import os

def interpretar_indice_compuesto(ranking_div, pesos):
    """
    Genera una interpretación automática del índice compuesto de diversidad.
    Recibe:
        - ranking_div: dataframe ordenado de mayor a menor
        - pesos: diccionario con los pesos usados
    """

    lineas = []
    lineas.append("===============================================")
    lineas.append(" INTERPRETACIÓN AUTOMÁTICA DEL ÍNDICE COMPUESTO")
    lineas.append("===============================================")
    lineas.append("")

    # ----------------------------------------------------------
    # 1. Identificación general
    # ----------------------------------------------------------
    cobertura_max = ranking_div.index[0]
    valor_max = ranking_div["Índice_compuesto"].iloc[0]

    cobertura_min = ranking_div.index[-1]
    valor_min = ranking_div["Índice_compuesto"].iloc[-1]

    lineas.append("1. Jerarquización general de coberturas")
    lineas.append(f"- La cobertura con mayor valor del índice compuesto es **{cobertura_max}** "
                  f"({valor_max:.3f}), lo que indica una estructura comunitaria más diversa, "
                  "con alta riqueza y baja dominancia.")
    lineas.append(f"- La cobertura con menor valor es **{cobertura_min}** "
                  f"({valor_min:.3f}), reflejando una comunidad más simple o dominada.")
    lineas.append("")

    # ----------------------------------------------------------
    # 2. Interpretación según pesos ecológicos usados
    # ----------------------------------------------------------
    lineas.append("2. Contribución ecológica de los componentes del índice")

    lineas.append("El índice compuesto se construyó ponderando cuatro dimensiones de la diversidad:")

    for k, v in pesos.items():
        lineas.append(f"- {k.capitalize()}: Peso = {v}")

    lineas.append(
        "Estos pesos determinan la importancia relativa de la diversidad efectiva, la equidad, "
        "la riqueza estimada y la riqueza observada para definir la prioridad ecológica."
    )
    lineas.append("")

    # ----------------------------------------------------------
    # 3. Interpretación detallada por cobertura
    # ----------------------------------------------------------
    lineas.append("3. Comportamiento por cobertura (ordenadas de mayor a menor):")
    lineas.append("")

    for cobertura, fila in ranking_div.iterrows():
        valor = fila["Índice_compuesto"]
        lineas.append(f"- {cobertura}: índice = {valor:.3f}.")
    lineas.append("")

    # ----------------------------------------------------------
    # 4. Síntesis ecológica final
    # ----------------------------------------------------------
    lineas.append("Síntesis ecológica general:")
    lineas.append(
        "El índice compuesto permite integrar diversidad, equidad y riqueza estimada en un solo "
        "valor estandarizado. Las coberturas con valores más altos representan núcleos funcionales "
        "de diversidad y deberían priorizarse para conservación. Las coberturas con valores bajos "
        "requieren revisión, pues pueden reflejar dominancia, presión antrópica o baja heterogeneidad. "
        "Este enfoque integrado facilita la toma de decisiones orientada al manejo y conservación del paisaje."
    )
    lineas.append("")

    return "\n".join(lineas)


# ============================================================
#     EJECUTAR Y GUARDAR INTERPRETACIÓN AUTOMÁTICA EN TXT
# ============================================================

interpreta = interpretar_indice_compuesto(ranking_div, pesos)

output_path = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados"
archivo_salida = os.path.join(output_path, "8.1.4_Interpretacion_Indice_Compuesto.txt")

with open(archivo_salida, "w", encoding="utf-8") as f:
    f.write(interpreta)

print("Interpretación automática guardada en:")
print(archivo_salida)

















# --------------------------------------------------
# 6️⃣ Dendrograma jerárquico tipo PAST (Estilo Minimalista Profesional)
# --------------------------------------------------
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# Crear matriz de enlace
linkage_matrix = linkage(
    dist_matrix.condensed_form(),
    method='average'
)

# =============================
#   Configuración de figura
# =============================
plt.figure(figsize=(10, 6))

# Dendrograma al estilo minimalista
dendrogram(
    linkage_matrix,
    labels=tabla_abundancia.index,
    leaf_rotation=45,
    color_threshold=0,          # Todo en un solo color
    above_threshold_color='black'
)

# =============================
#   Estética minimalista
# =============================
plt.title("", fontsize=14)
plt.ylabel("Distancia (Bray–Curtis)", fontsize=12)

# Bordes y estilo limpio
plt.tick_params(axis="x", labelsize=10)
plt.tick_params(axis="y", labelsize=10)

# Quitar bordes superiores y derechos (estilo profesional)
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Línea base más sutil
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

plt.tight_layout()

# Guardar figura en PNG
plt.savefig(
    r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\8_dendrograma_braycurtisPRO.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()






#----------------------------------------------Grafico Paired-----------------
# --------------------------------------------------
# Dendrograma jerárquico tipo PAST con paleta Paired
# --------------------------------------------------
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.cm as cm
import numpy as np

# Crear matriz de enlace
linkage_matrix = linkage(
    dist_matrix.condensed_form(),
    method='average'
)

# =============================
#   Configuración de figura
# =============================
plt.figure(figsize=(10, 6))

# Paleta Paired (12 colores)
cmap = cm.get_cmap("Paired", 12)

# Dendrograma sin colores automáticos
d = dendrogram(
    linkage_matrix,
    labels=tabla_abundancia.index,
    leaf_rotation=45,
    color_threshold=None,   # NO dejar que Matplotlib asigne colores
)

# -------------------------------------------------
#  Asignar color a cada hoja manualmente (Paired)
# -------------------------------------------------
leaf_colors = {}
leaves = d["leaves"]

for i, leaf_id in enumerate(leaves):
    leaf_colors[leaf_id] = cmap(i % 12)  # aplicar paleta Paired ciclada

# Ahora coloreamos manualmente las líneas del dendrograma
ax = plt.gca()

for xs, ys, leaf_ids in zip(d["icoord"], d["dcoord"], d["ivl"]):
    # xs, ys dibujan cada segmento
    # Determinar color usando la posición de la hoja
    left_leaf = int(d["leaves"][int(xs[1] / 10)])
    color = leaf_colors[left_leaf]

    ax.plot(xs, ys, color=color, linewidth=1.8)

# =============================
#   Estética minimalista
# =============================
plt.title("", fontsize=14)
plt.ylabel("Disimilitud (Bray–Curtis)", fontsize=12)

plt.tick_params(axis="x", labelsize=10)
plt.tick_params(axis="y", labelsize=10)

# Quitar bordes superiores y derechos
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

plt.tight_layout()

# Guardar figura
plt.savefig(
    r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\8_dendrograma_braycurtisPRO.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()













# ------------------------------------------------------------
# Dendrograma con índice Jaccard + paleta Paired (FUNCIONAL)
# Estilo Minimalista Profesional
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.cm as cm

# =============================
# 1. Convertir abundancia → PA
# =============================
tabla_pa = (tabla_abundancia > 0).astype(int)

# =============================
# 2. Calcular distancia Jaccard
# =============================
dist_jaccard = pdist(tabla_pa.values, metric="jaccard")

# =============================
# 3. Matriz de enlace (UPGMA)
# =============================
linkage_matrix = linkage(dist_jaccard, method="average")

# =============================
# 4. Crear figura
# =============================
plt.figure(figsize=(10, 6))

# Primero generar dendrograma sin colores automáticos
d = dendrogram(
    linkage_matrix,
    labels=tabla_abundancia.index,
    color_threshold=None,  # evita coloración automática
    above_threshold_color="black",
    leaf_rotation=45
)

# =============================
# 5. Aplicar paleta Paired
# =============================
cmap = cm.get_cmap("Paired", 12)

# Hoja → color
leaves = d["leaves"]
leaf_colors = {leaf: cmap(i % 12) for i, leaf in enumerate(leaves)}

ax = plt.gca()

# Pintar manualmente cada rama del dendrograma
for xs, ys in zip(d["icoord"], d["dcoord"]):
    # Detectar a qué hoja corresponde el segmento
    # Tomamos el punto medio de la parte inferior del segmento
    x_mid = xs[1]
    leaf_index = int(x_mid / 10)
    
    # Asignar color desde Paired
    if leaf_index < len(leaves):
        color = leaf_colors[leaves[leaf_index]]
    else:
        color = "black"
    
    ax.plot(xs, ys, color=color, linewidth=1.8)

# =============================
# 6. Estética minimalista
# =============================
plt.title("", fontsize=14)
plt.ylabel("Disimilitud (Jaccard)", fontsize=12)

plt.tick_params(axis="x", labelsize=10)
plt.tick_params(axis="y", labelsize=10)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

plt.tight_layout()

# =============================
# 7. Guardar
# =============================
plt.savefig(
    r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\8_Dendrograma_Jaccard_Paired.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

























import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform

# ======================================================
# 1. Bray–Curtis (distancias)
# ======================================================
bray_condensed = dist_matrix.condensed_form()

bray_square = pd.DataFrame(
    squareform(bray_condensed),
    index=tabla_abundancia.index,
    columns=tabla_abundancia.index
)

# Asegurar nombre del índice
bray_square.index.name = "Cobertura1"

dist_bray_long = (
    bray_square
    .reset_index()
    .melt(id_vars="Cobertura1",
          var_name="Cobertura2",
          value_name="BrayCurtis")
)

# ======================================================
# 2. Jaccard (presencia–ausencia)
# ======================================================
tabla_pa = (tabla_abundancia > 0).astype(int)

jaccard_condensed = pdist(tabla_pa.values, metric="jaccard")

jaccard_square = pd.DataFrame(
    squareform(jaccard_condensed),
    index=tabla_abundancia.index,
    columns=tabla_abundancia.index
)

# Asegurar nombre del índice
jaccard_square.index.name = "Cobertura1"

dist_jaccard_long = (
    jaccard_square
    .reset_index()
    .melt(id_vars="Cobertura1",
          var_name="Cobertura2",
          value_name="Jaccard")
)

# ======================================================
# 3. Índice de similitud de Jaccard
# ======================================================
similitud_jaccard = 1 - jaccard_square

similitud_jaccard.index.name = "Cobertura1"

similitud_long = (
    similitud_jaccard
    .reset_index()
    .melt(id_vars="Cobertura1",
          var_name="Cobertura2",
          value_name="Similitud")
)

# ======================================================
# 4. Unir todo
# ======================================================
tabla_final = (
    dist_bray_long
    .merge(dist_jaccard_long, on=["Cobertura1", "Cobertura2"])
    .merge(similitud_long, on=["Cobertura1", "Cobertura2"])
)

# ======================================================
# 5. Eliminar autocomparaciones
# ======================================================
tabla_final = tabla_final[tabla_final["Cobertura1"] != tabla_final["Cobertura2"]]

# ======================================================
# 6. Exportar a Excel
# ======================================================
output_path = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\8_Tabla_Similitudes_Distancias.xlsx"
tabla_final.to_excel(output_path, index=False)

print("✔ Tabla generada correctamente en:")
print(output_path)








































import pandas as pd
import random

# ============================================================
# 1. FUNCIONES DE CLASIFICACIÓN SEGÚN LOS RANGOS DEFINIDOS
# ============================================================

def clasificar_bray(valor):
    if valor <= 0.25:
        return "muy alta similitud"
    elif valor <= 0.50:
        return "alta similitud"
    elif valor <= 0.75:
        return "baja similitud"
    else:
        return "muy baja similitud"


def clasificar_jaccard(valor):
    if valor >= 0.75:
        return "muy alta similitud"
    elif valor >= 0.50:
        return "alta similitud"
    elif valor >= 0.25:
        return "baja similitud"
    else:
        return "muy baja similitud"


# ============================================================
# 2. OPCIONES DE REDACCIÓN PARA VARIAR EL TEXTO
# ============================================================

plantillas_texto = {
    "muy alta similitud": [
        "presentan prácticamente el mismo conjunto de especies",
        "muestran ensambles altamente homogéneos",
        "reflejan comunidades casi idénticas en composición",
        "revelan una coincidencia ecológica muy marcada"
    ],
    "alta similitud": [
        "comparten una proporción considerable de especies",
        "poseen comunidades comparables con diferencias menores",
        "mantienen una estructura comunitaria similar",
        "presentan afinidades ecológicas notables"
    ],
    "baja similitud": [
        "exhiben diferencias moderadas en la composición de especies",
        "presentan comunidades parcialmente divergentes",
        "reflejan ensamblajes con coincidencias limitadas",
        "muestran similitud reducida entre sus comunidades"
    ],
    "muy baja similitud": [
        "poseen ensamblajes marcadamente distintos",
        "reflejan una separación ecológica amplia",
        "muestran composiciones altamente divergentes",
        "presentan muy poca coincidencia en especies"
    ]
}


# ============================================================
# 3. FUNCIÓN PARA GENERAR INTERPRETACIÓN AUTOMÁTICA
# ============================================================

def interpretar_fila(fila):
    c1 = fila["Cobertura1"]
    c2 = fila["Cobertura2"]
    bc = fila["BrayCurtis"]
    jc = fila["Jaccard"]

    clase_bray = clasificar_bray(bc)
    clase_jaccard = clasificar_jaccard(jc)

    # Seleccionar una frase al azar según el nivel
    frase = random.choice(plantillas_texto[clase_jaccard])

    # Construcción del texto final
    texto = (
        f"Entre **{c1}** y **{c2}**, la disimilitud Bray–Curtis es de **{bc:.3f}**, "
        f"lo que indica *{clase_bray}*. "
        f"La similitud de Jaccard es **{jc:.3f}**, clasificándose como *{clase_jaccard}*. "
        f"En conjunto, ambas métricas sugieren que las dos coberturas {frase}."
    )
    return texto


# ============================================================
# 4. FUNCIÓN PRINCIPAL PARA APLICAR A TODA LA TABLA
# ============================================================

def generar_interpretaciones(tabla):
    tabla = tabla.copy()
    tabla["Interpretacion"] = tabla.apply(interpretar_fila, axis=1)
    return tabla


# ============================================================
# USO:
# tabla_final = pd.read_excel("TuArchivo.xlsx")
interpretada = generar_interpretaciones(tabla_final)
# interpretada.to_excel("Interpretaciones_Finales.xlsx", index=False)
# ============================================================





































import pandas as pd

# --- CATEGORIZACIONES ---
def cat_disimilitud(valor):
    if valor <= 0.25:
        return "muy baja disimilitud (alta similitud)"
    elif valor <= 0.50:
        return "baja disimilitud (similitud moderada)"
    elif valor <= 0.75:
        return "alta disimilitud (baja similitud)"
    else:
        return "muy alta disimilitud (muy baja similitud)"

def cat_similitud(valor):
    if valor >= 0.75:
        return "muy alta similitud"
    elif valor >= 0.50:
        return "alta similitud"
    elif valor >= 0.25:
        return "baja similitud"
    else:
        return "muy baja similitud"

# --- GENERADOR DE PÁRRAFO TÉCNICO ---
def interpretar_row(row):
    c1 = row["Cobertura1"]
    c2 = row["Cobertura2"]
    bray = row["BrayCurtis"]
    jacc = row["Jaccard"]
    sim = row["Similitud"]

    dis_bray = cat_disimilitud(bray)
    dis_jacc = cat_disimilitud(jacc)
    cat_sim = cat_similitud(sim)

    texto = (
        f"Comparación entre {c1} y {c2}. "
        f"La disimilitud Bray–Curtis es {bray:.3f}, lo que indica {dis_bray}. "
        f"La disimilitud Jaccard es {jacc:.3f}, situándose en el rango de {dis_jacc}, "
        f"mientras que la similitud derivada (S = {sim:.3f}) corresponde a {cat_sim}. "
    )

    # BLOQUES DE INTERPRETACIÓN
    if bray <= 0.25 and jacc <= 0.25:
        texto += ("Ambos índices sugieren una fuerte afinidad ecológica entre las coberturas, "
                  "con alta proporción de especies compartidas y abundancias similares.")
    elif bray > 0.75 and jacc > 0.75:
        texto += ("Los valores elevados de ambas métricas indican ensamblajes marcadamente distintos "
                  "tanto en composición como en estructura poblacional.")
    elif bray <= 0.35 and jacc >= 0.50:
        texto += ("Existe similitud estructural pero diferencias claras en la composición, "
                  "lo que sugiere dominancia compartida de pocas especies y variabilidad en especies raras.")
    elif jacc <= 0.35 and bray >= 0.50:
        texto += ("Las coberturas presentan especies similares en presencia/ausencia, "
                  "pero con diferencias notorias en la abundancia relativa.")
    else:
        texto += ("Los índices presentan concordancia parcial, evidenciando similitud moderada "
                  "con algunas diferencias en composición o estructura que requieren revisión específica.")

    return texto

# --- CONSTRUCCIÓN DEL INFORME ---
def generar_texto(tabla):
    bloques = []
    for _, row in tabla.iterrows():
        bloques.append(interpretar_row(row))
    texto_final = "\n\n".join(bloques)
    return texto_final

# --- USO ---
informe = generar_texto(interpretada)
print(informe)



# Guardar archivo TXT
ruta_salida = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\8.1_Informe_Interpretacion_Similitudes.txt"

with open(ruta_salida, "w", encoding="utf-8") as f:
    f.write(informe)

print("\n✔ Informe técnico generado correctamente en:")
print(ruta_salida)
























































































































