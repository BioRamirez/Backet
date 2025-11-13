# ==========================================
# 🦜 Análisis de diversidad y similitud ecológica por coberturas
# Autor: Juan C. Ramírez Gil
# ==========================================

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
Registros = pd.read_excel(ruta)

print("📄 Primeras filas del archivo:")
print(Registros.head())
print("\n📋 Columnas del DataFrame:")
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
        abreviacion = abreviacion.ljust(3, '_')

    return abreviacion.capitalize()


def abreviar_coberturas(df, columna='COBERTURA'):
    """
    Crea un diccionario de abreviaciones y reemplaza los nombres en el DataFrame.
    """
    coberturas_unicas = df[columna].dropna().unique()
    abreviaciones = {c: generar_abreviacion(c) for c in coberturas_unicas}

    print("\n🔤 Abreviaciones generadas automáticamente:")
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

print("\n📊 Tabla de abundancia (Cobertura x Especie):")
print(tabla_abundancia.head())

# --------------------------------------------------
# 4️⃣ Calcular diversidad alfa
# --------------------------------------------------
indices = ['shannon', 'simpson', 'chao1', 'observed_otus']

diversidad_alpha = pd.DataFrame({
    i: alpha_diversity(i, tabla_abundancia.values, ids=tabla_abundancia.index)
    for i in indices
})

print("\n🌱 Diversidad alfa por cobertura:")
print(diversidad_alpha.round(3))

# --------------------------------------------------
# 5️⃣ Calcular disimilitud beta (Bray–Curtis)
# --------------------------------------------------
dist_matrix = beta_diversity('braycurtis', tabla_abundancia.values, ids=tabla_abundancia.index)

print("\n📏 Matriz de disimilitud (Bray–Curtis):")
print(dist_matrix.to_data_frame().round(3))

# --------------------------------------------------
# 6️⃣ Dendrograma jerárquico tipo PAST
# --------------------------------------------------
linkage_matrix = linkage(dist_matrix.condensed_form(), method='average')

plt.figure(figsize=(8, 6))
dendrogram(linkage_matrix, labels=tabla_abundancia.index, leaf_rotation=45)
plt.title("Análisis de similitud entre coberturas (Bray–Curtis)")
plt.ylabel("Distancia (Bray–Curtis)")
plt.tight_layout()

# --- Guardar la figura en PNG ---
plt.savefig(r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\dendrograma_braycurtis.png", dpi=300, bbox_inches='tight')
plt.show()

# --------------------------------------------------
# 7️⃣ Mapa de calor de similitud
# --------------------------------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(1 - dist_matrix.to_data_frame(), cmap="YlGnBu", annot=True)
plt.title("Matriz de similitud (1 - Bray–Curtis)")
plt.tight_layout()

# --- Guardar la figura en PNG ---
plt.savefig(r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\heatmap_similitud_braycurtis.png", dpi=300, bbox_inches='tight')
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


# --- Ejecutar interpretación con tus datos ---
texto_interpretativo = interpretar_diversidad(diversidad_alpha, dist_matrix)
print("\n🧾 Interpretación automática:")
print(texto_interpretativo)

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
plt.savefig(r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\Prioridad_Conservacion.png", dpi=300, bbox_inches='tight')
plt.show()

print("🏆 Ranking de coberturas según diversidad:")
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
plt.xticks(rotation=45, ha='right')  # 🔹 Rota etiquetas si son largas
plt.tight_layout()

plt.savefig(
    r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\Prioridad_Conservacion.png", 
    dpi=300, 
    bbox_inches='tight'
)
plt.show()

print("🏆 Ranking de coberturas según diversidad:")
print(ranking_div[['Índice_compuesto']].round(3))
