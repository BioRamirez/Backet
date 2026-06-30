
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ============================================================
# 2. INSTALAR GOWER SI ES NECESARIO
# ============================================================

# pip install gower

import gower

# ============================================================
# 3. CONFIGURACIÓN GENERAL
# ============================================================

RUTA_ARCHIVO = r"D:\CORPONOR 2025\CORPONOR_2026\FLORA\DIversisdad Zulia....xlsx"

RUTA_SALIDA = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\Flora"

os.makedirs(RUTA_SALIDA, exist_ok=True)

df = pd.read_excel(RUTA_ARCHIVO)

print("\n====================================")
print("DATOS CARGADOS")
print("====================================")
print(df.head())
print(df.columns)


# ======================================================================
# 🌳 ANALISIS DE DIVERSIDAD DE RECURSOS FORESTALES
# POR COBERTURA VEGETAL
#
# Basado en:
# - Uso forestal de las especies
# - Condición maderable
# - Abundancia relativa
#
# Este análisis NO representa diversidad funcional ecológica,
# sino una aproximación a:
#
# ✔ Diversidad de recursos forestales
# ✔ Potencial de uso forestal
# ✔ Valor forestal relativo
# ✔ Heterogeneidad de aprovechamiento
#
# Autor: ChatGPT
# ======================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist

# ======================================================================
# 📁 CARPETA DE SALIDA
# ======================================================================

salida = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\Flora"

os.makedirs(salida, exist_ok=True)

# ======================================================================
# 📥 CARGAR BASE DE DATOS
# ======================================================================

archivo = r"D:\CORPONOR 2025\CORPONOR_2026\FLORA\DIversisdad Zulia....xlsx"

df = pd.read_excel(archivo)

# ======================================================================
# 🔎 REVISION GENERAL
# ======================================================================

print("\n====================================")
print("DATOS CARGADOS")
print("====================================")

print(df.head())
print(df.columns)

# ======================================================================
# 🌳 COBERTURAS
# ======================================================================

coberturas = [

    'B-denso -alto',
    'B-denso -bajo',
    'B-Fragm-VegtSecund',
    'Bosque de galería y ripario'
]

# ======================================================================
# 🧩 LIMPIEZA DE VARIABLES
# ======================================================================

df["Uso"] = df["Uso"].fillna("No Definido")
df["Maderables"] = df["Maderables"].fillna("No Definido")

# ======================================================================
# ⚖️ PESOS DE USO FORESTAL
# ======================================================================

pesos_uso = {

    # Bajo valor
    "Ninguno": 0,
    "No Definido": 1,

    # Valor moderado
    "Frutas": 2,
    "Frutas/Fibras": 4,
    "Frutas/Látex": 4,
    "Frutas/Resinas": 4,
    "Gomas/Resinas": 4,

    # Valor alto
    "Frutas/Aceites": 4,
    "Frutas/Medicinal": 4,

    # Valor muy alto
    "Medicinal": 5
}

# ======================================================================
# 🪵 PESOS MADERABLES
# ======================================================================

pesos_maderable = {

    "Sí": 1,
    "No": 0,
    "No Definido": 0
}

# ======================================================================
# 🔄 ASIGNACION DE PESOS
# ======================================================================

df["peso_uso"] = df["Uso"].map(pesos_uso)
df["peso_maderable"] = df["Maderables"].map(pesos_maderable)

# Reemplazar valores NA
df["peso_uso"] = df["peso_uso"].fillna(0)
df["peso_maderable"] = df["peso_maderable"].fillna(0)

# ======================================================================
# 📊 MATRIZ GENERAL DE RESULTADOS
# ======================================================================

resultados = []

# ======================================================================
# 🔬 ANALISIS POR COBERTURA
# ======================================================================

for cobertura in coberturas:

    print(f"\nProcesando cobertura: {cobertura}")

    # --------------------------------------------------------------
    # ESPECIES PRESENTES
    # --------------------------------------------------------------

    sub = df[df[cobertura].fillna(0) > 0].copy()

    if len(sub) < 2:

        print("Muy pocas especies")
        continue

    # --------------------------------------------------------------
    # ABUNDANCIA RELATIVA
    # --------------------------------------------------------------

    sub["abund_rel"] = (
        sub[cobertura] /
        sub[cobertura].sum()
    )

    # --------------------------------------------------------------
    # INDICE DE VALOR FORESTAL
    # --------------------------------------------------------------

    sub["IVRF"] = (

        (
            0.7 * sub["peso_uso"] +
            0.3 * sub["peso_maderable"]
        )

        * sub["abund_rel"]

    )

    # --------------------------------------------------------------
    # NORMALIZACION DEL IVRF
    # Escala 0 - 1
    # --------------------------------------------------------------

    scaler_ivrf = MinMaxScaler()

    sub["IVRF_normalizado"] = scaler_ivrf.fit_transform(
        sub[["IVRF"]]
    )

    # --------------------------------------------------------------
    # MATRIZ ANALITICA
    # --------------------------------------------------------------

    matriz = sub[[

        "peso_uso",
        "peso_maderable",
        "abund_rel",
        "IVRF_normalizado"

    ]]

    # --------------------------------------------------------------
    # ESTANDARIZACION
    # --------------------------------------------------------------

    scaler = StandardScaler()

    matriz_scaled = scaler.fit_transform(matriz)

    # --------------------------------------------------------------
    # PCA
    # --------------------------------------------------------------

    pca = PCA(n_components=2)

    coords = pca.fit_transform(matriz_scaled)

    sub["PC1"] = coords[:, 0]
    sub["PC2"] = coords[:, 1]

    # --------------------------------------------------------------
    # INDICE DE HETEROGENEIDAD FORESTAL
    # --------------------------------------------------------------

    distancias = pdist(coords)

    heterogeneidad_forestal = np.mean(distancias)

    # --------------------------------------------------------------
    # NORMALIZAR HETEROGENEIDAD
    # --------------------------------------------------------------

    heterogeneidad_norm = (
        heterogeneidad_forestal /
        (heterogeneidad_forestal + 1)
    )

    # --------------------------------------------------------------
    # ESTADISTICAS
    # --------------------------------------------------------------

    riqueza = len(sub)

    abundancia_total = sub[cobertura].sum()

    ivrf_total = sub["IVRF"].sum()

    ivrf_promedio = sub["IVRF_normalizado"].mean()

    resultados.append({

        "Cobertura": cobertura,

        "Riqueza_especies": riqueza,

        "Abundancia_total": abundancia_total,

        "Heterogeneidad_recursos_forestales":
            heterogeneidad_forestal,

        "Heterogeneidad_normalizada":
            heterogeneidad_norm,

        "IVRF_total":
            ivrf_total,

        "IVRF_promedio_normalizado":
            ivrf_promedio
    })

    # --------------------------------------------------------------
    # EXPORTAR ESPECIES
    # --------------------------------------------------------------

    nombre_archivo = (
        cobertura
        .replace(" ", "_")
        .replace("/", "_")
    )

    sub.to_excel(

        os.path.join(
            salida,
            f"Especies_Recursos_Forestales_{nombre_archivo}.xlsx"
        ),

        index=False
    )

    # --------------------------------------------------------------
    # GRAFICO PCA
    # --------------------------------------------------------------

    plt.figure(figsize=(9, 7))

    plt.scatter(

        sub["PC1"],
        sub["PC2"],
        s=80
    )

    for i, row in sub.iterrows():

        plt.text(

            row["PC1"],
            row["PC2"],
            str(row["ESPECIE"]),
            fontsize=7
        )

    plt.title(
        f"Estructura de recursos forestales\n{cobertura}"
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            salida,
            f"PCA_Recursos_Forestales_{nombre_archivo}.png"
        ),

        dpi=300
    )

    plt.close()

# ======================================================================
# 📋 RESULTADOS FINALES
# ======================================================================

resultados_df = pd.DataFrame(resultados)

# ======================================================================
# 📊 NORMALIZACION FINAL DE INDICES
# ======================================================================

cols_norm = [

    "Heterogeneidad_recursos_forestales",
    "IVRF_total"
]

scaler_final = MinMaxScaler()

resultados_df[[
    "Heterogeneidad_norm_global",
    "IVRF_total_norm"
]] = scaler_final.fit_transform(
    resultados_df[cols_norm]
)

# ======================================================================
# 🖨️ MOSTRAR RESULTADOS
# ======================================================================

print("\n====================================")
print("RESULTADOS DEL ANALISIS")
print("====================================")

print(resultados_df)

# ======================================================================
# 💾 EXPORTAR RESUMEN
# ======================================================================

resultados_df.to_excel(

    os.path.join(
        salida,
        "Resumen_Diversidad_Recursos_Forestales.xlsx"
    ),

    index=False
)

# ======================================================================
# 📈 GRAFICO FINAL
# ======================================================================

plt.figure(figsize=(10, 6))

plt.bar(

    resultados_df["Cobertura"],
    resultados_df["Heterogeneidad_norm_global"]
)

plt.ylabel(
    "Heterogeneidad de recursos forestales"
)

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(

    os.path.join(
        salida,
        "Heterogeneidad_Recursos_Forestales.png"
    ),

    dpi=300
)

plt.close()

# ======================================================================
# 📈 GRAFICO IVRF
# ======================================================================

plt.figure(figsize=(10, 6))

plt.bar(

    resultados_df["Cobertura"],
    resultados_df["IVRF_total_norm"]
)

plt.ylabel(
    "Indice de valor de recursos forestales"
)

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(

    os.path.join(
        salida,
        "IVRF_Coberturas.png"
    ),

    dpi=300
)

plt.close()

# ======================================================================
# ✅ FINAL
# ======================================================================

print("\n====================================")
print("ANALISIS FINALIZADO")
print("====================================")

print(f"Resultados guardados en:\n{salida}")