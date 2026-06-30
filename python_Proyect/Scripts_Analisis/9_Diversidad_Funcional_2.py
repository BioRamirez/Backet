# ============================================================
# PROYECTO:
# ÍNDICE DE VALOR FUNCIONAL Y MULTIFUNCIONALIDAD ECOLÓGICA
# ------------------------------------------------------------
# Autor: Juan Carlos Ramírez Gil
# Objetivo:
# Evaluar la funcionalidad ecológica del paisaje mediante:
#
# 1. Functional Value Index (FVI)
# 2. Rao's Quadratic Entropy (RaoQ)
# 3. Multifunctionality Index (IMF)
# 4. Sensibilidad de pesos ecológicos
# 5. Separación taxonómica (Aves / Mamíferos)
#
# MEJORAS IMPLEMENTADAS:
# ✔ Reemplazo de OrdinalEncoder por distancia Gower
# ✔ Corrección ecológica de pesos funcionales
# ✔ Estandarización por abundancia relativa
# ✔ Separación aves vs mamíferos
# ✔ Implementación de RaoQ
# ✔ Sensibilidad de pesos
# ✔ Reporte metodológico reproducible
#
# ============================================================

# ============================================================
# 1. LIBRERÍAS
# ============================================================

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

RUTA_ARCHIVO = r"D:\CORPONOR 2025\Backet\python_Proyect\data\POF_ZULIA_2025_BD_AVES_MAMIFEROS_F.xlsx"

RUTA_SALIDA = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\FVI_ROBUSTO"

os.makedirs(RUTA_SALIDA, exist_ok=True)

# ============================================================
# 4. CARGAR DATOS
# ============================================================

df = pd.read_excel(RUTA_ARCHIVO)

print("\n====================================")
print("DATOS CARGADOS")
print("====================================")
print(df.head())
print(df.columns)
print(df[['CITES', 'IUCN']].head())
print(df['MUNICIPIO'].unique())
print(df['CITES'].unique())
print(df['Dist_Geo'].unique())

# ============================================================
# 5. NORMALIZAR NOMBRES
# ============================================================

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str).str.strip()

# ============================================================
# 6. ABREVIAR COBERTURAS
# ============================================================

def generar_abreviacion(nombre):

    palabras = nombre.lower().split()

    palabras = [
        p for p in palabras
        if p not in ['de','del','la','el','y','con','en','los','las']
    ]

    abrev = ''.join([p[0] for p in palabras])

    return abrev.capitalize().ljust(3, '_')

df['COBERTURA_ORIGINAL'] = df['COBERTURA']

df['COBERTURA'] = df['COBERTURA'].apply(generar_abreviacion)

# ============================================================
# 7. DEFINIR GRUPO TAXONÓMICO
# ============================================================

# ⚠️ AJUSTAR SEGÚN TU BASE
# Ejemplos:
# Clase, TAXON, GRUPO, etc.

if 'CLASE' in df.columns:

    df['GRUPO'] = np.where(
        df['CLASE'].str.contains("Aves", case=False, na=False),
        'AVES',
        'MAMIFEROS'
    )

else:

    # AJUSTE MANUAL SI NO EXISTE COLUMNA
    df['GRUPO'] = 'AVES'

# ============================================================
# 8. PESOS ECOLÓGICOS CORREGIDOS
# ============================================================

# ------------------------------------------------------------
# FUNDAMENTO:
#
# Valores altos representan:
# - mayor singularidad ecológica
# - mayor vulnerabilidad
# - mayor importancia funcional
#
# Valores bajos representan:
# - generalismo
# - cosmopolitismo
# - baja especialización
#
# ------------------------------------------------------------

peso_gremio = {

    # Control biológico
    'Insectívoro': 7,

    # Dispersión semillas
    'Frugívoro': 8,

    # Polinización
    'Nectarívoro': 8,

    # Reciclaje materia
    'Carroñero': 9,

    # Regulación trófica
    'Carnívoro': 8,

    # Generalistas
    'Omnívoro': 4,

    # Baja especialización
    'Granívoro': 3,

    # Herbivoría
    'Herbivoro': 6
}

peso_migra = {

    # Residentes sostienen procesos permanentes
    'Res': 8,
    'Residentes': 8,

    # Migración regional
    'Lat': 6,
    'Latitudinal': 6,
    'Alt-Loc': 5,
    'Loc': 5,

    # Alta conectividad ecológica
    'Lat-Trans': 7,
    'Lat-Alt-Trans-Loc': 9,

    # Dinámica ecosistémica
    'Nomadismo': 7,
    'Estacional': 6
}

peso_uso = {

    # Sin presión humana
    'Sin uso conocido': 8,

    # Presión moderada
    'Cultural': 5,
    'Uso Cultural': 5,

    # Extracción fuerte
    'Subsistencia': 2,
    'Mascotas': 1,
    'Mascota': 1,

    # Aprovechamiento mixto
    'Medicinal': 4,
    'Otro': 4,

    'Medicinal, Cultural': 4,
    'Mascotas, Subsistencia': 2,
    'Subsistencia, Mascotas': 2,
    'Cultural, Mascotas': 3
}

peso_geo = {

    # Máxima singularidad
    'Endémica': 10,

    # Alta singularidad
    'Casi endémica': 9,

    # Distribución restringida
    'Restringida': 8,

    # Distribución regional
    'Nearctica, Neotropical': 6,

    # Distribución amplia
    'Neotropical': 5,

    # Generalistas
    'Cosmopolita': 2,

    # Introducidas
    'Introducida': 1
}

# ============================================================
# 9. DOCUMENTAR METODOLOGÍA AUTOMÁTICAMENTE
# ============================================================

metodologia = []

metodologia.append("FUNDAMENTO ECOLÓGICO DE LOS PESOS\n")

metodologia.append("GREMIO:")
for k,v in peso_gremio.items():
    metodologia.append(f"{k}: {v}")

metodologia.append("\nMIGRACIÓN:")
for k,v in peso_migra.items():
    metodologia.append(f"{k}: {v}")

metodologia.append("\nUSO:")
for k,v in peso_uso.items():
    metodologia.append(f"{k}: {v}")

metodologia.append("\nBIOGEOGRAFÍA:")
for k,v in peso_geo.items():
    metodologia.append(f"{k}: {v}")

with open(
    os.path.join(RUTA_SALIDA, "METODOLOGIA_PESOS.txt"),
    "w",
    encoding="utf-8"
) as f:

    f.write("\n".join(metodologia))

# ============================================================
# 10. MAPEAR VARIABLES FUNCIONALES
# ============================================================

df['Gremio_valor'] = df['Gremio'].map(peso_gremio).fillna(3)

df['Migracion_valor'] = df['Tipo_Migra'].map(peso_migra).fillna(3)

df['Uso_valor'] = df['Uso'].map(peso_uso).fillna(3)

df['Geo_valor'] = df['Dist_Geo'].map(peso_geo).fillna(3)

# ============================================================
# 11. DEFINIR PESOS DEL ÍNDICE FVI
# ============================================================

# ⚠️ IMPORTANTE:
# Los pesos deben sumar 1

PESOS_FVI = {

    # Función ecosistémica
    'Gremio_valor': 0.35,

    # Conectividad ecológica
    'Migracion_valor': 0.20,

    # Singularidad biogeográfica
    'Geo_valor': 0.35,

    # Presión antropogénica
    'Uso_valor': 0.10
}

# ============================================================
# 12. CALCULAR VALOR FUNCIONAL ESPECIE
# ============================================================

df['Valor_funcional_especie'] = (

    df['Gremio_valor'] * PESOS_FVI['Gremio_valor'] +

    df['Migracion_valor'] * PESOS_FVI['Migracion_valor'] +

    df['Geo_valor'] * PESOS_FVI['Geo_valor'] +

    df['Uso_valor'] * PESOS_FVI['Uso_valor']
)

# ============================================================
# 13. ESTANDARIZAR ABUNDANCIAS
# ============================================================

# ------------------------------------------------------------
# IMPORTANTÍSIMO:
#
# Evita que coberturas con mayor esfuerzo de muestreo
# dominen artificialmente los índices.
#
# ------------------------------------------------------------

total_por_cobertura = (
    df.groupby(['GRUPO','COBERTURA'])['INDIVIDUOS']
    .transform('sum')
)

df['ABUND_REL'] = df['INDIVIDUOS'] / total_por_cobertura

# ============================================================
# 14. FUNCIÓN FVI
# ============================================================

def calcular_FVI(datos):

    datos['FVI_individual'] = (

        datos['Valor_funcional_especie'] *

        datos['ABUND_REL']
    )

    FVI = (

        datos.groupby('COBERTURA')['FVI_individual']
        .sum()
        .sort_values(ascending=False)
    )

    return FVI

# ============================================================
# 15. MATRIZ FUNCIONAL CON GOWER
# ============================================================

def matriz_gower(datos):

    rasgos = [
        'Gremio',
        'Tipo_Migra',
        'Uso',
        'Dist_Geo'
    ]

    rasgos_especies = (
        datos.groupby('ESPECIE')[rasgos]
        .first()
    )

    matriz = gower.gower_matrix(rasgos_especies)

    matriz = pd.DataFrame(
        matriz,
        index=rasgos_especies.index,
        columns=rasgos_especies.index
    )

    return matriz

# ============================================================
# 16. IMPLEMENTAR RAOQ
# ============================================================

# Rao's Quadratic Entropy
#
# Q = ΣΣ dij * pi * pj

def calcular_RaoQ(dist_matrix, abundancias):

    resultados = {}

    for cobertura in abundancias.index:

        abund = abundancias.loc[cobertura]

        abund = abund[abund > 0]

        especies = abund.index

        if len(especies) < 2:

            resultados[cobertura] = 0
            continue

        # abundancia relativa
        p = abund / abund.sum()

        Q = 0

        for i in especies:

            for j in especies:

                dij = dist_matrix.loc[i, j]

                pi = p[i]

                pj = p[j]

                Q += dij * pi * pj

        resultados[cobertura] = Q

    return pd.Series(resultados).sort_values(ascending=False)

# ============================================================
# 17. ÍNDICE MULTIFUNCIONAL (IMF)
# ============================================================

def shannon(p):

    p = p[p > 0]

    return -np.sum(p * np.log(p))

def pielou(H, S):

    if S <= 1:
        return 0

    return H / np.log(S)

def calcular_IMF(datos):

    resultados = {}

    for cobertura in datos['COBERTURA'].unique():

        sub = datos[datos['COBERTURA'] == cobertura]

        prop = (
            sub.groupby('Gremio')['ABUND_REL']
            .sum()
        )

        H = shannon(prop.values)

        S = len(prop)

        J = pielou(H, S)

        IMF = (

            H * 0.4 +

            S * 0.3 +

            J * 0.3
        )

        resultados[cobertura] = IMF

    return pd.Series(resultados).sort_values(ascending=False)

# ============================================================
# 18. VALIDACIÓN DE SENSIBILIDAD DE PESOS
# ============================================================

# ------------------------------------------------------------
# Evalúa si pequeños cambios en pesos modifican drásticamente
# el ranking ecológico.
#
# Si el ranking cambia mucho:
# el índice NO es robusto.
#
# ------------------------------------------------------------

def sensibilidad_pesos(datos, iteraciones=100):

    rankings = []

    for i in range(iteraciones):

        pesos_random = np.random.dirichlet(np.ones(4), size=1)[0]

        datos['VF_temp'] = (

            datos['Gremio_valor'] * pesos_random[0] +

            datos['Migracion_valor'] * pesos_random[1] +

            datos['Geo_valor'] * pesos_random[2] +

            datos['Uso_valor'] * pesos_random[3]
        )

        datos['FVI_temp'] = (
            datos['VF_temp'] *
            datos['ABUND_REL']
        )

        ranking = (

            datos.groupby('COBERTURA')['FVI_temp']
            .sum()
            .sort_values(ascending=False)
            .index.tolist()
        )

        rankings.append(ranking)

    return rankings

# ============================================================
# 19. ANALISIS PRINCIPAL
# ============================================================

for grupo in df['GRUPO'].unique():

    print("\n====================================")
    print(f"GRUPO: {grupo}")
    print("====================================")

    datos_grupo = df[df['GRUPO'] == grupo].copy()

    carpeta = os.path.join(RUTA_SALIDA, grupo)

    os.makedirs(carpeta, exist_ok=True)

    # ========================================================
    # FVI
    # ========================================================

    FVI = calcular_FVI(datos_grupo)

    FVI.to_excel(
        os.path.join(carpeta, f"FVI_{grupo}.xlsx")
    )

    # ========================================================
    # MATRIZ GOWER
    # ========================================================

    matriz = matriz_gower(datos_grupo)

    # ========================================================
    # TABLA ABUNDANCIAS
    # ========================================================

    abund = (

        datos_grupo.groupby(['COBERTURA','ESPECIE'])['INDIVIDUOS']
        .sum()
        .unstack(fill_value=0)
    )

    # ========================================================
    # RAOQ
    # ========================================================

    RaoQ = calcular_RaoQ(matriz, abund)

    RaoQ.to_excel(
        os.path.join(carpeta, f"RaoQ_{grupo}.xlsx")
    )

    # ========================================================
    # IMF
    # ========================================================

    IMF = calcular_IMF(datos_grupo)

    IMF.to_excel(
        os.path.join(carpeta, f"IMF_{grupo}.xlsx")
    )

    # ========================================================
    # SENSIBILIDAD
    # ========================================================

    sensibilidad = sensibilidad_pesos(datos_grupo)

    with open(
        os.path.join(carpeta, f"Sensibilidad_{grupo}.txt"),
        "w",
        encoding="utf-8"
    ) as f:

        for s in sensibilidad[:20]:
            f.write(str(s) + "\n")

    # ========================================================
    # UNIFICAR RESULTADOS
    # ========================================================

    resumen = pd.DataFrame({

        'FVI': FVI,

        'RaoQ': RaoQ,

        'IMF': IMF
    })

    resumen = resumen.sort_values(
        by='FVI',
        ascending=False
    )

    resumen.to_excel(
        os.path.join(carpeta, f"RESUMEN_{grupo}.xlsx")
    )

    print(resumen)

    # ========================================================
    # GRAFICO
    # ========================================================

    plt.figure(figsize=(10,5))

    sns.barplot(
        x=resumen.index,
        y=resumen['FVI']
    )

    plt.title(f"FVI - {grupo}")

    plt.ylabel("Functional Value Index")

    plt.xlabel("Cobertura")

    plt.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()

    plt.savefig(
        os.path.join(carpeta, f"FVI_{grupo}.png"),
        dpi=300
    )

    plt.close()

    # ========================================================
    # INTERPRETACIÓN AUTOMÁTICA
    # ========================================================

    texto = []

    texto.append("====================================")
    texto.append(f"RESULTADOS FUNCIONALES - {grupo}")
    texto.append("====================================\n")

    texto.append("RANKING FVI\n")

    for cov, val in FVI.items():

        texto.append(f"{cov}: {val:.3f}")

    texto.append("\nRANKING RAOQ\n")

    for cov, val in RaoQ.items():

        texto.append(f"{cov}: {val:.3f}")

    texto.append("\nRANKING IMF\n")

    for cov, val in IMF.items():

        texto.append(f"{cov}: {val:.3f}")

    texto.append("\nINTERPRETACIÓN GENERAL\n")

    texto.append(
        "Coberturas con valores altos de FVI "
        "presentan mayor relevancia ecológica funcional."
    )

    texto.append(
        "Valores altos de RaoQ indican "
        "mayor diversidad funcional ponderada por abundancia."
    )

    texto.append(
        "Valores altos de IMF reflejan "
        "alta multifuncionalidad ecológica."
    )

    with open(
        os.path.join(carpeta, f"INTERPRETACION_{grupo}.txt"),
        "w",
        encoding="utf-8"
    ) as f:

        f.write("\n".join(texto))

# ============================================================
# 20. EXPORTAR BASE FINAL
# ============================================================

df.to_excel(
    os.path.join(RUTA_SALIDA, "BASE_FUNCIONAL_PROCESADA.xlsx"),
    index=False
)

print("\n====================================")
print("ANALISIS FINALIZADO")
print("====================================")
print("Resultados guardados en:")
print(RUTA_SALIDA)
print("====================================")







#------------------------------------------------------------------------------------------------------------

# ============================================================
# BLOQUE ADICIONAL
# FD + INTEGRACIÓN TOTAL DE ÍNDICES
# ------------------------------------------------------------
# ESTE BLOQUE NO MODIFICA EL SCRIPT ANTERIOR
# SOLO AGREGA:
#
# ✔ Functional Diversity (FD)
# ✔ Integración FVI + FD + RaoQ + IMF
# ✔ Interpretación ecológica comparativa
# ✔ Correlación entre índices
# ✔ Exportación automática
#
# ============================================================

# ============================================================
# 1. FUNCIÓN FD
# ============================================================

def calcular_FD(dist_matrix, abundancias):

    """
    Functional Diversity (FD)
    Mean Pairwise Functional Distance

    Interpreta:
    - amplitud funcional
    - heterogeneidad ecológica
    - diferencia funcional entre especies
    """

    resultados = {}

    for cobertura in abundancias.index:

        abund = abundancias.loc[cobertura]

        especies = abund[abund > 0].index

        if len(especies) < 2:

            resultados[cobertura] = 0
            continue

        # Submatriz funcional
        sub = dist_matrix.loc[especies, especies]

        # Extraer triángulo superior
        vals = sub.values[np.triu_indices_from(sub, k=1)]

        FD = np.mean(vals)

        resultados[cobertura] = FD

    return pd.Series(resultados).sort_values(ascending=False)

# ============================================================
# 2. NUEVA CARPETA DE RESULTADOS
# ============================================================

RUTA_EXTRA = os.path.join(RUTA_SALIDA, "INDICES_INTEGRADOS")

os.makedirs(RUTA_EXTRA, exist_ok=True)

# ============================================================
# 3. ANALISIS INTEGRADO
# ============================================================

for grupo in df['GRUPO'].unique():

    print("\n================================================")
    print(f"INDICES INTEGRADOS - {grupo}")
    print("================================================")

    datos_grupo = df[df['GRUPO'] == grupo].copy()

    carpeta = os.path.join(RUTA_EXTRA, grupo)

    os.makedirs(carpeta, exist_ok=True)

    # ========================================================
    # MATRIZ FUNCIONAL GOWER
    # ========================================================

    matriz = matriz_gower(datos_grupo)

    # ========================================================
    # TABLA ABUNDANCIAS
    # ========================================================

    abund = (

        datos_grupo
        .groupby(['COBERTURA','ESPECIE'])['INDIVIDUOS']
        .sum()
        .unstack(fill_value=0)
    )

    # ========================================================
    # CALCULAR ÍNDICES
    # ========================================================

    FVI = calcular_FVI(datos_grupo)

    FD = calcular_FD(matriz, abund)

    RaoQ = calcular_RaoQ(matriz, abund)

    IMF = calcular_IMF(datos_grupo)

    # ========================================================
    # UNIFICAR RESULTADOS
    # ========================================================

    resumen = pd.DataFrame({

        'FVI': FVI,

        'FD': FD,

        'RaoQ': RaoQ,

        'IMF': IMF
    })

    resumen = resumen.fillna(0)

    # ========================================================
    # NORMALIZAR ÍNDICES (0-1)
    # ========================================================

    resumen_norm = (

        resumen - resumen.min()

    ) / (

        resumen.max() - resumen.min()
    )

    resumen_norm = resumen_norm.fillna(0)

    # ========================================================
    # ÍNDICE ECOLÓGICO INTEGRADO
    # ========================================================

    resumen_norm['INDICE_INTEGRADO'] = (

        resumen_norm['FVI'] * 0.30 +

        resumen_norm['FD'] * 0.20 +

        resumen_norm['RaoQ'] * 0.30 +

        resumen_norm['IMF'] * 0.20
    )

    resumen_norm = resumen_norm.sort_values(
        by='INDICE_INTEGRADO',
        ascending=False
    )

    # ========================================================
    # EXPORTAR RESULTADOS
    # ========================================================

    resumen.to_excel(
        os.path.join(carpeta, f"INDICES_COMPLETOS_{grupo}.xlsx")
    )

    resumen_norm.to_excel(
        os.path.join(carpeta, f"INDICES_NORMALIZADOS_{grupo}.xlsx")
    )

    # ========================================================
    # CORRELACIÓN ENTRE ÍNDICES
    # ========================================================

    corr = resumen.corr(method='spearman')

    corr.to_excel(
        os.path.join(carpeta, f"CORRELACION_INDICES_{grupo}.xlsx")
    )

    # ========================================================
    # HEATMAP CORRELACIONES
    # ========================================================

    plt.figure(figsize=(7,5))

    sns.heatmap(
        corr,
        annot=True,
        cmap='viridis',
        vmin=-1,
        vmax=1
    )

    plt.title(f"Correlación índices funcionales - {grupo}")

    plt.tight_layout()

    plt.savefig(
        os.path.join(carpeta, f"Heatmap_Correlacion_{grupo}.png"),
        dpi=300
    )

    plt.close()

    # ========================================================
    # GRAFICO INDICE INTEGRADO
    # ========================================================

    plt.figure(figsize=(10,5))

    sns.barplot(
        x=resumen_norm.index,
        y=resumen_norm['INDICE_INTEGRADO']
    )

    plt.title(f"Índice Ecológico Integrado - {grupo}")

    plt.ylabel("Valor normalizado")

    plt.xlabel("Cobertura")

    plt.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()

    plt.savefig(
        os.path.join(carpeta, f"Indice_Integrado_{grupo}.png"),
        dpi=300
    )

    plt.close()

    # ========================================================
    # INTERPRETACIÓN AUTOMÁTICA
    # ========================================================

    texto = []

    texto.append("================================================")
    texto.append(f"INTERPRETACIÓN ECOLÓGICA INTEGRADA - {grupo}")
    texto.append("================================================\n")

    # --------------------------------------------------------
    # FVI
    # --------------------------------------------------------

    texto.append("1. FUNCTIONAL VALUE INDEX (FVI)\n")

    mejor_fvi = FVI.idxmax()

    texto.append(
        f"La cobertura con mayor valor funcional fue "
        f"{mejor_fvi} ({FVI.max():.3f})."
    )

    texto.append(
        "Valores altos indican mayor importancia ecológica "
        "y funcionalidad ecosistémica.\n"
    )

    # --------------------------------------------------------
    # FD
    # --------------------------------------------------------

    texto.append("2. FUNCTIONAL DIVERSITY (FD)\n")

    mejor_fd = FD.idxmax()

    texto.append(
        f"La mayor amplitud funcional se registró en "
        f"{mejor_fd} ({FD.max():.3f})."
    )

    texto.append(
        "FD refleja heterogeneidad ecológica y "
        "diferenciación funcional entre especies.\n"
    )

    # --------------------------------------------------------
    # RAOQ
    # --------------------------------------------------------

    texto.append("3. RAO's QUADRATIC ENTROPY (RaoQ)\n")

    mejor_raoq = RaoQ.idxmax()

    texto.append(
        f"La mayor diversidad funcional ponderada "
        f"se observó en {mejor_raoq} ({RaoQ.max():.3f})."
    )

    texto.append(
        "RaoQ incorpora simultáneamente "
        "abundancia y distancia funcional.\n"
    )

    # --------------------------------------------------------
    # IMF
    # --------------------------------------------------------

    texto.append("4. ÍNDICE MULTIFUNCIONAL (IMF)\n")

    mejor_imf = IMF.idxmax()

    texto.append(
        f"La mayor multifuncionalidad ecológica "
        f"correspondió a {mejor_imf} ({IMF.max():.3f})."
    )

    texto.append(
        "IMF refleja equilibrio funcional, "
        "riqueza y diversidad ecológica.\n"
    )

    # --------------------------------------------------------
    # ÍNDICE INTEGRADO
    # --------------------------------------------------------

    texto.append("5. ÍNDICE ECOLÓGICO INTEGRADO\n")

    mejor_integrado = resumen_norm['INDICE_INTEGRADO'].idxmax()

    texto.append(
        f"La cobertura ecológicamente más relevante "
        f"fue {mejor_integrado} "
        f"({resumen_norm['INDICE_INTEGRADO'].max():.3f})."
    )

    texto.append(
        "Este índice sintetiza valor funcional, "
        "diversidad funcional y multifuncionalidad.\n"
    )

    # --------------------------------------------------------
    # CORRELACIONES
    # --------------------------------------------------------

    texto.append("6. RELACIONES ENTRE ÍNDICES\n")

    for c1 in corr.columns:

        for c2 in corr.columns:

            if c1 != c2:

                r = corr.loc[c1, c2]

                texto.append(
                    f"{c1} vs {c2}: r = {r:.2f}"
                )

    # ========================================================
    # GUARDAR INFORME
    # ========================================================

    with open(
        os.path.join(carpeta, f"INTERPRETACION_INTEGRADA_{grupo}.txt"),
        "w",
        encoding="utf-8"
    ) as f:

        f.write("\n".join(texto))

    print("\nRESUMEN:")
    print(resumen_norm)

# ============================================================
# 4. RESUMEN GENERAL FINAL
# ============================================================

print("\n================================================")
print("ANÁLISIS FUNCIONAL INTEGRADO FINALIZADO")
print("================================================")
print("Índices calculados:")
print("✔ FVI")
print("✔ FD")
print("✔ RaoQ")
print("✔ IMF")
print("✔ Índice Integrado")
print("✔ Correlaciones")
print("✔ Interpretaciones automáticas")
print("================================================")




#---------------------------------------------------------------------------------------------------------------------

# =====================================================================
# 🔬 BLOQUE AVANZADO DE ANALISIS FUNCIONAL
# VERSION ROBUSTA Y AUTOCORREGIBLE
# =====================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# =====================================================================
# 📁 CARPETA DE SALIDA
# =====================================================================

ruta_extra = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\FVI_ROBUSTO\Analisis_Avanzados"

os.makedirs(ruta_extra, exist_ok=True)

# =====================================================================
# 🔎 VERIFICAR DATAFRAME BASE
# =====================================================================

if "df" not in globals():

    raise ValueError(
        "❌ No existe el dataframe 'df' en memoria."
    )

# =====================================================================
# 🔎 DETECTAR COLUMNAS FUNCIONALES EXISTENTES
# =====================================================================

columnas_posibles = [
    "Gremio_valor",
    "Tipo_Migra_valor",
    "Uso_valor",
    "Dist_Geo_valor",
    "Gremio",
    "Tipo_Migra",
    "Uso",
    "Dist_Geo"
]

columnas_disponibles = [
    c for c in columnas_posibles
    if c in df.columns
]

print("\n========================================")
print("COLUMNAS FUNCIONALES DETECTADAS")
print("========================================")
print(columnas_disponibles)

# =====================================================================
# 🔁 ANALISIS POR GRUPO TAXONOMICO
# =====================================================================

for grupo in df["CLASE"].dropna().unique():

    print("\n================================================")
    print(f"ANALISIS AVANZADO: {grupo}")
    print("================================================")

    try:

        datos = df[df["CLASE"] == grupo].copy()

        # =============================================================
        # MATRIZ DE ABUNDANCIA
        # =============================================================

        abund = (
            datos.groupby(
                ["COBERTURA", "ESPECIE"]
            )["INDIVIDUOS"]
            .sum()
            .unstack(fill_value=0)
        )

        # =============================================================
        # MATRIZ FUNCIONAL
        # =============================================================

        rasgos = []

        for col in columnas_disponibles:

            if datos[col].dtype == "object":

                datos[col] = (
                    datos[col]
                    .astype("category")
                    .cat.codes
                )

            rasgos.append(col)

        rasgos_sp = (
            datos.groupby("ESPECIE")[rasgos]
            .mean()
        )

        # =============================================================
        # ESPECIES COMUNES
        # =============================================================

        spp_comunes = abund.columns.intersection(
            rasgos_sp.index
        )

        abund = abund[spp_comunes]

        rasgos_sp = rasgos_sp.loc[spp_comunes]

        # =============================================================
        # ESTANDARIZAR
        # =============================================================

        scaler = StandardScaler()

        rasgos_std = pd.DataFrame(
            scaler.fit_transform(rasgos_sp),
            index=rasgos_sp.index,
            columns=rasgos_sp.columns
        )

        # =============================================================
        # MATRIZ DE DISTANCIA
        # =============================================================

        dist_matrix = pd.DataFrame(
            cdist(
                rasgos_std,
                rasgos_std,
                metric="euclidean"
            ),
            index=rasgos_std.index,
            columns=rasgos_std.index
        )

        # =============================================================
        # 🌿 FDIS
        # =============================================================

        FDis = {}

        centroide = (
            rasgos_std.mean(axis=0)
            .values
            .reshape(1, -1)
        )

        for cob in abund.index:

            spp = abund.loc[cob]

            spp = spp[spp > 0]

            if len(spp) < 2:

                FDis[cob] = 0
                continue

            rasgos_cov = rasgos_std.loc[spp.index]

            distancias = cdist(
                rasgos_cov,
                centroide,
                metric="euclidean"
            ).flatten()

            pesos = spp.values / spp.sum()

            FDis[cob] = np.sum(
                distancias * pesos
            )

        FDis = pd.Series(FDis, name="FDis")

        print("✔ FDis calculado")

        # =============================================================
        # 🌎 SINGULARIDAD FUNCIONAL
        # =============================================================

        singularidad_sp = dist_matrix.mean(axis=1)

        singularidad_cov = {}

        for cob in abund.index:

            spp = abund.loc[cob]

            spp = spp[spp > 0]

            if len(spp) == 0:

                singularidad_cov[cob] = 0
                continue

            pesos = spp / spp.sum()

            singularidad_cov[cob] = np.sum(
                singularidad_sp.loc[spp.index] * pesos
            )

        singularidad_cov = pd.Series(
            singularidad_cov,
            name="Singularidad"
        )

        print("✔ Singularidad funcional calculada")

        # =============================================================
        # ⚠️ REDUNDANCIA
        # =============================================================

        riqueza = (abund > 0).sum(axis=1)

        redundancia = riqueza / (FDis + 0.001)

        redundancia.name = "Redundancia"

        # =============================================================
        # ⚠️ VULNERABILIDAD
        # =============================================================

        vulnerabilidad = 1 / (redundancia + 0.001)

        vulnerabilidad.name = "Vulnerabilidad"

        print("✔ Vulnerabilidad calculada")

        # =============================================================
        # 📊 RESUMEN
        # =============================================================

        resumen = pd.concat([
            FDis,
            singularidad_cov,
            redundancia,
            vulnerabilidad
        ], axis=1)

        print("\nRESUMEN:")
        print(resumen.round(4))

        # =============================================================
        # 💾 EXPORTAR
        # =============================================================

        resumen.to_excel(
            os.path.join(
                ruta_extra,
                f"Resumen_Avanzado_{grupo}.xlsx"
            )
        )

        # =============================================================
        # 📈 PCA
        # =============================================================

        scaler2 = StandardScaler()

        X = scaler2.fit_transform(
            resumen.fillna(0)
        )

        if len(resumen) >= 2:

            pca = PCA(n_components=2)

            coords = pca.fit_transform(X)

            pca_df = pd.DataFrame(
                coords,
                columns=["PC1", "PC2"],
                index=resumen.index
            )

            # ---------------------------------------------------------

            plt.figure(figsize=(7, 6))

            plt.scatter(
                pca_df["PC1"],
                pca_df["PC2"]
            )

            for idx in pca_df.index:

                plt.text(
                    pca_df.loc[idx, "PC1"],
                    pca_df.loc[idx, "PC2"],
                    idx
                )

            plt.title(
                f"PCA Funcional - {grupo}"
            )

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    ruta_extra,
                    f"PCA_{grupo}.png"
                ),
                dpi=300
            )

            plt.close()

            print("✔ PCA exportado")

        # =============================================================
        # 📉 SENSIBILIDAD DE PESOS
        # =============================================================

        sensibilidad = []

        if "Gremio_valor" in datos.columns:

            pesos_test = np.linspace(0.5, 2, 20)

            for p in pesos_test:

                datos["TEMP"] = (
                    datos["Gremio_valor"] * p
                )

                temp = (
                    datos.groupby("COBERTURA")["TEMP"]
                    .mean()
                )

                for cob, val in temp.items():

                    sensibilidad.append([
                        cob,
                        p,
                        val
                    ])

            sensibilidad = pd.DataFrame(
                sensibilidad,
                columns=[
                    "COBERTURA",
                    "Peso",
                    "Valor"
                ]
            )

            sensibilidad.to_excel(
                os.path.join(
                    ruta_extra,
                    f"Sensibilidad_{grupo}.xlsx"
                ),
                index=False
            )

            print("✔ Sensibilidad calculada")

        # =============================================================
        # 📝 INTERPRETACION
        # =============================================================

        texto = []

        texto.append(
            f"ANALISIS FUNCIONAL AVANZADO - {grupo}\n"
        )

        texto.append(
            "\nRESUMEN DE INDICES:\n"
        )

        texto.append(
            str(resumen.round(4))
        )

        texto.append("\n")

        texto.append(
            f"Mayor FDis: {FDis.idxmax()}"
        )

        texto.append(
            f"\nMayor singularidad: "
            f"{singularidad_cov.idxmax()}"
        )

        texto.append(
            f"\nMayor vulnerabilidad: "
            f"{vulnerabilidad.idxmax()}"
        )

        with open(

            os.path.join(
                ruta_extra,
                f"Interpretacion_{grupo}.txt"
            ),

            "w",

            encoding="utf-8"

        ) as f:

            f.write("\n".join(texto))

        print("✔ Interpretación exportada")

    except Exception as e:

        print(f"\n❌ ERROR EN {grupo}")
        print(e)

# =====================================================================
print("\n================================================")
print("ANALISIS AVANZADOS FINALIZADOS")
print("================================================")
print(ruta_extra)
print("================================================")

#------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------
# ======================================================================
# 🌿 BLOQUE FINAL
# MATRIZ ECOFUNCIONAL PARA COBERTURAS
# ======================================================================

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

# ======================================================================
# 📁 RUTA BASE
# ======================================================================

ruta_base = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\FVI_ROBUSTO"

# ======================================================================
# 📁 CARPETA SALIDA
# ======================================================================

ruta_matriz = os.path.join(
    ruta_base,
    "MATRIZ_ECOFUNCIONAL"
)

os.makedirs(ruta_matriz, exist_ok=True)

# ======================================================================
# 📌 CLASIFICACION ECOFUNCIONAL
# ======================================================================

def clasificar_indice(valor):

    if valor >= 0.80:
        return "MUY ALTA"

    elif valor >= 0.60:
        return "ALTA"

    elif valor >= 0.40:
        return "MEDIA"

    elif valor >= 0.20:
        return "BAJA"

    else:
        return "MUY BAJA"

# ======================================================================
# 📌 PRIORIDAD DE MANEJO
# ======================================================================

def prioridad(valor):

    if valor >= 0.80:
        return "CONSERVACION PRIORITARIA"

    elif valor >= 0.60:
        return "MANEJO SOSTENIBLE"

    elif valor >= 0.40:
        return "RESTAURACION PARCIAL"

    else:
        return "RESTAURACION PRIORITARIA"

# ======================================================================
# 🔁 ANALISIS POR GRUPO
# ======================================================================

for grupo in df["CLASE"].dropna().unique():

    print("\n================================================")
    print(f"MATRIZ ECOFUNCIONAL - {grupo.upper()}")
    print("================================================")

    try:

        # ==============================================================
        # 🏷️ NORMALIZAR NOMBRE
        # ==============================================================

        grupo_upper = str(grupo).upper()

        # ==============================================================
        # 🐦 AVES
        # ==============================================================

        if grupo_upper == "AVES":

            archivo_indices = os.path.join(
                ruta_base,
                "INDICES_INTEGRADOS",
                "AVES",
                "INDICES_NORMALIZADOS_AVES.xlsx"
            )

            archivo_avanzado = os.path.join(
                ruta_base,
                "Analisis_Avanzados",
                "Resumen_Avanzado_Aves.xlsx"
            )

            nombre_exportacion = "AVES"

        # ==============================================================
        # 🐾 MAMIFEROS
        # ==============================================================

        elif grupo_upper in ["MAMMALIA", "MAMIFEROS"]:

            archivo_indices = os.path.join(
                ruta_base,
                "INDICES_INTEGRADOS",
                "MAMIFEROS",
                "INDICES_NORMALIZADOS_MAMIFEROS.xlsx"
            )

            archivo_avanzado = os.path.join(
                ruta_base,
                "Analisis_Avanzados",
                "Resumen_Avanzado_Mammalia.xlsx"
            )

            nombre_exportacion = "MAMIFEROS"

        else:

            print(f"⚠ Grupo no reconocido: {grupo}")
            continue

        # ==============================================================
        # 📂 VALIDAR ARCHIVOS
        # ==============================================================

        if not os.path.exists(archivo_indices):

            raise FileNotFoundError(
                f"No existe:\n{archivo_indices}"
            )

        if not os.path.exists(archivo_avanzado):

            raise FileNotFoundError(
                f"No existe:\n{archivo_avanzado}"
            )

        # ==============================================================
        # 📖 LEER ARCHIVOS
        # ==============================================================

        indices_base = pd.read_excel(
            archivo_indices,
            index_col=0
        )

        indices_av = pd.read_excel(
            archivo_avanzado,
            index_col=0
        )

        # ==============================================================
        # 🔗 UNIR MATRICES
        # ==============================================================

        matriz = pd.concat(
            [
                indices_base,
                indices_av
            ],
            axis=1
        )

        # ==============================================================
        # 🧹 ELIMINAR COLUMNAS DUPLICADAS
        # ==============================================================

        matriz = matriz.loc[
            :,
            ~matriz.columns.duplicated()
        ]

        # ==============================================================
        # 🔥 VARIABLES NUMERICAS
        # ==============================================================

        cols_numericas = matriz.select_dtypes(
            include=np.number
        ).columns

        # ==============================================================
        # 🔥 NORMALIZACION
        # ==============================================================

        scaler = MinMaxScaler()

        matriz_norm = matriz.copy()

        matriz_norm[cols_numericas] = scaler.fit_transform(
            matriz[cols_numericas]
        )

        # ==============================================================
        # ⚠ VARIABLES NEGATIVAS
        # ==============================================================

        if "Vulnerabilidad" in matriz_norm.columns:

            matriz_norm["Vulnerabilidad"] = (
                1 - matriz_norm["Vulnerabilidad"]
            )

        # ==============================================================
        # 🌿 PESOS ECOFUNCIONALES
        # ==============================================================

        pesos = {

            "FVI": 0.20,
            "FD": 0.15,
            "RaoQ": 0.15,
            "IMF": 0.15,
            "FDis": 0.10,
            "Singularidad": 0.10,
            "Redundancia": 0.10,
            "Vulnerabilidad": 0.05

        }

        # ==============================================================
        # ✔ VARIABLES EXISTENTES
        # ==============================================================

        pesos_validos = {

            k: v for k, v in pesos.items()
            if k in matriz_norm.columns

        }

        # ==============================================================
        # 🌎 INDICE ECOFUNCIONAL
        # ==============================================================

        matriz_norm["INDICE_ECOFUNCIONAL"] = 0

        for col, peso in pesos_validos.items():

            matriz_norm["INDICE_ECOFUNCIONAL"] += (

                matriz_norm[col] * peso

            )

        # ==============================================================
        # 🏆 RANKING
        # ==============================================================

        matriz_norm["RANKING"] = (

            matriz_norm["INDICE_ECOFUNCIONAL"]
            .rank(
                ascending=False,
                method="dense"
            )
            .astype(int)

        )

        # ==============================================================
        # 🌎 CLASIFICACION
        # ==============================================================

        matriz_norm["CATEGORIA_FUNCIONAL"] = (

            matriz_norm["INDICE_ECOFUNCIONAL"]
            .apply(clasificar_indice)

        )

        # ==============================================================
        # 🚨 PRIORIDAD
        # ==============================================================

        matriz_norm["PRIORIDAD_MANEJO"] = (

            matriz_norm["INDICE_ECOFUNCIONAL"]
            .apply(prioridad)

        )

        # ==============================================================
        # 📊 ORDENAR
        # ==============================================================

        matriz_norm = matriz_norm.sort_values(
            by="INDICE_ECOFUNCIONAL",
            ascending=False
        )

        # ==============================================================
        # 🖨 MOSTRAR RESULTADOS
        # ==============================================================

        print("\nRANKING ECOFUNCIONAL:\n")

        print(

            matriz_norm[
                [
                    "INDICE_ECOFUNCIONAL",
                    "CATEGORIA_FUNCIONAL",
                    "PRIORIDAD_MANEJO",
                    "RANKING"
                ]
            ]
            .round(4)

        )

        # ==============================================================
        # 💾 EXPORTAR MATRIZ
        # ==============================================================

        salida_excel = os.path.join(
            ruta_matriz,
            f"Matriz_Ecofuncional_{nombre_exportacion}.xlsx"
        )

        matriz_norm.to_excel(
            salida_excel
        )

        # ==============================================================
        # 🌍 EXPORTAR CSV SIG
        # ==============================================================

        tabla_sig = matriz_norm.reset_index()

        tabla_sig.rename(
            columns={"index": "COBERTURA"},
            inplace=True
        )

        salida_csv = os.path.join(
            ruta_matriz,
            f"Tabla_SIG_{nombre_exportacion}.csv"
        )

        tabla_sig.to_csv(
            salida_csv,
            index=False,
            encoding="utf-8-sig"
        )

        # ==============================================================
        # 📘 INTERPRETACION
        # ==============================================================

        texto = []

        texto.append(
            f"===== MATRIZ ECOFUNCIONAL - {nombre_exportacion} =====\n"
        )

        texto.append(
            "ESCALA INTERPRETATIVA:"
        )

        texto.append(
            "0.0–0.2 = Muy baja funcionalidad ecológica"
        )

        texto.append(
            "0.2–0.4 = Baja funcionalidad ecológica"
        )

        texto.append(
            "0.4–0.6 = Funcionalidad ecológica media"
        )

        texto.append(
            "0.6–0.8 = Alta funcionalidad ecológica"
        )

        texto.append(
            "0.8–1.0 = Muy alta funcionalidad ecológica"
        )

        texto.append("\n")

        mejor = matriz_norm.index[0]

        texto.append(
            f"La cobertura con mayor funcionalidad fue: {mejor}"
        )

        salida_txt = os.path.join(
            ruta_matriz,
            f"Interpretacion_Ecofuncional_{nombre_exportacion}.txt"
        )

        with open(
            salida_txt,
            "w",
            encoding="utf-8"
        ) as f:

            f.write("\n".join(texto))

        print("\n✔ MATRIZ ECOFUNCIONAL EXPORTADA")

    except Exception as e:

        print(f"\n❌ ERROR EN {grupo}")
        print(e)

# ======================================================================
# ✅ FINAL
# ======================================================================

print("\n================================================")
print("🌿 MATRICES ECOFUNCIONALES FINALIZADAS")
print("================================================")
print(ruta_matriz)
print("================================================")


#---------------------------------------------------------------------------------------------------------------------------

# ============================================================
# 🔍 CONSULTAR TODOS LOS ARCHIVOS GENERADOS
# ============================================================

import os

ruta_base = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\FVI_ROBUSTO"

print("\n================================================")
print("📂 ARCHIVOS GENERADOS EN FVI_ROBUSTO")
print("================================================")

for raiz, carpetas, archivos in os.walk(ruta_base):

    nivel = raiz.replace(ruta_base, "").count(os.sep)

    sangria = " " * 4 * nivel

    print(f"\n{sangria}📁 {os.path.basename(raiz)}")

    sub_sangria = " " * 4 * (nivel + 1)

    for archivo in archivos:

        ruta_completa = os.path.join(raiz, archivo)

        print(f"{sub_sangria}📄 {archivo}")

print("\n================================================")
print("✅ CONSULTA FINALIZADA")
print("================================================")


#---------------------------------------------------------------------------------------------------------------------------
# ======================================================================
# 🌎 ANALISIS ECOFUNCIONAL POR MUNICIPIO
# ======================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================================
# 📁 CARPETA BASE
# ======================================================================

RUTA_MUNICIPIOS = os.path.join(
    RUTA_SALIDA,
    "ANALISIS_MUNICIPIOS"
)

os.makedirs(RUTA_MUNICIPIOS, exist_ok=True)

# ======================================================================
# 🔎 VALIDAR COLUMNA MUNICIPIO
# ======================================================================

if "MUNICIPIO" not in df.columns:

    raise ValueError(
        "❌ No existe la columna MUNICIPIO"
    )

# ======================================================================
# 🧹 NORMALIZAR MUNICIPIOS
# ======================================================================

df["MUNICIPIO"] = (

    df["MUNICIPIO"]
    .astype(str)
    .str.upper()
    .str.strip()
)

# ======================================================================
# 📌 LISTA MUNICIPIOS
# ======================================================================

municipios = sorted(
    df["MUNICIPIO"]
    .dropna()
    .unique()
)

print("\n================================================")
print("🌎 MUNICIPIOS DETECTADOS")
print("================================================")
print(municipios)

# ======================================================================
# 🔁 LOOP PRINCIPAL
# ======================================================================

for municipio in municipios:

    print("\n================================================")
    print(f"🌿 ANALISIS MUNICIPIO: {municipio}")
    print("================================================")

    try:

        # ==============================================================
        # 📁 CARPETA MUNICIPIO
        # ==============================================================

        carpeta_municipio = os.path.join(
            RUTA_MUNICIPIOS,
            municipio.replace(" ", "_")
        )

        os.makedirs(carpeta_municipio, exist_ok=True)

        # ==============================================================
        # 📌 FILTRAR MUNICIPIO
        # ==============================================================

        df_mun = df[
            df["MUNICIPIO"] == municipio
        ].copy()

        # ==============================================================
        # 🔁 ANALISIS POR GRUPO
        # ==============================================================

        for grupo in df_mun["GRUPO"].dropna().unique():

            print(f"\n➡ GRUPO: {grupo}")

            # ----------------------------------------------------------
            # 📁 CARPETA GRUPO
            # ----------------------------------------------------------

            carpeta_grupo = os.path.join(
                carpeta_municipio,
                grupo
            )

            os.makedirs(carpeta_grupo, exist_ok=True)

            # ----------------------------------------------------------
            # 📌 DATOS GRUPO
            # ----------------------------------------------------------

            datos = df_mun[
                df_mun["GRUPO"] == grupo
            ].copy()

            # ----------------------------------------------------------
            # ⚠ VALIDAR DATOS
            # ----------------------------------------------------------

            if len(datos) == 0:

                print(f"⚠ Sin datos para {grupo}")
                continue

            # ----------------------------------------------------------
            # 🌿 FVI
            # ----------------------------------------------------------

            FVI = calcular_FVI(datos)

            FVI.to_excel(
                os.path.join(
                    carpeta_grupo,
                    f"FVI_{grupo}.xlsx"
                )
            )

            # ----------------------------------------------------------
            # 🌿 MATRIZ GOWER
            # ----------------------------------------------------------

            matriz = matriz_gower(datos)

            # ----------------------------------------------------------
            # 🌿 ABUNDANCIAS
            # ----------------------------------------------------------

            abund = (

                datos.groupby(
                    ["COBERTURA", "ESPECIE"]
                )["INDIVIDUOS"]
                .sum()
                .unstack(fill_value=0)

            )

            # ----------------------------------------------------------
            # 🌿 RAOQ
            # ----------------------------------------------------------

            RaoQ = calcular_RaoQ(
                matriz,
                abund
            )

            RaoQ.to_excel(
                os.path.join(
                    carpeta_grupo,
                    f"RaoQ_{grupo}.xlsx"
                )
            )

            # ----------------------------------------------------------
            # 🌿 FD
            # ----------------------------------------------------------

            FD = calcular_FD(
                matriz,
                abund
            )

            FD.to_excel(
                os.path.join(
                    carpeta_grupo,
                    f"FD_{grupo}.xlsx"
                )
            )

            # ----------------------------------------------------------
            # 🌿 IMF
            # ----------------------------------------------------------

            IMF = calcular_IMF(datos)

            IMF.to_excel(
                os.path.join(
                    carpeta_grupo,
                    f"IMF_{grupo}.xlsx"
                )
            )

            # ----------------------------------------------------------
            # 🌿 RESUMEN
            # ----------------------------------------------------------

            resumen = pd.DataFrame({

                "FVI": FVI,
                "FD": FD,
                "RaoQ": RaoQ,
                "IMF": IMF

            }).fillna(0)

            # ----------------------------------------------------------
            # 🌿 NORMALIZAR
            # ----------------------------------------------------------

            resumen_norm = (

                resumen - resumen.min()

            ) / (

                resumen.max() - resumen.min()
            )

            resumen_norm = resumen_norm.fillna(0)

            # ----------------------------------------------------------
            # 🌿 INDICE INTEGRADO
            # ----------------------------------------------------------

            resumen_norm["INDICE_INTEGRADO"] = (

                resumen_norm["FVI"] * 0.30 +

                resumen_norm["FD"] * 0.20 +

                resumen_norm["RaoQ"] * 0.30 +

                resumen_norm["IMF"] * 0.20

            )

            # ----------------------------------------------------------
            # 🌿 RANKING
            # ----------------------------------------------------------

            resumen_norm = resumen_norm.sort_values(
                by="INDICE_INTEGRADO",
                ascending=False
            )

            resumen_norm["RANKING"] = range(
                1,
                len(resumen_norm) + 1
            )

            # ----------------------------------------------------------
            # 💾 EXPORTAR
            # ----------------------------------------------------------

            resumen.to_excel(
                os.path.join(
                    carpeta_grupo,
                    f"INDICES_{grupo}.xlsx"
                )
            )

            resumen_norm.to_excel(
                os.path.join(
                    carpeta_grupo,
                    f"INDICES_NORMALIZADOS_{grupo}.xlsx"
                )
            )

            # ----------------------------------------------------------
            # 🌿 CSV SIG
            # ----------------------------------------------------------

            tabla_sig = resumen_norm.reset_index()

            tabla_sig.rename(
                columns={"index": "COBERTURA"},
                inplace=True
            )

            tabla_sig.to_csv(

                os.path.join(
                    carpeta_grupo,
                    f"SIG_{grupo}.csv"
                ),

                index=False,
                encoding="utf-8-sig"
            )

            # ----------------------------------------------------------
            # 📊 GRAFICO
            # ----------------------------------------------------------

            plt.figure(figsize=(10,5))

            sns.barplot(
                x=resumen_norm.index,
                y=resumen_norm["INDICE_INTEGRADO"]
            )

            plt.title(
                f"{municipio} - {grupo}"
            )

            plt.ylabel(
                "Índice Integrado"
            )

            plt.xlabel(
                "Cobertura"
            )

            plt.xticks(rotation=45)

            plt.grid(
                axis='y',
                linestyle='--',
                alpha=0.4
            )

            plt.tight_layout()

            plt.savefig(

                os.path.join(
                    carpeta_grupo,
                    f"INDICE_INTEGRADO_{grupo}.png"
                ),

                dpi=300
            )

            plt.close()

            # ----------------------------------------------------------
            # 📘 INTERPRETACION
            # ----------------------------------------------------------

            mejor = resumen_norm.index[0]

            texto = []

            texto.append(
                f"ANALISIS ECOFUNCIONAL - {municipio}"
            )

            texto.append(
                f"GRUPO: {grupo}\n"
            )

            texto.append(
                f"Cobertura con mayor funcionalidad:"
            )

            texto.append(
                f"{mejor}"
            )

            texto.append("\n")

            texto.append(
                "RANKING ECOFUNCIONAL:\n"
            )

            for idx, row in resumen_norm.iterrows():

                texto.append(

                    f"{idx}: "
                    f"{row['INDICE_INTEGRADO']:.3f}"

                )

            with open(

                os.path.join(
                    carpeta_grupo,
                    f"INTERPRETACION_{grupo}.txt"
                ),

                "w",

                encoding="utf-8"

            ) as f:

                f.write("\n".join(texto))

            print(f"✔ Analisis completado: {grupo}")

        print(f"\n✅ MUNICIPIO FINALIZADO: {municipio}")

    except Exception as e:

        print(f"\n❌ ERROR EN MUNICIPIO {municipio}")
        print(e)

# ======================================================================
# ✅ FINAL
# ======================================================================

print("\n================================================")
print("🌎 ANALISIS MUNICIPALES FINALIZADOS")
print("================================================")
print(RUTA_MUNICIPIOS)
print("================================================")



######DISPERSION DE SEMILLAS - TABLAS POR MUNICIPIO Y GRUPO
#---------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import os

# ============================================================
# RUTA DE SALIDA
# ============================================================

ruta_salida = os.path.join(RUTA_SALIDA, "DISPERSION_SEMILLAS")
os.makedirs(ruta_salida, exist_ok=True)

# ============================================================
# FILTRO DE DISPERSORES
# ============================================================

dispersores_validos = ["Frugívoro", "Omnívoro", "Herbívoro"]
df_disp = df[df["Gremio"].isin(dispersores_validos)].copy()

# ============================================================
# 1. GLOBAL (TODAS LAS ESPECIES)
# ============================================================

tabla_global = (
    df_disp.groupby(["ESPECIE", "GRUPO"])["INDIVIDUOS"]
    .sum()
    .reset_index()
    .rename(columns={"INDIVIDUOS": "INDIVIDUOS"})
    .sort_values("INDIVIDUOS", ascending=False)
)

tabla_global.to_excel(
    os.path.join(ruta_salida, "DISPERSORES_GLOBAL.xlsx"),
    index=False
)

# ============================================================
# 2. POR MUNICIPIO (TABLA COMPLETA)
# ============================================================

tabla_municipios = (
    df_disp.groupby(["MUNICIPIO", "ESPECIE", "GRUPO"])["INDIVIDUOS"]
    .sum()
    .reset_index()
    .sort_values(["MUNICIPIO", "INDIVIDUOS"], ascending=[True, False])
)

tabla_municipios.to_excel(
    os.path.join(ruta_salida, "DISPERSORES_POR_MUNICIPIO.xlsx"),
    index=False
)

# ============================================================
# 3. CARPETAS POR MUNICIPIO
# ============================================================

for mun in df_disp["MUNICIPIO"].unique():

    sub_mun = df_disp[df_disp["MUNICIPIO"] == mun]

    ruta_mun = os.path.join(ruta_salida, mun)
    os.makedirs(ruta_mun, exist_ok=True)

    # --------------------------------------------------------
    # Aves
    # --------------------------------------------------------
    aves = (
        sub_mun[sub_mun["GRUPO"] == "AVES"]
        .groupby(["ESPECIE", "GRUPO"])["INDIVIDUOS"]
        .sum()
        .reset_index()
        .sort_values("INDIVIDUOS", ascending=False)
    )

    aves.to_excel(
        os.path.join(ruta_mun, "DISPERSORES_AVES.xlsx"),
        index=False
    )

    # --------------------------------------------------------
    # Mamíferos
    # --------------------------------------------------------
    mam = (
        sub_mun[sub_mun["GRUPO"] == "MAMIFEROS"]
        .groupby(["ESPECIE", "GRUPO"])["INDIVIDUOS"]
        .sum()
        .reset_index()
        .sort_values("INDIVIDUOS", ascending=False)
    )

    mam.to_excel(
        os.path.join(ruta_mun, "DISPERSORES_MAMIFEROS.xlsx"),
        index=False
    )

    # --------------------------------------------------------
    # TOP 10 GENERAL MUNICIPIO
    # --------------------------------------------------------
    top10 = (
        sub_mun.groupby(["ESPECIE", "GRUPO"])["INDIVIDUOS"]
        .sum()
        .reset_index()
        .sort_values("INDIVIDUOS", ascending=False)
        .head(10)
    )

    top10.to_excel(
        os.path.join(ruta_mun, "TOP10_DISPERSORES_MUNICIPIO.xlsx"),
        index=False
    )

# ============================================================
# MENSAJE FINAL
# ============================================================

print("================================================")
print("ANÁLISIS DE DISPERSIÓN FINALIZADO")
print("================================================")
print("✔ DISPERSORES_GLOBAL.xlsx")
print("✔ DISPERSORES_POR_MUNICIPIO.xlsx")
print("✔ Carpetas por municipio creadas")
print("   - DISPERSORES_AVES.xlsx")
print("   - DISPERSORES_MAMIFEROS.xlsx")
print("   - TOP10_DISPERSORES_MUNICIPIO.xlsx")
print("================================================")


#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# ============================================================
# FILTRO DISPERSORES
# ============================================================

dispersores_validos = ["Frugívoro", "Omnívoro"]
df_disp = df[df["Gremio"].isin(dispersores_validos)].copy()

# ============================================================
# RUTA BASE
# ============================================================

ruta_base = os.path.join(RUTA_SALIDA, "DISPERSION_SEMILLAS")
ruta_graficos = os.path.join(ruta_base, "GRAFICOS_RANGO_ABUNDANCIA")
os.makedirs(ruta_graficos, exist_ok=True)

# ============================================================
# FUNCIÓN GRÁFICO RANGO–ABUNDANCIA
# ============================================================

def grafico_rango_abundancia(mun, data, ruta_salida):

    abund = (
        data.groupby("ESPECIE")["INDIVIDUOS"]
        .sum()
        .sort_values(ascending=False)
    )

    if len(abund) == 0:
        return

    especies = abund.index.tolist()
    valores = abund.values
    x = np.arange(1, len(especies) + 1)

    plt.figure(figsize=(10, 6))

    # curva general
    plt.plot(
        x,
        valores,
        marker="o",
        linewidth=1.5,
        color="gray",
        alpha=0.7
    )

    # ========================================================
    # TOP 3 ESPECIES
    # ========================================================

    top3 = abund.head(3)

    # offsets para evitar solapamiento
    y_offset = [10, 0, -10]

    colores = ["red", "blue", "green"]

    for i, (sp, val) in enumerate(top3.items()):

        idx = especies.index(sp) + 1

        plt.scatter(
            idx,
            val,
            s=120,
            color=colores[i]
        )

        plt.annotate(
            sp,
            xy=(idx, val),
            xytext=(12, y_offset[i]),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.7)
        )

    # ============================================================
    # FORMATO
    # ============================================================

    plt.title(f"Curva Rango–Abundancia de Dispersores\n{mun}")
    plt.xlabel("Rango de especies (1 = más abundante)")
    plt.ylabel("Número de individuos")

    plt.xticks(x, especies, rotation=45, ha="right")

    plt.grid(alpha=0.3)
    plt.tight_layout()

    # ============================================================
    # GUARDAR
    # ============================================================

    plt.savefig(
        os.path.join(ruta_salida, f"Rango_Abundancia_{mun}.png"),
        dpi=300
    )

    plt.close()

# ============================================================
# EJECUCIÓN POR MUNICIPIO
# ============================================================

for mun in df_disp["MUNICIPIO"].unique():

    sub = df_disp[df_disp["MUNICIPIO"] == mun]

    ruta_mun = os.path.join(ruta_graficos, mun)
    os.makedirs(ruta_mun, exist_ok=True)

    grafico_rango_abundancia(mun, sub, ruta_mun)

# ============================================================
# GLOBAL
# ============================================================

grafico_rango_abundancia("GLOBAL", df_disp, ruta_graficos)

# ============================================================
# MENSAJE FINAL
# ============================================================

print("================================================")
print("GRÁFICOS RANGO–ABUNDANCIA GENERADOS")
print("================================================")
print("✔ Sin solapamiento de etiquetas")
print("✔ Top 3 especies destacadas")
print("✔ Organización por municipio + global")
print("================================================")


#------------------------------------------------------------------------------------------------------------------------------------------
# ======================================================================
# 🌎 BLOQUE INTEGRADO
# RECÁLCULO ECOFUNCIONAL TOTAL
# AVES + MAMÍFEROS
# ----------------------------------------------------------------------
# ESTE BLOQUE:
#
# ✔ NO MODIFICA LOS ANÁLISIS EXISTENTES
# ✔ CONSERVA LOS ANÁLISIS SEPARADOS
# ✔ CREA UN NUEVO ANÁLISIS INTEGRADO
# ✔ RECALCULA TODOS LOS ÍNDICES
# ✔ EXPORTA RESULTADOS COMPLETOS
#
# RESULTADO:
# FUNCIONALIDAD ECOLÓGICA TOTAL DEL PAISAJE
#
# ======================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================================
# 📁 CARPETA SALIDA
# ======================================================================

RUTA_GENERAL = os.path.join(
    RUTA_SALIDA,
    "GENERAL_FAUNA"
)

os.makedirs(RUTA_GENERAL, exist_ok=True)

# ======================================================================
# 📌 COPIA TOTAL DE DATOS
# ======================================================================

df_general = df.copy()

print("\n================================================")
print("🌎 ANALISIS ECOFUNCIONAL INTEGRADO")
print("AVES + MAMIFEROS")
print("================================================")

# ======================================================================
# 🌿 FVI
# ======================================================================

FVI_general = calcular_FVI(df_general)

FVI_general.to_excel(
    os.path.join(
        RUTA_GENERAL,
        "FVI_GENERAL.xlsx"
    )
)

print("✔ FVI calculado")

# ======================================================================
# 🌿 MATRIZ FUNCIONAL GOWER
# ======================================================================

matriz_general = matriz_gower(df_general)

print("✔ Matriz Gower calculada")

# ======================================================================
# 🌿 MATRIZ DE ABUNDANCIAS
# ======================================================================

abund_general = (

    df_general
    .groupby(["COBERTURA", "ESPECIE"])["INDIVIDUOS"]
    .sum()
    .unstack(fill_value=0)

)

print("✔ Matriz de abundancias calculada")

# ======================================================================
# 🌿 FD
# ======================================================================

FD_general = calcular_FD(
    matriz_general,
    abund_general
)

FD_general.to_excel(
    os.path.join(
        RUTA_GENERAL,
        "FD_GENERAL.xlsx"
    )
)

print("✔ FD calculado")

# ======================================================================
# 🌿 RAOQ
# ======================================================================

RaoQ_general = calcular_RaoQ(
    matriz_general,
    abund_general
)

RaoQ_general.to_excel(
    os.path.join(
        RUTA_GENERAL,
        "RaoQ_GENERAL.xlsx"
    )
)

print("✔ RaoQ calculado")

# ======================================================================
# 🌿 IMF
# ======================================================================

IMF_general = calcular_IMF(df_general)

IMF_general.to_excel(
    os.path.join(
        RUTA_GENERAL,
        "IMF_GENERAL.xlsx"
    )
)

print("✔ IMF calculado")

# ======================================================================
# 🌿 RESUMEN GENERAL
# ======================================================================

resumen_general = pd.DataFrame({

    "FVI": FVI_general,
    "FD": FD_general,
    "RaoQ": RaoQ_general,
    "IMF": IMF_general

}).fillna(0)

# ======================================================================
# 🌿 NORMALIZAR
# ======================================================================

resumen_norm = (

    resumen_general - resumen_general.min()

) / (

    resumen_general.max() - resumen_general.min()

)

resumen_norm = resumen_norm.fillna(0)

# ======================================================================
# 🌎 ÍNDICE ECOLÓGICO INTEGRADO
# ======================================================================

resumen_norm["INDICE_INTEGRADO"] = (

    resumen_norm["FVI"] * 0.30 +

    resumen_norm["FD"] * 0.20 +

    resumen_norm["RaoQ"] * 0.30 +

    resumen_norm["IMF"] * 0.20

)

# ======================================================================
# 🏆 RANKING
# ======================================================================

resumen_norm = resumen_norm.sort_values(
    by="INDICE_INTEGRADO",
    ascending=False
)

resumen_norm["RANKING"] = range(
    1,
    len(resumen_norm) + 1
)

# ======================================================================
# 💾 EXPORTAR
# ======================================================================

resumen_general.to_excel(
    os.path.join(
        RUTA_GENERAL,
        "INDICES_COMPLETOS_GENERAL.xlsx"
    )
)

resumen_norm.to_excel(
    os.path.join(
        RUTA_GENERAL,
        "INDICES_NORMALIZADOS_GENERAL.xlsx"
    )
)

print("✔ Índice integrado exportado")

# ======================================================================
# 🌿 CORRELACIONES
# ======================================================================

corr_general = resumen_general.corr(
    method="spearman"
)

corr_general.to_excel(
    os.path.join(
        RUTA_GENERAL,
        "CORRELACION_INDICES_GENERAL.xlsx"
    )
)

# ======================================================================
# 📊 HEATMAP
# ======================================================================

plt.figure(figsize=(7,5))

sns.heatmap(
    corr_general,
    annot=True,
    cmap="viridis",
    vmin=-1,
    vmax=1
)

plt.title(
    "Correlación índices funcionales - GENERAL"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RUTA_GENERAL,
        "Heatmap_Correlacion_GENERAL.png"
    ),
    dpi=300
)

plt.close()

print("✔ Heatmap exportado")

# ======================================================================
# 📈 GRAFICO INDICE INTEGRADO
# ======================================================================

plt.figure(figsize=(11,6))

sns.barplot(
    x=resumen_norm.index,
    y=resumen_norm["INDICE_INTEGRADO"]
)

plt.title(
    "Índice Ecofuncional Integrado - AVES + MAMÍFEROS"
)

plt.ylabel("Valor normalizado")

plt.xlabel("Cobertura")

plt.xticks(rotation=45)

plt.grid(
    axis='y',
    linestyle='--',
    alpha=0.4
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RUTA_GENERAL,
        "Indice_Integrado_GENERAL.png"
    ),
    dpi=300
)

plt.close()

print("✔ Gráfico exportado")

# ======================================================================
# 🌎 APORTE TAXONÓMICO POR COBERTURA
# ======================================================================

aporte = (

    df_general
    .groupby(["COBERTURA", "GRUPO"])["INDIVIDUOS"]
    .sum()
    .reset_index()

)

aporte["PROP_RELATIVA"] = (

    aporte.groupby("COBERTURA")["INDIVIDUOS"]
    .transform(lambda x: x / x.sum())

)

aporte.to_excel(
    os.path.join(
        RUTA_GENERAL,
        "APORTE_TAXONOMICO.xlsx"
    ),
    index=False
)

print("✔ Aporte taxonómico exportado")

# ======================================================================
# 📘 INTERPRETACIÓN AUTOMÁTICA
# ======================================================================

texto = []

texto.append(
    "================================================"
)

texto.append(
    "INTERPRETACION ECOFUNCIONAL INTEGRADA"
)

texto.append(
    "AVES + MAMIFEROS"
)

texto.append(
    "================================================\n"
)

# ----------------------------------------------------------------------

mejor = resumen_norm.index[0]

texto.append(
    f"La cobertura con mayor funcionalidad "
    f"ecológica integrada fue: {mejor}"
)

texto.append(
    f"\nValor del índice integrado: "
    f"{resumen_norm['INDICE_INTEGRADO'].max():.3f}"
)

texto.append("\n")

texto.append(
    "El análisis integrado representa "
    "la funcionalidad ecológica total "
    "del paisaje considerando conjuntamente "
    "aves y mamíferos."
)

texto.append("\n")

texto.append(
    "Valores altos reflejan:"
)

texto.append(
    "- mayor diversidad funcional"
)

texto.append(
    "- mayor amplitud funcional"
)

texto.append(
    "- mayor multifuncionalidad ecológica"
)

texto.append(
    "- mayor diferenciación funcional"
)

texto.append(
    "- mayor importancia ecosistémica"
)

texto.append("\n")

texto.append(
    "RANKING ECOFUNCIONAL:\n"
)

for idx, row in resumen_norm.iterrows():

    texto.append(

        f"{idx}: "
        f"{row['INDICE_INTEGRADO']:.3f}"

    )

# ======================================================================
# 💾 EXPORTAR INTERPRETACIÓN
# ======================================================================

with open(

    os.path.join(
        RUTA_GENERAL,
        "INTERPRETACION_GENERAL.txt"
    ),

    "w",

    encoding="utf-8"

) as f:

    f.write("\n".join(texto))

print("✔ Interpretación exportada")

# ======================================================================
# ✅ FINAL
# ======================================================================

print("\n================================================")
print("🌎 ANALISIS ECOFUNCIONAL GENERAL FINALIZADO")
print("================================================")
print(RUTA_GENERAL)
print("================================================")

#-------------------------------------------------------------------------------------------------------------

# ======================================================================
# 🌎 BLOQUE MUNICIPAL INTEGRADO
# ANALISIS ECOFUNCIONAL POR MUNICIPIO
# AVES + MAMIFEROS
# ======================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================================
# 📁 CARPETA PRINCIPAL
# ======================================================================

RUTA_MUNICIPIOS_GENERAL = os.path.join(
    RUTA_SALIDA,
    "MUNICIPIOS_GENERAL"
)

os.makedirs(
    RUTA_MUNICIPIOS_GENERAL,
    exist_ok=True
)

# ======================================================================
# 🌎 RECORRER MUNICIPIOS
# ======================================================================

for municipio in sorted(df["MUNICIPIO"].dropna().unique()):

    print("\n================================================")
    print(f"🌎 MUNICIPIO: {municipio}")
    print("ANALISIS INTEGRADO AVES + MAMIFEROS")
    print("================================================")

    # ==================================================================
    # 📌 FILTRAR MUNICIPIO
    # ==================================================================

    df_mun = df[
        df["MUNICIPIO"] == municipio
    ].copy()

    # ==================================================================
    # ⚠ VALIDAR DATOS
    # ==================================================================

    if len(df_mun) < 5:

        print("⚠ Datos insuficientes")
        continue

    # ==================================================================
    # 📁 CARPETA MUNICIPAL
    # ==================================================================

    nombre_mun = str(municipio).replace(" ", "_")

    ruta_mun = os.path.join(
        RUTA_MUNICIPIOS_GENERAL,
        nombre_mun
    )

    os.makedirs(
        ruta_mun,
        exist_ok=True
    )

    # ==================================================================
    # 🌿 FVI
    # ==================================================================

    FVI_mun = calcular_FVI(df_mun)

    FVI_mun.to_excel(
        os.path.join(
            ruta_mun,
            "FVI_GENERAL.xlsx"
        )
    )

    # ==================================================================
    # 🌿 MATRIZ GOWER
    # ==================================================================

    matriz_mun = matriz_gower(df_mun)

    # ==================================================================
    # 🌿 MATRIZ ABUNDANCIA
    # ==================================================================

    abund_mun = (

        df_mun
        .groupby(["COBERTURA", "ESPECIE"])["INDIVIDUOS"]
        .sum()
        .unstack(fill_value=0)

    )

    # ==================================================================
    # 🌿 FD
    # ==================================================================

    FD_mun = calcular_FD(
        matriz_mun,
        abund_mun
    )

    FD_mun.to_excel(
        os.path.join(
            ruta_mun,
            "FD_GENERAL.xlsx"
        )
    )

    # ==================================================================
    # 🌿 RAOQ
    # ==================================================================

    RaoQ_mun = calcular_RaoQ(
        matriz_mun,
        abund_mun
    )

    RaoQ_mun.to_excel(
        os.path.join(
            ruta_mun,
            "RaoQ_GENERAL.xlsx"
        )
    )

    # ==================================================================
    # 🌿 IMF
    # ==================================================================

    IMF_mun = calcular_IMF(df_mun)

    IMF_mun.to_excel(
        os.path.join(
            ruta_mun,
            "IMF_GENERAL.xlsx"
        )
    )

    # ==================================================================
    # 🌿 TABLA RESUMEN
    # ==================================================================

    resumen = pd.DataFrame({

        "FVI": FVI_mun,
        "FD": FD_mun,
        "RaoQ": RaoQ_mun,
        "IMF": IMF_mun

    }).fillna(0)

    # ==================================================================
    # 🌿 NORMALIZAR
    # ==================================================================

    resumen_norm = (

        resumen - resumen.min()

    ) / (

        resumen.max() - resumen.min()

    )

    resumen_norm = resumen_norm.fillna(0)

    # ==================================================================
    # 🌎 INDICE INTEGRADO
    # ==================================================================

    resumen_norm["INDICE_INTEGRADO"] = (

        resumen_norm["FVI"] * 0.30 +

        resumen_norm["FD"] * 0.20 +

        resumen_norm["RaoQ"] * 0.30 +

        resumen_norm["IMF"] * 0.20

    )

    # ==================================================================
    # 🏆 RANKING
    # ==================================================================

    resumen_norm = resumen_norm.sort_values(
        by="INDICE_INTEGRADO",
        ascending=False
    )

    resumen_norm["RANKING"] = range(
        1,
        len(resumen_norm) + 1
    )

    # ==================================================================
    # 💾 EXPORTAR
    # ==================================================================

    resumen.to_excel(
        os.path.join(
            ruta_mun,
            "INDICES_COMPLETOS_GENERAL.xlsx"
        )
    )

    resumen_norm.to_excel(
        os.path.join(
            ruta_mun,
            "INDICES_NORMALIZADOS_GENERAL.xlsx"
        )
    )

    # ==================================================================
    # 🌿 CORRELACIONES
    # ==================================================================

    corr = resumen.corr(
        method="spearman"
    )

    corr.to_excel(
        os.path.join(
            ruta_mun,
            "CORRELACION_INDICES_GENERAL.xlsx"
        )
    )

    # ==================================================================
    # 📊 HEATMAP
    # ==================================================================

    plt.figure(figsize=(7,5))

    sns.heatmap(
        corr,
        annot=True,
        cmap="viridis",
        vmin=-1,
        vmax=1
    )

    plt.title(
        f"Correlación índices\n{municipio}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            ruta_mun,
            "Heatmap_Correlacion_GENERAL.png"
        ),
        dpi=300
    )

    plt.close()

    # ==================================================================
    # 📈 GRAFICO PRINCIPAL
    # ==================================================================

    plt.figure(figsize=(11,6))

    sns.barplot(
        x=resumen_norm.index,
        y=resumen_norm["INDICE_INTEGRADO"]
    )

    plt.title(
        f"Índice Ecofuncional Integrado\n{municipio}"
    )

    plt.ylabel("Valor normalizado")

    plt.xlabel("Cobertura")

    plt.xticks(rotation=45)

    plt.grid(
        axis='y',
        linestyle='--',
        alpha=0.4
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            ruta_mun,
            "Indice_Integrado_GENERAL.png"
        ),
        dpi=300
    )

    plt.close()

    # ==================================================================
    # 🌎 APORTE TAXONOMICO
    # ==================================================================

    aporte = (

        df_mun
        .groupby(["COBERTURA", "GRUPO"])["INDIVIDUOS"]
        .sum()
        .reset_index()

    )

    aporte["PROP_RELATIVA"] = (

        aporte.groupby("COBERTURA")["INDIVIDUOS"]
        .transform(lambda x: x / x.sum())

    )

    aporte.to_excel(
        os.path.join(
            ruta_mun,
            "APORTE_TAXONOMICO.xlsx"
        ),
        index=False
    )

    # ==================================================================
    # 🌿 CSV PARA SIG
    # ==================================================================

    resumen_norm.reset_index().to_csv(
        os.path.join(
            ruta_mun,
            "SIG_GENERAL.csv"
        ),
        index=False
    )

    # ==================================================================
    # 📘 INTERPRETACION
    # ==================================================================

    mejor = resumen_norm.index[0]

    texto = []

    texto.append(
        f"ANALISIS ECOFUNCIONAL INTEGRADO\n"
    )

    texto.append(
        f"MUNICIPIO: {municipio}\n"
    )

    texto.append(
        f"Mejor cobertura: {mejor}"
    )

    texto.append(
        f"\nIndice integrado: "
        f"{resumen_norm['INDICE_INTEGRADO'].max():.3f}"
    )

    texto.append("\n")

    texto.append(
        "El análisis integrado representa "
        "la funcionalidad ecológica total "
        "del municipio considerando "
        "simultáneamente aves y mamíferos."
    )

    texto.append("\n")

    texto.append(
        "Coberturas con valores altos "
        "presentan mayor diversidad funcional, "
        "multifuncionalidad y relevancia ecológica."
    )

    with open(

        os.path.join(
            ruta_mun,
            "INTERPRETACION_GENERAL.txt"
        ),

        "w",

        encoding="utf-8"

    ) as f:

        f.write("\n".join(texto))

    print("✔ Analisis finalizado")

# ======================================================================
# ✅ FINAL
# ======================================================================

print("\n================================================")
print("🌎 ANALISIS MUNICIPAL INTEGRADO FINALIZADO")
print("================================================")
































# ======================================================================
# 🌎 ANALISIS DE CONSERVACION BIOGEOGRAFICA
# VERSION CORREGIDA PARA TUS CATEGORIAS
# ======================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================================
# 📁 CARPETA SALIDA
# ======================================================================

RUTA_CONSERVACION = os.path.join(
    RUTA_SALIDA,
    "CONSERVACION_BIOGEOGRAFICA"
)

os.makedirs(RUTA_CONSERVACION, exist_ok=True)

# ======================================================================
# 🧹 LIMPIEZA VARIABLES
# ======================================================================

df["IUCN"] = (

    df["IUCN"]
    .astype(str)
    .str.strip()

)

df["CITES"] = (

    df["CITES"]
    .astype(str)
    .str.strip()

)

df["Dist_Geo"] = (

    df["Dist_Geo"]
    .astype(str)
    .str.strip()

)

# ======================================================================
# 🌎 PESOS IUCN
# ======================================================================

peso_iucn = {

    "Preocupación Menor (LC)": 2,
    "Casi Amenazada (NT)": 4,
    "Vulnerable (VU)": 6,
    "En Peligro (EN)": 8,
    "En Peligro Crítico (CR)": 10,
    "Datos Insuficientes (DD)": 5,
    "No Evaluada (NE)": 3

}

# ======================================================================
# 🌎 PESOS CITES
# ======================================================================

peso_cites = {

    "Apendice I": 10,
    "Apendice II": 7,
    "Apendice III": 5,
    "No aplica": 1

}

# ======================================================================
# 🌎 PESOS DISTRIBUCION GEOGRAFICA
# ======================================================================

peso_geo = {

    "Casi endémica": 10,
    "Restringida": 8,
    "Migratoria": 6,
    "Cosmopolita": 2

}

# ======================================================================
# 🔥 MAPEAR VARIABLES
# ======================================================================

df["IUCN_valor"] = (

    df["IUCN"]
    .map(peso_iucn)
    .fillna(2)

)

df["CITES_valor"] = (

    df["CITES"]
    .map(peso_cites)
    .fillna(1)

)

df["GeoConserv_valor"] = (

    df["Dist_Geo"]
    .map(peso_geo)
    .fillna(2)

)

# ======================================================================
# 🔎 VERIFICAR RESULTADOS
# ======================================================================

print("\n==============================")
print("VALORES IUCN")
print("==============================")
print(df[["IUCN", "IUCN_valor"]].drop_duplicates())

print("\n==============================")
print("VALORES CITES")
print("==============================")
print(df[["CITES", "CITES_valor"]].drop_duplicates())

print("\n==============================")
print("VALORES DIST_GEO")
print("==============================")
print(df[["Dist_Geo", "GeoConserv_valor"]].drop_duplicates())

# ======================================================================
# 🌎 PESOS DEL INDICE
# ======================================================================

PESOS_CONSERVACION = {

    "IUCN_valor": 0.45,
    "CITES_valor": 0.20,
    "GeoConserv_valor": 0.35

}

# ======================================================================
# 🌎 VALOR CONSERVACION ESPECIE
# ======================================================================

df["Valor_Conservacion"] = (

    df["IUCN_valor"]

    * PESOS_CONSERVACION["IUCN_valor"]

    +

    df["CITES_valor"]

    * PESOS_CONSERVACION["CITES_valor"]

    +

    df["GeoConserv_valor"]

    * PESOS_CONSERVACION["GeoConserv_valor"]

)

# ======================================================================
# 🌎 ABUNDANCIA RELATIVA
# ======================================================================

total_cov = (

    df.groupby("COBERTURA")["INDIVIDUOS"]
    .transform("sum")

)

df["ABUND_REL_CONS"] = (

    df["INDIVIDUOS"]

    / total_cov

)

# ======================================================================
# 🌎 INDICE CONSERVACION BIOGEOGRAFICA
# ======================================================================

df["ICB_individual"] = (

    df["Valor_Conservacion"]

    * df["ABUND_REL_CONS"]

)

ICB_general = (

    df.groupby("COBERTURA")["ICB_individual"]
    .sum()
    .sort_values(ascending=False)

)

# ======================================================================
# 📊 EXPORTAR RESULTADOS
# ======================================================================

ICB_general.to_excel(

    os.path.join(
        RUTA_CONSERVACION,
        "ICB_GENERAL.xlsx"
    )

)

# ======================================================================
# 📊 GRAFICO
# ======================================================================

plt.figure(figsize=(10,5))

sns.barplot(

    x=ICB_general.index,
    y=ICB_general.values

)

plt.title(
    "Indice de Conservacion Biogeografica"
)

plt.ylabel(
    "ICB"
)

plt.xlabel(
    "Cobertura"
)

plt.xticks(rotation=45)

plt.grid(
    axis='y',
    linestyle='--',
    alpha=0.4
)

plt.tight_layout()

plt.savefig(

    os.path.join(
        RUTA_CONSERVACION,
        "ICB_GENERAL.png"
    ),

    dpi=300

)

plt.close()

# ======================================================================
# 🌎 TOP ESPECIES
# ======================================================================

top_especies = (

    df.groupby("ESPECIE")["Valor_Conservacion"]
    .mean()
    .sort_values(ascending=False)

)

top_especies.to_excel(

    os.path.join(
        RUTA_CONSERVACION,
        "TOP_ESPECIES_CONSERVACION.xlsx"
    )

)

# ======================================================================
# 🌎 RESULTADOS MUNICIPIO
# ======================================================================

ICB_municipio = (

    df.groupby(
        ["MUNICIPIO", "COBERTURA"]
    )["ICB_individual"]

    .sum()
    .reset_index()

)

ICB_municipio.to_excel(

    os.path.join(
        RUTA_CONSERVACION,
        "ICB_MUNICIPIO.xlsx"
    ),

    index=False

)

# ======================================================================
# ✅ FINAL
# ======================================================================

print("\n================================================")
print("🌎 ANALISIS DE CONSERVACION FINALIZADO")
print("================================================")

print(
    ICB_general
)

print("\n================================================")
print(
    RUTA_CONSERVACION
)

print("================================================")