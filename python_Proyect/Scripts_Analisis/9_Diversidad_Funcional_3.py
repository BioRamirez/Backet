# =====================================================================
# 🌿 FRAMEWORK ECOFUNCIONAL MODULAR
# VERSION REFACTORIZADA Y ESCALABLE
# ---------------------------------------------------------------------
# Autor: Juan Carlos Ramírez Gil
#
# OBJETIVO:
# Framework modular para:
#
# ✔ FVI
# ✔ FD
# ✔ RaoQ
# ✔ IMF
# ✔ FDis
# ✔ Singularidad funcional
# ✔ Redundancia funcional
# ✔ Vulnerabilidad funcional
# ✔ Índice ecofuncional integrado
# ✔ Exportación SIG
# ✔ Graficos automáticos
# ✔ Análisis:
#       - AVES
#       - MAMIFEROS
#       - GENERAL
#       - MUNICIPIOS
#       - MUNICIPIOS + GRUPOS
#
# PRINCIPIO:
# "CALCULAR UNA VEZ, REUTILIZAR MUCHAS"
# =====================================================================

# =====================================================================
# 1. LIBRERIAS
# =====================================================================

import os
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import gower

from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# =====================================================================
# 2. CONFIGURACION
# =====================================================================

RUTA_ARCHIVO = r"D:\CORPONOR 2025\Backet\python_Proyect\data\POF_PAMPLONITA_2023_BD_AVES_MAMIFEROS.xlsx"

RUTA_SALIDA = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\ECOFUNCIONAL"

os.makedirs(RUTA_SALIDA, exist_ok=True)

# =====================================================================
# 3. PESOS ECOLOGICOS
# =====================================================================

peso_gremio = {

    'Insectívoro': 7,
    'Frugívoro': 8,
    'Nectarívoro': 8,
    'Carroñero': 9,
    'Carnívoro': 8,
    'Omnívoro': 4,
    'Granívoro': 3,
    'Herbívoro': 6,
    'Herbivoro': 6
}

peso_migra = {

    'Res': 8,
    'Residentes': 8,
    'Lat': 6,
    'Latitudinal': 6,
    'Alt-Loc': 5,
    'Loc': 5,
    'Lat-Trans': 7,
    'Lat-Alt-Trans-Loc': 9,
    'Nomadismo': 7,
    'Estacional': 6
}

peso_uso = {

    'Sin uso conocido': 8,
    'Cultural': 5,
    'Uso Cultural': 5,
    'Subsistencia': 2,
    'Mascotas': 1,
    'Mascota': 1,
    'Medicinal': 4,
    'Otro': 4
}

peso_geo = {

    'Endémica': 10,
    'Casi endémica': 9,
    'Restringida': 8,
    'Nearctica, Neotropical': 6,
    'Neotropical': 5,
    'Cosmopolita': 2,
    'Introducida': 1
}

# =====================================================================
# 4. PESOS INDICE INTEGRADO
# =====================================================================

PESOS_INDICES = {

    "FVI": 0.20,
    "FD": 0.15,
    "RaoQ": 0.15,
    "IMF": 0.15,
    "FDis": 0.10,
    "Singularidad": 0.10,
    "Redundancia": 0.10,
    "Vulnerabilidad": 0.05
}

# =====================================================================
# 5. CARGAR DATOS
# =====================================================================

df = pd.read_excel(RUTA_ARCHIVO)

# =====================================================================
# 6. LIMPIEZA GENERAL
# =====================================================================

def limpiar_datos(df):

    df = df.copy()

    for col in df.columns:

        if df[col].dtype == "object":

            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
            )

    return df

df = limpiar_datos(df)

# =====================================================================
# 7. NORMALIZAR MUNICIPIOS
# =====================================================================

if "MUNICIPIO" in df.columns:

    df["MUNICIPIO"] = (
        df["MUNICIPIO"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

# =====================================================================
# 8. DEFINIR GRUPOS
# =====================================================================

if "CLASE" in df.columns:

    df["GRUPO"] = np.where(

        df["CLASE"]
        .str.contains("Aves", case=False, na=False),

        "AVES",

        "MAMIFEROS"
    )

else:

    raise ValueError(
        "❌ No existe columna CLASE"
    )

# =====================================================================
# 9. PESOS FUNCIONALES
# =====================================================================

df["Gremio_valor"] = (
    df["Gremio"]
    .map(peso_gremio)
    .fillna(3)
)

df["Migracion_valor"] = (
    df["Tipo_Migra"]
    .map(peso_migra)
    .fillna(3)
)

df["Uso_valor"] = (
    df["Uso"]
    .map(peso_uso)
    .fillna(3)
)

df["Geo_valor"] = (
    df["Dist_Geo"]
    .map(peso_geo)
    .fillna(3)
)

# =====================================================================
# 10. VALOR FUNCIONAL ESPECIE
# =====================================================================

df["Valor_funcional_especie"] = (

    df["Gremio_valor"] * 0.35 +

    df["Migracion_valor"] * 0.20 +

    df["Geo_valor"] * 0.35 +

    df["Uso_valor"] * 0.10
)

# =====================================================================
# 11. ABUNDANCIA RELATIVA
# =====================================================================

total_cov = (

    df.groupby(
        ["GRUPO", "COBERTURA"]
    )["INDIVIDUOS"]

    .transform("sum")
)

df["ABUND_REL"] = (
    df["INDIVIDUOS"] / total_cov
)

# =====================================================================
# 12. FUNCIONES BASE
# =====================================================================

def crear_matriz_abundancia(datos):

    return (

        datos.groupby(
            ["COBERTURA", "ESPECIE"]
        )["INDIVIDUOS"]

        .sum()

        .unstack(fill_value=0)
    )

# ---------------------------------------------------------------------

def matriz_gower(datos):

    rasgos = [
        "Gremio",
        "Tipo_Migra",
        "Uso",
        "Dist_Geo"
    ]

    rasgos_sp = (

        datos.groupby("ESPECIE")[rasgos]
        .first()
    )

    matriz = gower.gower_matrix(rasgos_sp)

    return pd.DataFrame(

        matriz,
        index=rasgos_sp.index,
        columns=rasgos_sp.index
    )

# ---------------------------------------------------------------------

def normalizar(df):

    return (

        (df - df.min()) /

        (df.max() - df.min())

    ).fillna(0)

# ---------------------------------------------------------------------

def exportar_excel(df, ruta):

    df.to_excel(ruta)

# ---------------------------------------------------------------------

def exportar_csv(df, ruta):

    df.to_csv(
        ruta,
        index=False,
        encoding="utf-8-sig"
    )

# =====================================================================
# 13. METRICAS ECOFUNCIONALES
# =====================================================================

def calcular_FVI(datos):

    datos = datos.copy()

    datos["FVI_ind"] = (

        datos["Valor_funcional_especie"] *

        datos["ABUND_REL"]
    )

    return (

        datos.groupby("COBERTURA")["FVI_ind"]

        .sum()

        .sort_values(ascending=False)
    )

# ---------------------------------------------------------------------

def calcular_FD(dist_matrix, abund):

    resultados = {}

    for cov in abund.index:

        spp = abund.loc[cov]

        spp = spp[spp > 0]

        if len(spp) < 2:

            resultados[cov] = 0
            continue

        sub = dist_matrix.loc[
            spp.index,
            spp.index
        ]

        vals = sub.values[
            np.triu_indices_from(sub, k=1)
        ]

        resultados[cov] = np.mean(vals)

    return pd.Series(resultados)

# ---------------------------------------------------------------------

def calcular_RaoQ(dist_matrix, abund):

    resultados = {}

    for cov in abund.index:

        spp = abund.loc[cov]

        spp = spp[spp > 0]

        if len(spp) < 2:

            resultados[cov] = 0
            continue

        p = spp / spp.sum()

        Q = 0

        for i in spp.index:

            for j in spp.index:

                Q += (
                    dist_matrix.loc[i, j] *
                    p[i] *
                    p[j]
                )

        resultados[cov] = Q

    return pd.Series(resultados)

# ---------------------------------------------------------------------

def shannon(p):

    p = p[p > 0]

    return -np.sum(p * np.log(p))

# ---------------------------------------------------------------------

def pielou(H, S):

    if S <= 1:
        return 0

    return H / np.log(S)

# ---------------------------------------------------------------------

def calcular_IMF(datos):

    resultados = {}

    for cov in datos["COBERTURA"].unique():

        sub = datos[
            datos["COBERTURA"] == cov
        ]

        prop = (

            sub.groupby("Gremio")["ABUND_REL"]

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

        resultados[cov] = IMF

    return pd.Series(resultados)

# ---------------------------------------------------------------------

def calcular_FDis(datos, abund):

    rasgos = [
        "Gremio_valor",
        "Migracion_valor",
        "Uso_valor",
        "Geo_valor"
    ]

    rasgos_sp = (

        datos.groupby("ESPECIE")[rasgos]
        .mean()
    )

    scaler = StandardScaler()

    rasgos_std = pd.DataFrame(

        scaler.fit_transform(rasgos_sp),

        index=rasgos_sp.index,

        columns=rasgos_sp.columns
    )

    centroide = (
        rasgos_std.mean(axis=0)
        .values
        .reshape(1, -1)
    )

    resultados = {}

    for cov in abund.index:

        spp = abund.loc[cov]

        spp = spp[spp > 0]

        if len(spp) < 2:

            resultados[cov] = 0
            continue

        rasgos_cov = rasgos_std.loc[spp.index]

        distancias = cdist(

            rasgos_cov,
            centroide,
            metric="euclidean"

        ).flatten()

        pesos = spp.values / spp.sum()

        resultados[cov] = np.sum(
            distancias * pesos
        )

    return pd.Series(resultados)

# ---------------------------------------------------------------------

def calcular_singularidad(dist_matrix, abund):

    singularidad_sp = (
        dist_matrix.mean(axis=1)
    )

    resultados = {}

    for cov in abund.index:

        spp = abund.loc[cov]

        spp = spp[spp > 0]

        if len(spp) == 0:

            resultados[cov] = 0
            continue

        pesos = spp / spp.sum()

        resultados[cov] = np.sum(

            singularidad_sp.loc[spp.index] *

            pesos
        )

    return pd.Series(resultados)

# =====================================================================
# 14. GRAFICOS
# =====================================================================

def guardar_barplot(
    serie,
    titulo,
    ruta
):

    plt.figure(figsize=(10, 5))

    sns.barplot(
        x=serie.index,
        y=serie.values
    )

    plt.title(titulo)

    plt.xticks(rotation=45)

    plt.grid(
        axis='y',
        linestyle='--',
        alpha=0.4
    )

    plt.tight_layout()

    plt.savefig(
        ruta,
        dpi=300
    )

    plt.close()

# ---------------------------------------------------------------------

def guardar_heatmap(
    corr,
    titulo,
    ruta
):

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        corr,
        annot=True,
        cmap="viridis",
        vmin=-1,
        vmax=1
    )

    plt.title(titulo)

    plt.tight_layout()

    plt.savefig(
        ruta,
        dpi=300
    )

    plt.close()

# =====================================================================
# 15. CLASIFICACION
# =====================================================================

def clasificar(valor):

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

# =====================================================================
# 16. PIPELINE MAESTRO
# =====================================================================

def ejecutar_pipeline_ecofuncional(
    datos,
    ruta_salida,
    nombre
):

    print("\n================================================")
    print(f"🌿 ANALISIS: {nombre}")
    print("================================================")

    os.makedirs(
        ruta_salida,
        exist_ok=True
    )

    # ================================================================
    # MATRICES
    # ================================================================

    abund = crear_matriz_abundancia(datos)

    matriz = matriz_gower(datos)

    # ================================================================
    # INDICES
    # ================================================================

    FVI = calcular_FVI(datos)

    FD = calcular_FD(
        matriz,
        abund
    )

    RaoQ = calcular_RaoQ(
        matriz,
        abund
    )

    IMF = calcular_IMF(datos)

    FDis = calcular_FDis(
        datos,
        abund
    )

    Singularidad = calcular_singularidad(
        matriz,
        abund
    )

    # ================================================================
    # REDUNDANCIA
    # ================================================================

    riqueza = (
        abund > 0
    ).sum(axis=1)

    Redundancia = (
        riqueza / (FDis + 0.001)
    )

    Vulnerabilidad = (
        1 / (Redundancia + 0.001)
    )

    # ================================================================
    # RESUMEN
    # ================================================================

    resumen = pd.DataFrame({

        "FVI": FVI,
        "FD": FD,
        "RaoQ": RaoQ,
        "IMF": IMF,
        "FDis": FDis,
        "Singularidad": Singularidad,
        "Redundancia": Redundancia,
        "Vulnerabilidad": Vulnerabilidad
    }).fillna(0)

    # ================================================================
    # NORMALIZAR
    # ================================================================

    resumen_norm = normalizar(resumen)

    # ================================================================
    # INDICE INTEGRADO
    # ================================================================

    resumen_norm["INDICE_ECOFUNCIONAL"] = 0

    for col, peso in PESOS_INDICES.items():

        if col in resumen_norm.columns:

            resumen_norm["INDICE_ECOFUNCIONAL"] += (

                resumen_norm[col] * peso
            )

    # ================================================================
    # RANKING
    # ================================================================

    resumen_norm = resumen_norm.sort_values(

        by="INDICE_ECOFUNCIONAL",

        ascending=False
    )

    resumen_norm["RANKING"] = range(

        1,
        len(resumen_norm) + 1
    )

    # ================================================================
    # CATEGORIA
    # ================================================================

    resumen_norm["CATEGORIA"] = (

        resumen_norm["INDICE_ECOFUNCIONAL"]

        .apply(clasificar)
    )

    # ================================================================
    # EXPORTACIONES
    # ================================================================

    exportar_excel(

        resumen,

        os.path.join(
            ruta_salida,
            f"INDICES_COMPLETOS_{nombre}.xlsx"
        )
    )

    exportar_excel(

        resumen_norm,

        os.path.join(
            ruta_salida,
            f"INDICES_NORMALIZADOS_{nombre}.xlsx"
        )
    )

    # ================================================================
    # TABLA SIG
    # ================================================================

    tabla_sig = (
        resumen_norm
        .reset_index()
    )

    tabla_sig.rename(

        columns={"index": "COBERTURA"},

        inplace=True
    )

    exportar_csv(

        tabla_sig,

        os.path.join(
            ruta_salida,
            f"TABLA_SIG_{nombre}.csv"
        )
    )

    # ================================================================
    # CORRELACIONES
    # ================================================================

    corr = resumen.corr(
        method="spearman"
    )

    exportar_excel(

        corr,

        os.path.join(
            ruta_salida,
            f"CORRELACION_{nombre}.xlsx"
        )
    )

    # ================================================================
    # GRAFICOS
    # ================================================================

    guardar_barplot(

        resumen_norm["INDICE_ECOFUNCIONAL"],

        f"Indice Ecofuncional - {nombre}",

        os.path.join(
            ruta_salida,
            f"INDICE_ECOFUNCIONAL_{nombre}.png"
        )
    )

    guardar_heatmap(

        corr,

        f"Correlacion indices - {nombre}",

        os.path.join(
            ruta_salida,
            f"HEATMAP_{nombre}.png"
        )
    )

    # ================================================================
    # PCA
    # ================================================================

    scaler = StandardScaler()

    X = scaler.fit_transform(
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
            f"PCA Ecofuncional - {nombre}"
        )

        plt.tight_layout()

        plt.savefig(

            os.path.join(
                ruta_salida,
                f"PCA_{nombre}.png"
            ),

            dpi=300
        )

        plt.close()

    # ================================================================
    # INTERPRETACION
    # ================================================================

    mejor = resumen_norm.index[0]

    texto = []

    texto.append(
        f"ANALISIS ECOFUNCIONAL - {nombre}\n"
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

            f"{row['INDICE_ECOFUNCIONAL']:.3f}"
        )

    with open(

        os.path.join(
            ruta_salida,
            f"INTERPRETACION_{nombre}.txt"
        ),

        "w",

        encoding="utf-8"

    ) as f:

        f.write("\n".join(texto))

    print("✔ FINALIZADO")

# =====================================================================
# 17. ANALISIS AVES
# =====================================================================

df_aves = df[
    df["GRUPO"] == "AVES"
].copy()

ejecutar_pipeline_ecofuncional(

    datos=df_aves,

    ruta_salida=os.path.join(
        RUTA_SALIDA,
        "AVES"
    ),

    nombre="AVES"
)

# =====================================================================
# 18. ANALISIS MAMIFEROS
# =====================================================================

df_mam = df[
    df["GRUPO"] == "MAMIFEROS"
].copy()

ejecutar_pipeline_ecofuncional(

    datos=df_mam,

    ruta_salida=os.path.join(
        RUTA_SALIDA,
        "MAMIFEROS"
    ),

    nombre="MAMIFEROS"
)

# =====================================================================
# 19. ANALISIS GENERAL
# =====================================================================

ejecutar_pipeline_ecofuncional(

    datos=df,

    ruta_salida=os.path.join(
        RUTA_SALIDA,
        "GENERAL"
    ),

    nombre="GENERAL"
)

# =====================================================================
# 20. ANALISIS MUNICIPIOS
# =====================================================================

if "MUNICIPIO" in df.columns:

    for municipio in sorted(

        df["MUNICIPIO"]
        .dropna()
        .unique()
    ):

        print("\n================================================")
        print(f"🌎 MUNICIPIO: {municipio}")
        print("================================================")

        df_mun = df[
            df["MUNICIPIO"] == municipio
        ].copy()

        # ============================================================
        # MUNICIPIO GENERAL
        # ============================================================

        ejecutar_pipeline_ecofuncional(

            datos=df_mun,

            ruta_salida=os.path.join(

                RUTA_SALIDA,

                "MUNICIPIOS",

                municipio,

                "GENERAL"
            ),

            nombre=f"{municipio}_GENERAL"
        )

        # ============================================================
        # MUNICIPIO + GRUPOS
        # ============================================================

        for grupo in [

            "AVES",
            "MAMIFEROS"
        ]:

            df_sub = df_mun[
                df_mun["GRUPO"] == grupo
            ].copy()

            if len(df_sub) == 0:
                continue

            ejecutar_pipeline_ecofuncional(

                datos=df_sub,

                ruta_salida=os.path.join(

                    RUTA_SALIDA,

                    "MUNICIPIOS",

                    municipio,

                    grupo
                ),

                nombre=f"{municipio}_{grupo}"
            )

# =====================================================================
# 21. FINAL
# =====================================================================

print("\n================================================")
print("🌿 FRAMEWORK ECOFUNCIONAL FINALIZADO")
print("================================================")
print(RUTA_SALIDA)
print("================================================")