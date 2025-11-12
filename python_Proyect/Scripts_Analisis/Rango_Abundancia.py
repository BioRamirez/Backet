
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

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# --- Verificar columnas clave ---
print(Registros.columns)

# --- Convertir 'INDIVIDUOS' a numérico ---
Registros["INDIVIDUOS"] = pd.to_numeric(Registros["INDIVIDUOS"], errors="coerce")

# --- Eliminar filas sin especie o sin valor de individuos ---
Registros = Registros.dropna(subset=["ESPECIE", "INDIVIDUOS"])

# --- 3. Asegurar que los nombres están bien ---
# (Por si hay diferencias en mayúsculas/minúsculas)
#Registros.columns = Registros.columns.str.strip().str.upper()

# --- 4. Calcular abundancia total por especie (todas las coberturas) ---
abund_total = (
    Registros.groupby("ESPECIE")["INDIVIDUOS"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)
abund_total["LOG10_ABUND"] = np.log10(abund_total["INDIVIDUOS"] + 1)
abund_total["RANGO"] = range(1, len(abund_total) + 1)
abund_total["COBERTURA"] = "TOTAL"

# --- 5. Calcular abundancia por cobertura ---
abund_cob = (
    Registros.groupby(["COBERTURA", "ESPECIE"])["INDIVIDUOS"]
    .sum()
    .reset_index()
)

# Crear una lista para guardar todos los DataFrames (total + coberturas)
curvas = [abund_total]

# --- 6. Procesar cada cobertura individual ---
for cobertura, df in abund_cob.groupby("COBERTURA"):
    df_sorted = df.sort_values(by="INDIVIDUOS", ascending=False).reset_index(drop=True)
    df_sorted["LOG10_ABUND"] = np.log10(df_sorted["INDIVIDUOS"] + 1)
    df_sorted["RANGO"] = range(1, len(df_sorted) + 1)
    df_sorted["COBERTURA"] = cobertura
    curvas.append(df_sorted)

# --- 7. Unir todas las curvas ---
curvas_df = pd.concat(curvas, ignore_index=True)

# --- 8. Graficar ---


import matplotlib.pyplot as plt

# --- 1. Calcular la abundancia total por cobertura ---
abundancia_por_cobertura = (
    curvas_df.groupby("COBERTURA")["INDIVIDUOS"].sum().sort_values(ascending=False)
)

# --- 2. Ordenar coberturas (TOTAL primero si existe) ---
orden_coberturas = ["TOTAL"] + [c for c in abundancia_por_cobertura.index if c != "TOTAL"]

# --- 3. Crear figura (una columna por cobertura) ---
fig, axes = plt.subplots(1, len(orden_coberturas), figsize=(4 * len(orden_coberturas), 5), sharey=True)
if len(orden_coberturas) == 1:
    axes = [axes]

# --- 4. Función para abreviar nombres de especies ---
def abreviar_nombre(nombre):
    partes = nombre.split()
    if len(partes) >= 2:
        return f"{partes[0][0]}. {partes[1]}"
    else:
        return nombre

# --- 5. Dibujar las curvas de rango-abundancia ---
for i, (ax, cobertura) in enumerate(zip(axes, orden_coberturas)):
    sub = curvas_df[curvas_df["COBERTURA"] == cobertura].sort_values("RANGO")
    
    if sub.empty:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", fontsize=12)
        continue

    # Curva de abundancia
    ax.plot(sub["RANGO"], sub["LOG10_ABUND"], marker="o", linestyle="-", color=f"C{i}")

    # Etiquetas (5 especies salteadas por curva)
    etiquetas_idx = sub.index[::max(1, len(sub)//5)]
    for j in etiquetas_idx:
        especie = abreviar_nombre(sub.loc[j, "ESPECIE"])
        ax.text(sub.loc[j, "RANGO"], sub.loc[j, "LOG10_ABUND"] + 0.1, especie,
                rotation=45, fontsize=8)

    ax.set_title(cobertura, fontsize=11, fontweight="bold")
    ax.set_xlabel("Rango de especies")
    if i == 0:
        ax.set_ylabel("Log₁₀ (Abundancia + 1)")
    ax.grid(True, linestyle="--", alpha=0.5)

# --- 6. Ajustes finales y visualización ---
plt.suptitle("Curvas rango-abundancia por cobertura", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


#-----------------------------fin del script-----------------------------



















































#-------------------Segundo grafico curvas rango-abundancia--------------------


import matplotlib.pyplot as plt
import numpy as np

# --- Asegúrate de tener 'curvas_df' con columnas: ESPECIE, COBERTURA, RANGO, LOG10_ABUND ---
# --- Si tus columnas están en mayúsculas, úsalo tal cual; si no, adáptalo. ---

def abreviar_nombre(nombre):
    partes = str(nombre).split()
    if len(partes) >= 2:
        return f"{partes[0][0]}. {partes[1]}"
    else:
        return nombre

def plot_curvas_grid(curvas_df, ncols=3, figsize_per_col=(4,5)):
    """
    Dibuja las curvas rango-abundancia en una grilla con ncols columnas.
    ncols: 1 (apiladas), 2 o 3 (u otro valor entero)
    figsize_per_col: tupla (ancho, alto) para cada columna
    """
    # ordenar coberturas (TOTAL primero si existe)
    abundancia_por_cobertura = curvas_df.groupby("COBERTURA")["INDIVIDUOS"].sum().sort_values(ascending=False)
    orden_coberturas = ["TOTAL"] + [c for c in abundancia_por_cobertura.index if c != "TOTAL"]

    n = len(orden_coberturas)
    # calcular layout
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))

    fig_w = figsize_per_col[0] * ncols
    fig_h = figsize_per_col[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharey='row')
    
    # normalizar axes a lista
    if isinstance(axes, plt.Axes):
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes).reshape(nrows, ncols)

    # iterar y graficar
    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            if idx >= n:
                # eje sobrante: ocultar
                axes[r, c].axis('off')
                idx += 1
                continue

            cov = orden_coberturas[idx]
            ax = axes[r, c]
            sub = curvas_df[curvas_df["COBERTURA"] == cov].sort_values("RANGO")
            if sub.empty:
                ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", fontsize=12)
                ax.set_title(cov, fontsize=11, fontweight="bold")
                ax.set_xlabel("Rango de especies")
                if r == 0 and c == 0:
                    ax.set_ylabel("Log₁₀ (Abundancia + 1)")
                idx += 1
                continue

            ax.plot(sub["RANGO"], sub["LOG10_ABUND"], marker="o", linestyle="-", color=f"C{idx % 10}")
            # etiquetas: 5 salteadas
            etiquetas_idx = sub.index[::max(1, len(sub)//5)]
            for j in etiquetas_idx:
                especie = abreviar_nombre(sub.loc[j, "ESPECIE"])
                ax.text(sub.loc[j, "RANGO"], sub.loc[j, "LOG10_ABUND"] + 0.01, especie,
                        rotation=30, fontsize=8, style='italic', ha='center')
            ax.set_title(cov, fontsize=11, fontweight="bold")
            ax.set_xlabel("Rango de especies")
            if c == 0:
                ax.set_ylabel("Log₁₀ (Abundancia + 1)")
            ax.grid(True, linestyle="--", alpha=0.1)
            idx += 1

    plt.suptitle("Curvas rango-abundancia por cobertura", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# ------------------------------
# Ejemplos de uso:
# Si quieres 1 columna (apiladas):
plot_curvas_grid(curvas_df, ncols=1)

# Si quieres en 2 columnas:
plot_curvas_grid(curvas_df, ncols=2)

# Si quieres en 3 columnas:
plot_curvas_grid(curvas_df, ncols=3)

































































































#-----------------------------fin del segundo grafico-----------------------------

import matplotlib.pyplot as plt
import numpy as np

# --- Ordenar coberturas por abundancia total ---
abundancia_por_cobertura = (
    curvas_df.groupby("COBERTURA")["INDIVIDUOS"].sum().sort_values(ascending=False)
)
orden_coberturas = ["TOTAL"] + [c for c in abundancia_por_cobertura.index if c != "TOTAL"]

# --- Función para abreviar nombres ---
def abreviar_nombre(nombre):
    partes = nombre.split()
    return f"{partes[0][0]}. {partes[1]}" if len(partes) >= 2 else nombre

# --- Crear figura ---
plt.figure(figsize=(12, 6))

# --- Graficar todas las coberturas ---
for i, cobertura in enumerate(orden_coberturas):
    sub = curvas_df[curvas_df["COBERTURA"] == cobertura].copy()
    sub = sub.sort_values("INDIVIDUOS", ascending=False).reset_index(drop=True)
    sub["RANGO"] = np.arange(1, len(sub) + 1)

    # Dibujar curva
    plt.plot(
        sub["RANGO"],
        sub["LOG10_ABUND"],
        marker="o",
        linestyle="-",
        color=f"C{i}",
        label=cobertura,
        linewidth=2,
        alpha=0.8
    )

# --- Etiquetas salteadas solo para TOTAL (para no saturar) ---
sub_total = curvas_df[curvas_df["COBERTURA"] == "TOTAL"].sort_values("INDIVIDUOS", ascending=False).reset_index(drop=True)
sub_total["RANGO"] = np.arange(1, len(sub_total) + 1)
n_etiquetas = 6
idx_etiquetas = np.linspace(0, len(sub_total) - 1, n_etiquetas, dtype=int)

for j in idx_etiquetas:
    especie = abreviar_nombre(sub_total.loc[j, "ESPECIE"])
    plt.text(
        sub_total.loc[j, "RANGO"],
        sub_total.loc[j, "LOG10_ABUND"] + 0.05,
        especie,
        rotation=45,
        ha="right",
        va="bottom",
        fontsize=8
    )

# --- Personalización del gráfico ---
plt.title("Curvas rango–abundancia comparativas por cobertura", fontsize=14, fontweight="bold")
plt.xlabel("Rango de especies (ordenadas por abundancia)")
plt.ylabel("Log₁₀ (Abundancia + 1)")
plt.legend(title="Cobertura")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


#--------------------------------------

















































#-------------------Otro intento de grafico--------------------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Copia del dataframe ---
df_long = curvas_df.rename(columns={
    "ESPECIE": "Especie",
    "COBERTURA": "Cobertura",
    "INDIVIDUOS": "Abundancia"
})

# --- Abreviar nombres científicos ---
def abreviar_nombre(nombre):
    partes = nombre.split()
    if len(partes) >= 2:
        return f"{partes[0][0]}. {partes[1]}"
    else:
        return nombre

df_long["Especie_abrev"] = df_long["Especie"].apply(abreviar_nombre)

# --- Orden de coberturas y desplazamientos personalizados ---
orden_coberturas = ["TOTAL", "Bdatf", "Bdbtf", "Bfvs", "Bgr"]
desplazamientos = {
    "TOTAL": 0,
    "Bdatf": 120,
    "Bdbtf": 250,
    "Bfvs": 360,
    "Bgr": 400
}

# --- Crear figura ---
plt.figure(figsize=(14, 7))

# --- Dibujar cada curva con su desplazamiento propio ---
for cobertura in orden_coberturas:
    subset = df_long[df_long["Cobertura"] == cobertura].copy()
    subset = subset.sort_values("Abundancia", ascending=False)
    subset["log_abund"] = np.log10(subset["Abundancia"] + 1)

    # rango desplazado según cobertura
    rango = np.arange(1, len(subset) + 1) + desplazamientos[cobertura]

    # graficar curva
    plt.plot(rango, subset["log_abund"], marker='o', label=cobertura)

    # mostrar nombres abreviados en ~5 puntos
    for j in np.linspace(0, len(subset) - 1, 5, dtype=int):
        plt.text(
            rango[j],
            subset["log_abund"].iloc[j] + 0.01,
            subset["Especie_abrev"].iloc[j],
            fontsize=8,
            rotation=15,
            ha='center',
            style='italic'
        )

# --- Etiquetas, leyenda y formato ---
plt.title("Curvas rango–abundancia comparativas por cobertura (alineadas)", fontsize=14, fontweight='bold')
plt.xlabel("Rango de especies (desplazadas por cobertura)", fontsize=12)
plt.ylabel("Log₁₀ (Abundancia + 1)", fontsize=12)
plt.legend(title="Cobertura", loc="upper right")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

#-----------------------------------------------
















































import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Copia del dataframe ---
df_long = curvas_df.rename(columns={
    "ESPECIE": "Especie",
    "COBERTURA": "Cobertura",
    "INDIVIDUOS": "Abundancia"
})

# --- Abreviar nombres científicos ---
def abreviar_nombre(nombre):
    partes = nombre.split()
    if len(partes) >= 2:
        return f"{partes[0][0]}. {partes[1]}"
    else:
        return nombre

df_long["Especie_abrev"] = df_long["Especie"].apply(abreviar_nombre)

# --- Orden de coberturas y desplazamientos personalizados ---
orden_coberturas = ["TOTAL", "Bdatf", "Bdbtf", "Bfvs", "Bgr"]

# desplazamientos horizontales (en eje X)
desplazamientos_x = {
    "TOTAL": 0,
    "Bdatf": 120,
    "Bdbtf": 240,
    "Bfvs": 360,
    "Bgr": 380
}

# desplazamientos verticales (en eje Y)
desplazamientos_y = {
    "TOTAL": 0.0,
    "Bdatf": 0.3,
    "Bdbtf": 0.6,
    "Bfvs": 0.9,
    "Bgr": 1.2
}

# --- Crear figura ---
plt.figure(figsize=(14, 8))

# --- Dibujar cada curva con desplazamientos horizontales y verticales ---
for cobertura in orden_coberturas:
    subset = df_long[df_long["Cobertura"] == cobertura].copy()
    subset = subset.sort_values("Abundancia", ascending=False)
    subset["log_abund"] = np.log10(subset["Abundancia"] + 1)
    
    # aplicar desplazamientos
    rango = np.arange(1, len(subset) + 1) + desplazamientos_x[cobertura]
    log_abund_desplazada = subset["log_abund"] + desplazamientos_y[cobertura]
    
    # graficar curva
    plt.plot(rango, log_abund_desplazada, marker='o', label=cobertura)
    
    # mostrar nombres abreviados en ~5 puntos por curva
    for j in np.linspace(0, len(subset) - 1, 5, dtype=int):
        plt.text(
            rango[j],
            log_abund_desplazada.iloc[j] + 0.03,
            subset["Especie_abrev"].iloc[j],
            fontsize=8,
            rotation=45,
            ha='center',
            style='italic'  # etiquetas en cursiva
        )

# --- Etiquetas, leyenda y formato ---
plt.title("Curvas rango–abundancia comparativas por cobertura (alineadas y desplazadas)", 
          fontsize=14, fontweight='bold')
plt.xlabel("Rango de especies (desplazadas por cobertura)", fontsize=12)
plt.ylabel("Log₁₀ (Abundancia + 1) + desplazamiento vertical", fontsize=12)
plt.legend(title="Cobertura", loc="upper right")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()




#----------------------

















































































