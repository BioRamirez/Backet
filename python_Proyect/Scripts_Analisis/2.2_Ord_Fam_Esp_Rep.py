

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl import load_workbook
from openpyxl.drawing.image import Image
import os

# Ruta del archivo
ruta = r"D:\Forestal Consultores\2026\FAUNA\BD\AVES\Aves_Secundario_San_Roque.xlsx"
# Leer el archivo Excel
Registros = pd.read_excel(ruta)
# =========================================================
# FILTRAR UNA SOLA CLASE
# =========================================================

grupo = "AVES"   # AVES, MAMIFEROS, REPTILES, ANFIBIOS etc.

Registros = Registros[
    Registros["CLASE"].astype(str).str.upper() == grupo.upper()
].copy()

# Reiniciar índice
Registros.reset_index(drop=True, inplace=True)

# =========================================================
# VERIFICACIÓN
# =========================================================

print(f"\nGrupo seleccionado: {grupo}")
print(f"Número de registros: {len(Registros)}")

print("\nPrimeras filas:")
print(Registros.head())

# Mostrar las primeras filas
print(" Primeras filas del archivo:")
print(Registros.head())

# --- Copiar dataframe base ---
df = Registros.copy()

# Mostrar las primeras filas
print(" Primeras filas del archivo:")
print(Registros.head())

# Mostrar nombres de las columnas
print("\n Columnas del DataFrame:")
print(Registros.columns)

import pandas as pd

# --- Cargar el df (ya lo tienes cargado como Registros) ---
df = Registros.copy()






































import pandas as pd

def generar_texto_por_orden(df):
    textos = {}

    total_individuos = df["INDIVIDUOS"].sum()

    for orden, subdf in df.groupby("Orden"):

        # Conteos
        n_familias = subdf["Familia"].nunique()
        n_especies = subdf["ESPECIE"].nunique()
        n_individuos = subdf["INDIVIDUOS"].sum()

        pct = round((n_individuos / total_individuos) * 100, 2)

        # Familia más abundante
        familia_abund = (
            subdf.groupby("Familia")["INDIVIDUOS"]
            .sum()
            .sort_values(ascending=False)
            .index[0]
        )

        # Top especies
        top_especies = (
            subdf.groupby("ESPECIE")["INDIVIDUOS"]
            .sum()
            .sort_values(ascending=False)
        )

        # Manejar distintos escenarios
        if len(top_especies) == 0:
            especie1 = "No disponible"
            ind1 = 0
            especie2 = "No disponible"
            ind2 = 0

        elif len(top_especies) == 1:
            especie1 = top_especies.index[0]
            ind1 = int(top_especies.iloc[0])
            especie2 = "No hay segunda especie"
            ind2 = 0

        else:
            especie1 = top_especies.index[0]
            ind1 = int(top_especies.iloc[0])
            especie2 = top_especies.index[1]
            ind2 = int(top_especies.iloc[1])

        # TEXTO AUTOMÁTICO
        texto = f"""
El orden **{orden}** presentó un total de **{n_individuos}** individuos, lo que representa el **{pct}%** del total registrado.
Se identificaron **{n_familias} familias** y **{n_especies} especies** dentro del orden.
La familia más abundante fue **{familia_abund}**.
Las especies con mayor representatividad fueron **{especie1}** ({ind1} individuos) y **{especie2}** ({ind2} individuos).
Estos resultados reflejan la importancia del orden {orden} dentro de la estructura ecológica del ensamblaje observado.
        """

        textos[orden] = texto.strip()

    return textos





textos_generados = generar_texto_por_orden(df)

for orden, texto in textos_generados.items():
    print("\n" + "="*80)
    print(texto)




#------------------------------



import pandas as pd
import os
from collections import defaultdict

# ---------------------------
# Entrada: detecta df o Registros
# ---------------------------
if 'df' in globals():
    df_input = df.copy()
elif 'Registros' in globals():
    df_input = Registros.copy()
else:
    raise ValueError("No se encontró DataFrame. Define 'df' o 'Registros' en el entorno.")

# ---------------------------
# Buscar columnas (tolerante a mayúsculas/minúsculas)
# ---------------------------
def find_col(cols, target):
    tl = target.lower()
    for c in cols:
        if c.lower() == tl:
            return c
    return None

cols = df_input.columns.tolist()
col_especie = find_col(cols, 'ESPECIE')
col_orden   = find_col(cols, 'Orden')
col_familia = find_col(cols, 'Familia')
col_ind     = find_col(cols, 'INDIVIDUOS')

if not (col_especie and col_orden and col_familia and col_ind):
    missing = [name for name,found in [('ESPECIE',col_especie),('Orden',col_orden),('Familia',col_familia),('INDIVIDUOS',col_ind)] if not found]
    raise ValueError(f"Faltan columnas requeridas en el DataFrame: {missing}. Asegúrate de tener ESPECIE, Orden, Familia, INDIVIDUOS.")

# Renombrar internamente
df = df_input.rename(columns={col_especie: 'ESPECIE', col_orden: 'Orden', col_familia: 'Familia', col_ind: 'INDIVIDUOS'}).copy()

# ---------------------------
# Limpiar y tipificar
# ---------------------------
df['ESPECIE'] = df['ESPECIE'].astype(str).str.strip()
df['Orden']   = df['Orden'].astype(str).str.strip()
df['Familia'] = df['Familia'].astype(str).str.strip()
df['INDIVIDUOS'] = pd.to_numeric(df['INDIVIDUOS'], errors='coerce').fillna(0).astype(int)

# Eliminar filas sin especie válida
df = df[~df['ESPECIE'].isin(['', 'nan', 'None'])]

# Totales globales
total_especies = df['ESPECIE'].nunique()
total_individuos = int(df['INDIVIDUOS'].sum())
total_ordenes = df['Orden'].nunique()

# ---------------------------
# Especie más abundante global
# ---------------------------
global_top_series = df.groupby('ESPECIE')['INDIVIDUOS'].sum().sort_values(ascending=False)
if len(global_top_series) > 0:
    especie_mas_abund_global = global_top_series.index[0]
    especie_mas_abund_global_n = int(global_top_series.iloc[0])
else:
    especie_mas_abund_global = None
    especie_mas_abund_global_n = 0

# ---------------------------
# Resumen por orden: n_especies, n_individuos, n_familias
# ---------------------------
ordenes_resumen = (
    df.groupby('Orden')
      .agg(
          n_especies = ('ESPECIE', 'nunique'),
          n_individuos = ('INDIVIDUOS', 'sum'),
          n_familias = ('Familia', 'nunique')
      )
)
# Ordenar por n_especies desc (richness)
ordenes_resumen = ordenes_resumen.sort_values(['n_especies','n_individuos'], ascending=[False, False])

# ---------------------------
# Preparar datos por orden/familia/especie
# ---------------------------
# Familias por orden con (n_especies_fam, n_individuos_fam)
familias_por_orden = {}
especies_por_orden = {}
for ord_name, sub in df.groupby('Orden'):
    fam_tab = (
        sub.groupby('Familia')
           .agg(n_especies_fam=('ESPECIE','nunique'),
                n_individuos_fam=('INDIVIDUOS','sum'))
           .sort_values('n_individuos_fam', ascending=False)
    )
    familias_por_orden[ord_name] = fam_tab
    sp_tab = sub.groupby('ESPECIE')['INDIVIDUOS'].sum().sort_values(ascending=False)
    especies_por_orden[ord_name] = sp_tab

# ---------------------------
# Construir PÁRRAFO GLOBAL según reglas del ejemplo
# ---------------------------
if ordenes_resumen.shape[0] == 0:
    parrafo_global = "No se registraron órdenes en la base de datos."
else:
    # Top orden
    first_ord = ordenes_resumen.index[0]
    first_nsp = int(ordenes_resumen.loc[first_ord, 'n_especies'])
    first_pct_sp = round((first_nsp / total_especies) * 100, 2) if total_especies > 0 else 0.0

    # Preparar lista de strings
    global_parts = []

    # Familias del orden más representativo con número de especies
    fam_tab_first = familias_por_orden.get(first_ord, pd.DataFrame())

    if not fam_tab_first.empty:
        fams_first_str = ", ".join(
            [
                f"{fam} ({int(row['n_especies_fam'])} spp.)"
                for fam, row in fam_tab_first.iterrows()
            ]
        )
    else:
        fams_first_str = ""

    part1 = (
        f"En general, el orden más representativo fue el orden **{first_ord}**, "
        f"con {first_nsp} especies, pertenecientes a familias como: {fams_first_str}. "
        f"Este orden representó el {first_pct_sp}% del total de especies ({total_especies})."
    )

    # Agregar primer párrafo al global
    global_parts.append(part1)

    # Segundo y tercer orden (si existen)
    siguientes = ordenes_resumen.index[1:4]
    if len(siguientes) > 0:
        seg_parts = []
        for ord_name in siguientes:
            nsp = int(ordenes_resumen.loc[ord_name, 'n_especies'])
            pct_sp = round((nsp / total_especies) * 100, 2)
            fam_tab = familias_por_orden.get(ord_name, pd.DataFrame())

            if not fam_tab.empty:
                fams_str = ", ".join(fam_tab.index)
            else:
                fams_str = ""

            if fams_str:
                seg_parts.append(
                    f"**{ord_name}**, con {nsp} especies, pertenecientes a las familias {fams_str} ({pct_sp}%)"
                )
            else:
                seg_parts.append(
                    f"**{ord_name}** ({nsp} especies; {pct_sp}%)"
                )

        global_parts.append(
            "El segundo y siguientes órdenes más representativos fueron: "
            + "; ".join(seg_parts) + "."
        )

    # Para los órdenes restantes
    restantes = ordenes_resumen.index[4:]
    if len(restantes) > 0:
        group_by_count = defaultdict(list)
        for ord_name in restantes:
            cnt = int(ordenes_resumen.loc[ord_name, 'n_especies'])
            group_by_count[cnt].append(ord_name)

        frases = []
        for cnt in sorted(group_by_count.keys(), reverse=True):
            lista = group_by_count[cnt]
            if len(lista) == 1:
                frases.append(f"el orden {lista[0]}, con {cnt} especies")
            else:
                frases.append(f"los órdenes {', '.join(lista)}, cada uno con {cnt} especies")

        if frases:
            global_parts.append("Por consiguiente, " + "; ".join(frases) + ".")

    # Añadir especie más abundante global
    if especie_mas_abund_global:
        global_parts.append(
            f"La especie más abundante registrada en el muestreo fue **{especie_mas_abund_global}**, "
            f"con {especie_mas_abund_global_n} individuos."
        )

    parrafo_global = " ".join(global_parts)

# Construir párrafos por orden (detallados) en el orden solicitado (más a menos especies)
# ---------------------------
orden_paragraphs = []
for ord_name, stats in ordenes_resumen.iterrows():
    n_especies = int(stats['n_especies'])
    n_individuos = int(stats['n_individuos'])
    n_familias = int(stats['n_familias'])
    pct_sp = round((n_especies / total_especies) * 100, 2) if total_especies>0 else 0.0
    pct_ind = round((n_individuos / total_individuos) * 100, 2) if total_individuos>0 else 0.0

    # Familias (todas) - presentar como lista "Familia: X especies y Y individuos"
    fam_tab = familias_por_orden.get(ord_name, pd.DataFrame())
    familias_lines = []
    # ordenar por n_especies_fam o n_individuos_fam (escogemos por individuos)
    if not fam_tab.empty:
        fam_tab_sorted = fam_tab.sort_values('n_individuos_fam', ascending=False)
        for fam, frow in fam_tab_sorted.iterrows():
            familias_lines.append(f"{fam}: {int(frow['n_especies_fam'])} especies y {int(frow['n_individuos_fam'])} individuos")
        familias_text = "; ".join(familias_lines)
    else:
        familias_text = "No hay familias registradas."

    # Familia más representativa (por especies y por individuos)
    if not fam_tab.empty:
        fam_por_especies = fam_tab.sort_values('n_especies_fam', ascending=False).index[0]
        nsp_fam = int(fam_tab.sort_values('n_especies_fam', ascending=False).iloc[0]['n_especies_fam'])
        fam_por_ind = fam_tab.sort_values('n_individuos_fam', ascending=False).index[0]
        nind_fam = int(fam_tab.sort_values('n_individuos_fam', ascending=False).iloc[0]['n_individuos_fam'])
    else:
        fam_por_especies = "No disponible"
        nsp_fam = 0
        fam_por_ind = "No disponible"
        nind_fam = 0

    # Especies más abundantes del orden (top 3 si hay)
    sp_series = especies_por_orden.get(ord_name, pd.Series(dtype=int))
    top_n = min(3, len(sp_series))
    top_lines = []
    for i in range(top_n):
        sp = sp_series.index[i]
        nsp = int(sp_series.iloc[i])
        nota = ""
        if especie_mas_abund_global and sp == especie_mas_abund_global:
            nota = " — **especie más abundante global**"
        top_lines.append(f"{i+1}. {sp} ({nsp} individuos){nota}")
    if not top_lines:
        top_text = "No hay especies registradas en este orden."
    else:
        top_text = "\n".join(top_lines)

    # Construir párrafo narrativo (estilo formal técnico)
    parrafo = (
        f"El orden {ord_name} agrupa un total de {n_especies} especies "
        f"({pct_sp}% del total de especies), y suma {n_individuos} individuos, "
        f"equivalente al {pct_ind}% del total de individuos registrados ({total_individuos}). "
        f"Este orden está representado por las siguientes familias: {familias_text}. "
        f"La familia con mayor riqueza específica dentro del orden fue {fam_por_especies} "
        f"({nsp_fam} especies), mientras que la familia con mayor abundancia en individuos fue {fam_por_ind} "
        f"({nind_fam} individuos). "
        f"Las especies más abundantes dentro de {ord_name} fueron:\n{top_text}."
    )

    # Si la especie más abundante global está en este orden, añadir frase destacada
    if especie_mas_abund_global and especie_mas_abund_global in sp_series.index:
        parrafo += f" Cabe resaltar que **{especie_mas_abund_global}** es también la especie más abundante registrada en todo el muestreo ({especie_mas_abund_global_n} individuos)."

    orden_paragraphs.append(parrafo)

import os
import textwrap

# Ruta fija donde guardar
ruta_resultados = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados"
os.makedirs(ruta_resultados, exist_ok=True)

WRAP_WIDTH = 90  # ancho máximo por línea

# ---------------------------
# Montar texto final con wrap
# ---------------------------
lines = []
lines.append("DESCRIPCIÓN ECOLÓGICA POR ÓRDENES")
lines.append("=" * WRAP_WIDTH)

totales = (
    f"Totales generales: {total_ordenes} órdenes, "
    f"{df['Familia'].nunique()} familias, "
    f"{total_especies} especies y {total_individuos} individuos."
)
lines.extend(textwrap.wrap(totales, width=WRAP_WIDTH))
lines.append("")

# PÁRRAFO GLOBAL
lines.append("PÁRRAFO GLOBAL:")
wrapped_global = textwrap.wrap(parrafo_global.strip(), width=WRAP_WIDTH)
lines.extend(wrapped_global)
lines.append("")

# DETALLE POR ORDEN
lines.append("DETALLE POR ORDEN (ordenado de mayor a menor riqueza específica):")
lines.append("")

for par in orden_paragraphs:
    par = par.strip()
    if par:
        wrapped_par = textwrap.wrap(par, width=WRAP_WIDTH)
        lines.extend(wrapped_par)
        lines.append("-" * WRAP_WIDTH)

# ---------------------------
# Guardar en la carpeta resultados especificada
# ---------------------------
output_filename = os.path.join(ruta_resultados, "2.2_descripcion_ordenes_detal.txt")
with open(output_filename, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Archivo guardado en: {os.path.abspath(output_filename)}")
