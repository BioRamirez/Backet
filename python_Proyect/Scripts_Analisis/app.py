# =========================================================
# SISTEMA FINANCIERO PROFESIONAL - JUAN CARLOS
# =========================================================
# Autor: ChatGPT
# Requiere:
# pip install pandas openpyxl xlsxwriter
#
# EJECUTAR:
# python sistema_financiero.py
#
# GENERA:
# Sistema_Financiero_Juan.xlsx
# =========================================================

import pandas as pd
import xlsxwriter
from datetime import datetime

# =========================================================
# ARCHIVO SALIDA
# =========================================================

archivo = "Sistema_Financiero_Juan.xlsx"

# =========================================================
# DATOS INICIALES
# =========================================================

datos = [
    ["10/04/2026","Entrada","Consultoría","Corponor","Pago contrato",3038000,"Transferencia","Juan"],
    ["14/04/2026","Gasto","Consultoría","Sacyr","Gasolina",20359,"Efectivo","Juan"],
    ["15/04/2026","Entrada","Yogurt","Camilo","Venta yogurt",80000,"Nequi","Camilo"],
    ["16/04/2026","Gasto","Yogurt","Producción","Leche yogurt",30280,"Efectivo","Deisy"],
    ["17/04/2026","Gasto","Personal","Pareja","Cena",25000,"Efectivo","Juan"],
]

columnas = [
    "Fecha",
    "Tipo",
    "Area",
    "Proyecto",
    "Descripcion",
    "Valor",
    "Metodo",
    "Responsable"
]

df = pd.DataFrame(datos, columns=columnas)

# =========================================================
# CREAR EXCEL
# =========================================================

writer = pd.ExcelWriter(
    archivo,
    engine='xlsxwriter'
)

# =========================================================
# EXPORTAR REGISTROS
# =========================================================

df.to_excel(writer, sheet_name="REGISTROS", index=False)

workbook = writer.book

# =========================================================
# FORMATOS
# =========================================================

titulo = workbook.add_format({
    'bold': True,
    'font_size': 18,
    'font_color': 'white',
    'bg_color': '#1F4E78',
    'align': 'center',
    'valign': 'vcenter'
})

header = workbook.add_format({
    'bold': True,
    'bg_color': '#D9EAF7',
    'border': 1,
    'align': 'center'
})

money = workbook.add_format({
    'num_format': '$#,##0',
    'border': 1
})

normal = workbook.add_format({
    'border': 1
})

verde = workbook.add_format({
    'bg_color': '#C6EFCE',
    'font_color': '#006100'
})

rojo = workbook.add_format({
    'bg_color': '#FFC7CE',
    'font_color': '#9C0006'
})

# =========================================================
# HOJA REGISTROS
# =========================================================

ws = writer.sheets["REGISTROS"]

ws.set_column("A:A", 15)
ws.set_column("B:D", 18)
ws.set_column("E:E", 30)
ws.set_column("F:F", 18)
ws.set_column("G:H", 18)

for col_num, value in enumerate(df.columns.values):
    ws.write(0, col_num, value, header)

# =========================================================
# DASHBOARD GENERAL
# =========================================================

dashboard = workbook.add_worksheet("DASHBOARD")

dashboard.merge_range("A1:F2",
                      "SISTEMA FINANCIERO PERSONAL Y PROFESIONAL",
                      titulo)

# KPIs

dashboard.write("A5", "Dinero Total", header)
dashboard.write_formula("B5",
                        '=SUMIF(REGISTROS!B:B,"Entrada",REGISTROS!F:F)-SUMIF(REGISTROS!B:B,"Gasto",REGISTROS!F:F)',
                        money)

dashboard.write("A6", "Entradas Totales", header)
dashboard.write_formula("B6",
                        '=SUMIF(REGISTROS!B:B,"Entrada",REGISTROS!F:F)',
                        money)

dashboard.write("A7", "Gastos Totales", header)
dashboard.write_formula("B7",
                        '=SUMIF(REGISTROS!B:B,"Gasto",REGISTROS!F:F)',
                        money)

dashboard.write("A8", "Consultoría", header)
dashboard.write_formula("B8",
                        '=SUMIFS(REGISTROS!F:F,REGISTROS!C:C,"Consultoría",REGISTROS!B:B,"Entrada")-SUMIFS(REGISTROS!F:F,REGISTROS!C:C,"Consultoría",REGISTROS!B:B,"Gasto")',
                        money)

dashboard.write("A9", "Yogurt", header)
dashboard.write_formula("B9",
                        '=SUMIFS(REGISTROS!F:F,REGISTROS!C:C,"Yogurt",REGISTROS!B:B,"Entrada")-SUMIFS(REGISTROS!F:F,REGISTROS!C:C,"Yogurt",REGISTROS!B:B,"Gasto")',
                        money)

dashboard.write("A10", "Personal", header)
dashboard.write_formula("B10",
                        '=SUMIFS(REGISTROS!F:F,REGISTROS!C:C,"Personal",REGISTROS!B:B,"Gasto")',
                        money)

dashboard.write("D5", "Distribución Recomendada", header)

dashboard.write("D6", "Operación 40%")
dashboard.write_formula("E6", "=B5*0.4", money)

dashboard.write("D7", "Personal 30%")
dashboard.write_formula("E7", "=B5*0.3", money)

dashboard.write("D8", "Ahorro 20%")
dashboard.write_formula("E8", "=B5*0.2", money)

dashboard.write("D9", "Yogurt 10%")
dashboard.write_formula("E9", "=B5*0.1", money)

# =========================================================
# GRAFICO
# =========================================================

chart = workbook.add_chart({'type': 'pie'})

chart.add_series({
    'name': 'Distribución',
    'categories': '=DASHBOARD!$D$6:$D$9',
    'values': '=DASHBOARD!$E$6:$E$9'
})

chart.set_title({'name': 'Distribución Financiera'})
chart.set_style(10)

dashboard.insert_chart('G5', chart)

# =========================================================
# CONSULTORIA
# =========================================================

consultoria = workbook.add_worksheet("CONSULTORIA")

consultoria.merge_range("A1:F2",
                        "ANALISIS CONSULTORIA",
                        titulo)

consultoria.write_row("A5",
                      ["Proyecto","Ingresos","Gastos","Utilidad","Margen"],
                      header)

proyectos = ["Corponor","Sacyr","San Roque"]

fila = 5

for p in proyectos:

    consultoria.write(fila,0,p)

    consultoria.write_formula(
        fila,1,
        f'=SUMIFS(REGISTROS!F:F,REGISTROS!D:D,"{p}",REGISTROS!B:B,"Entrada")',
        money
    )

    consultoria.write_formula(
        fila,2,
        f'=SUMIFS(REGISTROS!F:F,REGISTROS!D:D,"{p}",REGISTROS!B:B,"Gasto")',
        money
    )

    consultoria.write_formula(
        fila,3,
        f'=B{fila+1}-C{fila+1}',
        money
    )

    consultoria.write_formula(
        fila,4,
        f'=IF(B{fila+1}=0,0,D{fila+1}/B{fila+1})'
    )

    fila +=1

# =========================================================
# YOGURT
# =========================================================

yogurt = workbook.add_worksheet("YOGURT")

yogurt.merge_range("A1:F2",
                   "NEGOCIO YOGURT",
                   titulo)

yogurt.write_row("A5",
                 ["Indicador","Valor"],
                 header)

yogurt.write("A6","Ingresos")
yogurt.write_formula(
    "B6",
    '=SUMIFS(REGISTROS!F:F,REGISTROS!C:C,"Yogurt",REGISTROS!B:B,"Entrada")',
    money
)

yogurt.write("A7","Gastos")
yogurt.write_formula(
    "B7",
    '=SUMIFS(REGISTROS!F:F,REGISTROS!C:C,"Yogurt",REGISTROS!B:B,"Gasto")',
    money
)

yogurt.write("A8","Utilidad")
yogurt.write_formula(
    "B8",
    '=B6-B7',
    money
)

yogurt.write("A9","Margen")
yogurt.write_formula(
    "B9",
    '=IF(B6=0,0,B8/B6)'
)

yogurt.write("A10","Reinversión sugerida")
yogurt.write_formula(
    "B10",
    '=B8*0.4',
    money
)

# =========================================================
# PERSONAL
# =========================================================

personal = workbook.add_worksheet("PERSONAL")

personal.merge_range("A1:F2",
                     "VIDA PERSONAL Y BIENESTAR",
                     titulo)

personal.write_row("A5",
                   ["Categoría","Valor"],
                   header)

categorias = [
    "Pareja",
    "Bienestar",
    "Social",
    "Descanso",
    "Emocional"
]

fila = 5

for c in categorias:

    personal.write(fila,0,c)

    personal.write_formula(
        fila,1,
        f'=SUMIFS(REGISTROS!F:F,REGISTROS!D:D,"{c}")',
        money
    )

    fila +=1

# =========================================================
# PRESUPUESTO
# =========================================================

presupuesto = workbook.add_worksheet("PRESUPUESTO")

presupuesto.merge_range("A1:F2",
                        "PRESUPUESTO KAKEIBO",
                        titulo)

presupuesto.write_row("A5",
                      ["Categoría","Presupuesto","Real","Diferencia"],
                      header)

presupuesto_data = [
    ["Supervivencia",1200000],
    ["Operación",800000],
    ["Crecimiento",500000],
    ["Bienestar",300000]
]

fila = 5

for item in presupuesto_data:

    presupuesto.write(fila,0,item[0])
    presupuesto.write(fila,1,item[1],money)

    presupuesto.write_formula(
        fila,2,
        '=SUMIF(REGISTROS!B:B,"Gasto",REGISTROS!F:F)',
        money
    )

    presupuesto.write_formula(
        fila,3,
        f'=B{fila+1}-C{fila+1}',
        money
    )

    fila +=1

# =========================================================
# METAS
# =========================================================

metas = workbook.add_worksheet("METAS")

metas.merge_range("A1:F2",
                  "CRECIMIENTO Y FUTURO",
                  titulo)

metas.write_row("A5",
                ["Meta","Objetivo","Ahorrado","Faltante","%"],
                header)

metas_data = [
    ["Fondo Emergencia",5000000],
    ["Posgrado",25000000],
    ["Laptop SIG",4500000],
    ["Equipos Campo",3200000],
]

fila = 5

for meta in metas_data:

    metas.write(fila,0,meta[0])
    metas.write(fila,1,meta[1],money)

    metas.write_formula(
        fila,2,
        '=DASHBOARD!E8',
        money
    )

    metas.write_formula(
        fila,3,
        f'=B{fila+1}-C{fila+1}',
        money
    )

    metas.write_formula(
        fila,4,
        f'=C{fila+1}/B{fila+1}'
    )

    fila +=1

# =========================================================
# FORMATO CONDICIONAL
# =========================================================

presupuesto.conditional_format(
    'D6:D9',
    {
        'type': 'cell',
        'criteria': '>=',
        'value': 0,
        'format': verde
    }
)

presupuesto.conditional_format(
    'D6:D9',
    {
        'type': 'cell',
        'criteria': '<',
        'value': 0,
        'format': rojo
    }
)

# =========================================================
# FINALIZAR
# =========================================================

writer.close()

print("\n✅ SISTEMA FINANCIERO CREADO EXITOSAMENTE")
print("📁 Archivo generado:")
print("Sistema_Financiero_Juan.xlsx")
print("\n🚀 Ya puedes abrirlo en Excel.")