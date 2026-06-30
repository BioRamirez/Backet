from qgis.core import *
import processing
import math
import pandas as pd
import os

# ---------------------------
# CONFIGURACIÓN DEL USUARIO
# ---------------------------

kml_layer1 = QgsProject.instance().mapLayersByName("pof_zulia_tr_291025")[0]
kml_layer2 = QgsProject.instance().mapLayersByName("pof_zulia_tr_291025 (#2)")[0]

elev_layer = QgsProject.instance().mapLayersByName("Elevacion Zulia")[0]
veredas_layer = QgsProject.instance().mapLayersByName("Veredas_de_Colombia")[0]
cobertura_layer = QgsProject.instance().mapLayersByName("CoberturaTierraIDEAM2020")[0]

output_excel = "D:/CORPONOR 2025/Backet/python_Proyect/data/Puntos_Transectos_SIG.xlsx"

target_crs = QgsCoordinateReferenceSystem("EPSG:3116")

# ---------------------------------------
# SPATIAL INDEX PARA COBERTURA (RÁPIDO)
# ---------------------------------------
cobertura_index = QgsSpatialIndex(cobertura_layer.getFeatures())

# ---------------------------------------
# FUNCIÓN SEXAGESIMAL
# ---------------------------------------
def decimal_to_sexagesimal(value, lat=True):
    degrees = int(value)
    minutes_float = abs((value - degrees) * 60)
    minutes = int(minutes_float)
    seconds = (minutes_float - minutes) * 60
    hemi = 'N' if (lat and value >= 0) else 'S' if lat else 'E' if value >= 0 else 'W'
    return f"{abs(degrees)}°{minutes}'{seconds:.2f}\"{hemi}"

# ---------------------------------------
# FUNCIÓN PARA PROCESAR UNA CAPA
# ---------------------------------------
def procesar_capa(layer):
    results = []

    # Transformación hacia MAGNA 3116 para intersectar con cobertura
    transform_to_3116 = QgsCoordinateTransform(layer.crs(), cobertura_layer.crs(), QgsProject.instance())
    transform_3116 = QgsCoordinateTransform(layer.crs(), target_crs, QgsProject.instance())

    for feat in layer.getFeatures():

        geom = feat.geometry()
        if geom.isNull():
            continue

        fid = feat.id()
        Name = feat["Name"] if "Name" in feat.fields().names() else None
        obj_id = feat["id"] if "id" in feat.fields().names() else None

        # ---------------------------------------
        # MANEJO DE GEOMETRÍA
        # ---------------------------------------
        if geom.type() == QgsWkbTypes.LineGeometry:
            line = geom.asMultiPolyline()[0] if QgsWkbTypes.isMultiType(geom.wkbType()) else geom.asPolyline()
            start = line[0]
            end = line[-1]
            length_m = geom.length()

        elif geom.type() == QgsWkbTypes.PointGeometry:
            p = geom.asPoint()
            start = end = p
            length_m = 0

        else:
            continue

        # Coordenadas decimales
        lat_i, lon_i = start.y(), start.x()
        lat_f, lon_f = end.y(), end.x()

        # Sexagesimal
        lat_i_sex = decimal_to_sexagesimal(lat_i, True)
        lon_i_sex = decimal_to_sexagesimal(lon_i, False)
        lat_f_sex = decimal_to_sexagesimal(lat_f, True)
        lon_f_sex = decimal_to_sexagesimal(lon_f, False)

        # MAGNA para exportar
        p_i_3116 = transform_3116.transform(QgsPointXY(start))
        p_f_3116 = transform_3116.transform(QgsPointXY(end))

        # ---------------------------------------
        # COTAS DESDE DEM
        # ---------------------------------------
        def sample_elev(pt):
            ident = elev_layer.dataProvider().identify(pt, QgsRaster.IdentifyFormatValue).results()
            return ident[1] if 1 in ident else None

        cota_i = sample_elev(start)
        cota_f = sample_elev(end)

        if cota_i is None or cota_f is None:
            cota = cota_min = cota_max = None
        else:
            cota_min = min(cota_i, cota_f)
            cota_max = max(cota_i, cota_f)
            cota = (cota_min + cota_max) / 2

        # ---------------------------------------
        # INTERSECCIÓN CON VEREDAS
        # ---------------------------------------
        municipio = None
        departamento = None
        vereda_nombre = None

        for v in veredas_layer.getFeatures():
            if v.geometry().contains(QgsPointXY(start)):
                municipio = v["NOMB_MPIO"]
                departamento = v["NOM_DEP"]
                vereda_nombre = v["NOMBRE_VER"]
                break

        # ---------------------------------------
        # INTERSECCIÓN CON COBERTURA CORREGIDA
        # Reproyectamos la geometría del KML a 3116
        # ---------------------------------------
        cobertura = None

        geom_3116 = QgsGeometry(geom)  # copiar
        geom_3116.transform(transform_to_3116)

        ids = cobertura_index.intersects(geom_3116.boundingBox())

        for cid in ids:
            c = cobertura_layer.getFeature(cid)
            if c.geometry().intersects(geom_3116):
                cobertura = c["leyenda"]  # ahora sí funciona porque CRS coincide
                break

        # ---------------------------------------
        # GUARDAR RESULTADOS
        # ---------------------------------------
        results.append({
            "ID1": obj_id,
            "ID": Name,
            "LONG_m": length_m,
            "DEPARTAMENTO": departamento,
            "MUNICIPIO": municipio,
            "VEREDA": vereda_nombre,
            "COBERTURA": cobertura,
            "COTA": cota,
            "COTA_MIN": cota_min,
            "COTA_MAX": cota_max,
            "LAT_decimal_I": lat_i,
            "LONG_decimal_I": lon_i,
            "LAT_decimal_F": lat_f,
            "LONG_decimal_F": lon_f,
            "LAT_sexagesimal_I": lat_i_sex,
            "LONG_sexagesimal_I": lon_i_sex,
            "LAT_sexagesimal_F": lat_f_sex,
            "LONG_sexagesimal_F": lon_f_sex,
            "POINT_Y_MAGNA_I": p_i_3116.y(),
            "POINT_X_MAGNA_I": p_i_3116.x(),
            "POINT_Y_MAGNA_F": p_f_3116.y(),
            "POINT_X_MAGNA_F": p_f_3116.x()
        })

    return results

# ---------------------------
# EJECUTAR PARA DOS CAPAS
# ---------------------------
resultados1 = procesar_capa(kml_layer1)
resultados2 = procesar_capa(kml_layer2)

df = pd.DataFrame(resultados1 + resultados2)

# ----------------------------------------------
# LIMPIAR CÓDIGOS AL INICIO DE LA COBERTURA
# ----------------------------------------------
import re

if "COBERTURA" in df.columns:
    df["COBERTURA"] = df["COBERTURA"].astype(str).apply(
        lambda x: re.sub(r"^[\d\.]+\s*", "", x) if isinstance(x, str) else x
    )


df.to_excel(output_excel, index=False)
print(f"Archivo generado exitosamente en:\n{output_excel}")






































#CUANDO TIENE SOLO TRANSECTOS



from qgis.core import *
import processing
import math
import pandas as pd
import os
import re

# ---------------------------
# CONFIGURACIÓN DEL USUARIO
# ---------------------------

kml_layer1 = QgsProject.instance().mapLayersByName("Transectos_Magallanes")[0]

elev_layer = QgsProject.instance().mapLayersByName("SRTMGL1[Memory]")[0]
veredas_layer = QgsProject.instance().mapLayersByName("Veredas_de_Colombia")[0]
cobertura_layer = QgsProject.instance().mapLayersByName("CoberturaTierraIDEAM2020")[0]

output_excel = "D:/CORPONOR 2025/Backet/python_Proyect/data/Puntos_Transectos_SIG.xlsx"

target_crs = QgsCoordinateReferenceSystem("EPSG:3116")

# ---------------------------------------
# SPATIAL INDEX PARA COBERTURA (RÁPIDO)
# ---------------------------------------
cobertura_index = QgsSpatialIndex(cobertura_layer.getFeatures())

# ---------------------------------------
# FUNCIÓN SEXAGESIMAL
# ---------------------------------------
def decimal_to_sexagesimal(value, lat=True):
    degrees = int(value)
    minutes_float = abs((value - degrees) * 60)
    minutes = int(minutes_float)
    seconds = (minutes_float - minutes) * 60
    hemi = 'N' if (lat and value >= 0) else 'S' if lat else 'E' if value >= 0 else 'W'
    return f"{abs(degrees)}°{minutes}'{seconds:.2f}\"{hemi}"

# ---------------------------------------
# FUNCIÓN PARA PROCESAR UNA CAPA
# ---------------------------------------
def procesar_capa(layer):
    results = []

    transform_to_3116 = QgsCoordinateTransform(layer.crs(), cobertura_layer.crs(), QgsProject.instance())
    transform_3116 = QgsCoordinateTransform(layer.crs(), target_crs, QgsProject.instance())

    for feat in layer.getFeatures():

        geom = feat.geometry()
        if geom.isNull():
            continue

        Name = feat["Name"] if "Name" in feat.fields().names() else None
        obj_id = feat["id"] if "id" in feat.fields().names() else None

        # ---------------------------------------
        # MANEJO DE GEOMETRÍA
        # ---------------------------------------
        if geom.type() == QgsWkbTypes.LineGeometry:
            line = geom.asMultiPolyline()[0] if QgsWkbTypes.isMultiType(geom.wkbType()) else geom.asPolyline()
            start = line[0]
            end = line[-1]
            length_m = geom.length()

        elif geom.type() == QgsWkbTypes.PointGeometry:
            p = geom.asPoint()
            start = end = p
            length_m = 0

        else:
            continue

        # Coordenadas decimales
        lat_i, lon_i = start.y(), start.x()
        lat_f, lon_f = end.y(), end.x()

        # Sexagesimal
        lat_i_sex = decimal_to_sexagesimal(lat_i, True)
        lon_i_sex = decimal_to_sexagesimal(lon_i, False)
        lat_f_sex = decimal_to_sexagesimal(lat_f, True)
        lon_f_sex = decimal_to_sexagesimal(lon_f, False)

        # MAGNA para exportar
        p_i_3116 = transform_3116.transform(QgsPointXY(start))
        p_f_3116 = transform_3116.transform(QgsPointXY(end))

        # ---------------------------------------
        # COTAS DESDE DEM
        # ---------------------------------------
        def sample_elev(pt):
            ident = elev_layer.dataProvider().identify(pt, QgsRaster.IdentifyFormatValue).results()
            return ident[1] if 1 in ident else None

        cota_i = sample_elev(start)
        cota_f = sample_elev(end)

        if cota_i is None or cota_f is None:
            cota = cota_min = cota_max = None
        else:
            cota_min = min(cota_i, cota_f)
            cota_max = max(cota_i, cota_f)
            cota = (cota_min + cota_max) / 2

        # ---------------------------------------
        # INTERSECCIÓN CON VEREDAS
        # ---------------------------------------
        municipio = None
        departamento = None
        vereda_nombre = None

        for v in veredas_layer.getFeatures():
            if v.geometry().contains(QgsPointXY(start)):
                municipio = v["NOMB_MPIO"]
                departamento = v["NOM_DEP"]
                vereda_nombre = v["NOMBRE_VER"]
                break

        # ---------------------------------------
        # INTERSECCIÓN CON COBERTURA CORREGIDA
        # ---------------------------------------
        cobertura = None

        geom_3116 = QgsGeometry(geom)
        geom_3116.transform(transform_to_3116)

        ids = cobertura_index.intersects(geom_3116.boundingBox())

        for cid in ids:
            c = cobertura_layer.getFeature(cid)
            if c.geometry().intersects(geom_3116):
                cobertura = c["leyenda"]
                break

        # ---------------------------------------
        # GUARDAR RESULTADOS
        # ---------------------------------------
        results.append({
            "ID1": obj_id,
            "ID": Name,
            "LONG_m": length_m,
            "DEPARTAMENTO": departamento,
            "MUNICIPIO": municipio,
            "VEREDA": vereda_nombre,
            "COBERTURA": cobertura,
            "COTA": cota,
            "COTA_MIN": cota_min,
            "COTA_MAX": cota_max,
            "LAT_decimal_I": lat_i,
            "LONG_decimal_I": lon_i,
            "LAT_decimal_F": lat_f,
            "LONG_decimal_F": lon_f,
            "LAT_sexagesimal_I": lat_i_sex,
            "LONG_sexagesimal_I": lon_i_sex,
            "LAT_sexagesimal_F": lat_f_sex,
            "LONG_sexagesimal_F": lon_f_sex,
            "POINT_Y_MAGNA_I": p_i_3116.y(),
            "POINT_X_MAGNA_I": p_i_3116.x(),
            "POINT_Y_MAGNA_F": p_f_3116.y(),
            "POINT_X_MAGNA_F": p_f_3116.x()
        })

    return results

# ---------------------------
# EJECUTAR SOLO UNA CAPA
# ---------------------------
resultados = procesar_capa(kml_layer1)

df = pd.DataFrame(resultados)

# ----------------------------------------------
# LIMPIAR CÓDIGOS AL INICIO DE LA COBERTURA
# ----------------------------------------------
if "COBERTURA" in df.columns:
    df["COBERTURA"] = df["COBERTURA"].astype(str).apply(
        lambda x: re.sub(r"^[\d\.]+\s*", "", x) if isinstance(x, str) else x
    )

df.to_excel(output_excel, index=False)
print(f"Archivo generado exitosamente en:\n{output_excel}")














#Cuando las SCR NO SON IGUALES
from qgis.core import *
import pandas as pd
import re

# ---------------------------
# CONFIGURACIÓN
# ---------------------------

kml_layer1 = QgsProject.instance().mapLayersByName("Transecto_Jueves")[0]
kml_layer2 = QgsProject.instance().mapLayersByName("Punto_Muestreo_Jueves")[0]

elev_layer = QgsProject.instance().mapLayersByName("DEM_OsoPardo")[0]
veredas_layer = QgsProject.instance().mapLayersByName("Veredas_de_Colombia")[0]
cobertura_layer = QgsProject.instance().mapLayersByName("Area_influencia")[0]

output_excel = r"D:/CORPONOR 2025/Backet/python_Proyect/data/Puntos_Transectos_SIG2.xlsx"

target_crs = QgsCoordinateReferenceSystem("EPSG:3116")

# Índice espacial
cobertura_index = QgsSpatialIndex(cobertura_layer.getFeatures())

# ---------------------------
# FUNCIONES
# ---------------------------

def decimal_to_sexagesimal(value, lat=True):
    degrees = int(value)
    minutes_float = abs((value - degrees) * 60)
    minutes = int(minutes_float)
    seconds = (minutes_float - minutes) * 60
    hemi = 'N' if (lat and value >= 0) else 'S' if lat else 'E' if value >= 0 else 'W'
    return f"{abs(degrees)}°{minutes}'{seconds:.2f}\"{hemi}"


def transformar_a_magna(pt, layer):
    """Transforma a EPSG:3116 solo si es necesario"""
    if layer.crs().authid() != "EPSG:3116":
        tr = QgsCoordinateTransform(layer.crs(), target_crs, QgsProject.instance())
        return tr.transform(QgsPointXY(pt))
    return QgsPointXY(pt)


# ---------------------------
# FUNCIÓN PRINCIPAL
# ---------------------------

def procesar_capa(layer):

    results = []

    for feat in layer.getFeatures():

        geom = feat.geometry()
        if geom.isNull():
            continue

        # -------------------
        # GEOMETRÍA
        # -------------------
        if geom.type() == QgsWkbTypes.LineGeometry:

            # Geometría original
            line = geom.asPolyline() if not geom.isMultipart() else geom.asMultiPolyline()[0]
            start = line[0]
            end = line[-1]

            # 🔹 Transformar a MAGNA (EPSG:3116) para medir en metros
            geom_m = QgsGeometry(geom)
            tr_len = QgsCoordinateTransform(
            layer.crs(),
            QgsCoordinateReferenceSystem("EPSG:3116"),
            QgsProject.instance()
            )
            geom_m.transform(tr_len)

            length_m = geom_m.length()

        else:
            start = end = geom.asPoint()
            length_m = 0
        # -------------------
        # COORDENADAS
        # -------------------
        if layer.crs().isGeographic():
            lat_i, lon_i = start.y(), start.x()
            lat_f, lon_f = end.y(), end.x()

            lat_i_sex = decimal_to_sexagesimal(lat_i, True)
            lon_i_sex = decimal_to_sexagesimal(lon_i, False)
            lat_f_sex = decimal_to_sexagesimal(lat_f, True)
            lon_f_sex = decimal_to_sexagesimal(lon_f, False)
        else:
            lat_i = lon_i = lat_f = lon_f = None
            lat_i_sex = lon_i_sex = lat_f_sex = lon_f_sex = None

        # -------------------
        # MAGNA
        # -------------------
        p_i_3116 = transformar_a_magna(start, layer)
        p_f_3116 = transformar_a_magna(end, layer)

        # -------------------
        # ELEVACIÓN
        # -------------------
        tr_to_dem = QgsCoordinateTransform(layer.crs(), elev_layer.crs(), QgsProject.instance())
        p_dem = tr_to_dem.transform(QgsPointXY(start))

        ident = elev_layer.dataProvider().identify(
            p_dem, QgsRaster.IdentifyFormatValue
        ).results()

        cota = ident.get(1)
        cota_min = cota_max = cota

        # -------------------
        # VEREDAS
        # -------------------
        municipio = departamento = vereda_nombre = None
        tr_ver = QgsCoordinateTransform(layer.crs(), veredas_layer.crs(), QgsProject.instance())
        p_ver = tr_ver.transform(QgsPointXY(start))

        for v in veredas_layer.getFeatures():
            if v.geometry().contains(QgsGeometry.fromPointXY(p_ver)):
                municipio = v["NOMB_MPIO"]
                departamento = v["NOM_DEP"]
                vereda_nombre = v["NOMBRE_VER"]
                break

        # -------------------
        # COBERTURA
        # -------------------
        cobertura = None
        tr_cob = QgsCoordinateTransform(layer.crs(), cobertura_layer.crs(), QgsProject.instance())
        geom_cob = QgsGeometry(geom)
        geom_cob.transform(tr_cob)

        for cid in cobertura_index.intersects(geom_cob.boundingBox()):
            c = cobertura_layer.getFeature(cid)
            if c.geometry().intersects(geom_cob):
                cobertura = c["Cobertura"]
                break

        # -------------------
        # RESULTADO
        # -------------------
        results.append({
            "ID1": feat["id"] if "id" in feat.fields().names() else None,
            "ID": feat["Name"] if "Name" in feat.fields().names() else None,
            "LONG_m": length_m,
            "DEPARTAMENTO": departamento,
            "MUNICIPIO": municipio,
            "VEREDA": vereda_nombre,
            "COBERTURA": cobertura,
            "COTA": cota,
            "COTA_MIN": cota_min,
            "COTA_MAX": cota_max,
            "LAT_decimal_I": lat_i,
            "LONG_decimal_I": lon_i,
            "LAT_decimal_F": lat_f,
            "LONG_decimal_F": lon_f,
            "LAT_sexagesimal_I": lat_i_sex,
            "LONG_sexagesimal_I": lon_i_sex,
            "LAT_sexagesimal_F": lat_f_sex,
            "LONG_sexagesimal_F": lon_f_sex,
            "POINT_Y_MAGNA_I": p_i_3116.y(),
            "POINT_X_MAGNA_I": p_i_3116.x(),
            "POINT_Y_MAGNA_F": p_f_3116.y(),
            "POINT_X_MAGNA_F": p_f_3116.x()
        })

    return results


# ---------------------------
# EJECUCIÓN
# ---------------------------

res = procesar_capa(kml_layer1) + procesar_capa(kml_layer2)

df = pd.DataFrame(res)

df["COBERTURA"] = df["COBERTURA"].astype(str).apply(
    lambda x: re.sub(r"^[\d\.]+\s*", "", x)
)

df.to_excel(output_excel, index=False)

print("✔ Archivo generado correctamente")
