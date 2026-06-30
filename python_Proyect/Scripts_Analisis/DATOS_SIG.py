# =========================================================
# PROCESAMIENTO SIG - PUNTOS Y TRANSECTOS
# Compatible con capas vectoriales en distintos CRS
# QGIS 3.x
# =========================================================

from qgis.core import *
import pandas as pd
import re
import os

# =========================================================
# CONFIGURACIÓN
# =========================================================

output_excel = r"D:/CORPONOR 2025/Backet/python_Proyect/data/Puntos_Transectos_SIG_SanRoque.xlsx"

# CRS destino MAGNA
target_crs = QgsCoordinateReferenceSystem("EPSG:3116")

# =========================================================
# FUNCIÓN SEGURA PARA CARGAR CAPAS
# =========================================================

def obtener_capa(nombre):

    capas = QgsProject.instance().mapLayersByName(nombre)

    if not capas:
        raise Exception(f"❌ No se encontró la capa: {nombre}")

    print(f"✔ Capa cargada: {nombre}")

    return capas[0]

# =========================================================
# CARGA DE CAPAS
# =========================================================

kml_layer1 = obtener_capa("Transecto_Muestreo_Fauna_LaSuerte")
kml_layer2 = obtener_capa("Punto_Muestreo_Fauna_LaSuerte")

# CURVAS DE NIVEL (VECTORIAL)
elev_layer = obtener_capa("Curvas_Nivel_SanRoque")

veredas_layer = obtener_capa("Veredas_de_Colombia")

cobertura_layer = obtener_capa("SANROQUEPOLIGONO")

# =========================================================
# ÍNDICES ESPACIALES
# =========================================================

print("🚀 Creando índices espaciales...")

cobertura_index = QgsSpatialIndex(cobertura_layer.getFeatures())

veredas_index = QgsSpatialIndex(veredas_layer.getFeatures())

elev_index = QgsSpatialIndex(elev_layer.getFeatures())

print("✔ Índices espaciales creados")

# =========================================================
# FUNCIÓN SEXAGESIMAL
# =========================================================

def decimal_to_sexagesimal(value, lat=True):

    degrees = int(value)

    minutes_float = abs((value - degrees) * 60)

    minutes = int(minutes_float)

    seconds = (minutes_float - minutes) * 60

    if lat:
        hemi = 'N' if value >= 0 else 'S'
    else:
        hemi = 'E' if value >= 0 else 'W'

    return f"{abs(degrees)}°{minutes}'{seconds:.2f}\"{hemi}"

# =========================================================
# TRANSFORMAR A MAGNA
# =========================================================

def transformar_a_magna(pt, layer):

    if layer.crs().authid() != "EPSG:3116":

        tr = QgsCoordinateTransform(
            layer.crs(),
            target_crs,
            QgsProject.instance()
        )

        return tr.transform(QgsPointXY(pt))

    return QgsPointXY(pt)

# =========================================================
# OBTENER COTA DESDE CURVAS DE NIVEL
# =========================================================

def obtener_cota_desde_curvas(point_geom, layer):

    try:

        tr = QgsCoordinateTransform(
            layer.crs(),
            elev_layer.crs(),
            QgsProject.instance()
        )

        punto_transformado = tr.transform(QgsPointXY(point_geom))

        punto_geom_qgs = QgsGeometry.fromPointXY(punto_transformado)

        # Buscar curvas cercanas
        nearest_ids = elev_index.nearestNeighbor(punto_transformado, 5)

        distancia_min = float("inf")

        mejor_cota = None

        for fid in nearest_ids:

            feat = elev_layer.getFeature(fid)

            geom = feat.geometry()

            if geom is None or geom.isNull():
                continue

            distancia = geom.distance(punto_geom_qgs)

            if distancia < distancia_min:

                distancia_min = distancia

                campos = feat.fields().names()

                posibles_campos = [
                    "ELEV",
                    "ELEVACION",
                    "COTA",
                    "ALTURA",
                    "CONTOUR",
                    "Z",
                    "GRID_CODE"
                ]

                for campo in posibles_campos:

                    if campo in campos:

                        mejor_cota = feat[campo]
                        break

        return mejor_cota

    except Exception as e:

        print(f"⚠ Error obteniendo cota: {e}")

        return None

# =========================================================
# OBTENER VEREDA
# =========================================================

def obtener_vereda(point_geom, layer):

    municipio = None
    departamento = None
    vereda_nombre = None

    try:

        tr = QgsCoordinateTransform(
            layer.crs(),
            veredas_layer.crs(),
            QgsProject.instance()
        )

        p_ver = tr.transform(QgsPointXY(point_geom))

        punto_ver = QgsGeometry.fromPointXY(p_ver)

        ids = veredas_index.intersects(
            punto_ver.boundingBox()
        )

        for fid in ids:

            feat = veredas_layer.getFeature(fid)

            if feat.geometry().contains(punto_ver):

                campos = feat.fields().names()

                if "NOMB_MPIO" in campos:
                    municipio = feat["NOMB_MPIO"]

                if "NOM_DEP" in campos:
                    departamento = feat["NOM_DEP"]

                if "NOMBRE_VER" in campos:
                    vereda_nombre = feat["NOMBRE_VER"]

                break

    except Exception as e:

        print(f"⚠ Error obteniendo vereda: {e}")

    return municipio, departamento, vereda_nombre

# =========================================================
# OBTENER COBERTURA
# =========================================================

def obtener_cobertura(geom, layer):

    cobertura = None

    try:

        tr = QgsCoordinateTransform(
            layer.crs(),
            cobertura_layer.crs(),
            QgsProject.instance()
        )

        geom_cob = QgsGeometry(geom)

        geom_cob.transform(tr)

        ids = cobertura_index.intersects(
            geom_cob.boundingBox()
        )

        for fid in ids:

            feat = cobertura_layer.getFeature(fid)

            if feat.geometry().intersects(geom_cob):

                campos = feat.fields().names()

                posibles = [
                    "Cobertura",
                    "COBERTURA",
                    "leyenda",
                    "Leyenda",
                    "DESCRIPCION"
                ]

                for campo in posibles:

                    if campo in campos:

                        cobertura = feat[campo]
                        break

                break

    except Exception as e:

        print(f"⚠ Error obteniendo cobertura: {e}")

    return cobertura

# =========================================================
# FUNCIÓN PRINCIPAL
# =========================================================

def procesar_capa(layer):

    resultados = []

    print(f"\n🚀 Procesando capa: {layer.name()}")

    for feat in layer.getFeatures():

        try:

            geom = feat.geometry()

            if geom is None or geom.isNull():
                continue

            # =================================================
            # GEOMETRÍA
            # =================================================

            if geom.type() == QgsWkbTypes.LineGeometry:

                if geom.isMultipart():

                    multi = geom.asMultiPolyline()

                    if not multi:
                        continue

                    line = multi[0]

                else:

                    line = geom.asPolyline()

                if not line:
                    continue

                start = line[0]

                end = line[-1]

                # LONGITUD EN METROS
                geom_m = QgsGeometry(geom)

                tr_len = QgsCoordinateTransform(
                    layer.crs(),
                    target_crs,
                    QgsProject.instance()
                )

                geom_m.transform(tr_len)

                length_m = geom_m.length()

            elif geom.type() == QgsWkbTypes.PointGeometry:

                start = end = geom.asPoint()

                length_m = 0

            else:

                continue

            # =================================================
            # COORDENADAS
            # =================================================

            if layer.crs().isGeographic():

                lat_i, lon_i = start.y(), start.x()

                lat_f, lon_f = end.y(), end.x()

                lat_i_sex = decimal_to_sexagesimal(lat_i, True)

                lon_i_sex = decimal_to_sexagesimal(lon_i, False)

                lat_f_sex = decimal_to_sexagesimal(lat_f, True)

                lon_f_sex = decimal_to_sexagesimal(lon_f, False)

            else:

                lat_i = lon_i = None

                lat_f = lon_f = None

                lat_i_sex = None
                lon_i_sex = None

                lat_f_sex = None
                lon_f_sex = None

            # =================================================
            # MAGNA
            # =================================================

            p_i_3116 = transformar_a_magna(start, layer)

            p_f_3116 = transformar_a_magna(end, layer)

            # =================================================
            # COTA
            # =================================================

            cota_i = obtener_cota_desde_curvas(start, layer)

            cota_f = obtener_cota_desde_curvas(end, layer)

            if cota_i is not None and cota_f is not None:

                cota_min = min(cota_i, cota_f)

                cota_max = max(cota_i, cota_f)

                cota = (cota_min + cota_max) / 2

            else:

                cota = None
                cota_min = None
                cota_max = None

            # =================================================
            # VEREDA
            # =================================================

            municipio, departamento, vereda_nombre = obtener_vereda(
                start,
                layer
            )

            # =================================================
            # COBERTURA
            # =================================================

            cobertura = obtener_cobertura(
                geom,
                layer
            )

            # =================================================
            # IDs
            # =================================================

            campos = feat.fields().names()

            obj_id = feat["id"] if "id" in campos else None

            nombre = feat["Name"] if "Name" in campos else None

            # =================================================
            # RESULTADOS
            # =================================================

            resultados.append({

                "ID1": obj_id,

                "ID": nombre,

                "LONG_m": round(length_m, 2),

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

        except Exception as e:

            print(f"⚠ Error procesando feature {feat.id()}: {e}")

    print(f"✔ Finalizada capa: {layer.name()}")

    return resultados

# =========================================================
# EJECUCIÓN
# =========================================================

print("\n🚀 INICIANDO PROCESAMIENTO...\n")

resultados1 = procesar_capa(kml_layer1)

resultados2 = procesar_capa(kml_layer2)

df = pd.DataFrame(resultados1 + resultados2)

# =========================================================
# LIMPIAR COBERTURA
# =========================================================

if "COBERTURA" in df.columns:

    df["COBERTURA"] = df["COBERTURA"].astype(str).apply(
        lambda x: re.sub(r"^[\d\.]+\s*", "", x)
    )

# =========================================================
# EXPORTAR EXCEL
# =========================================================

os.makedirs(os.path.dirname(output_excel), exist_ok=True)

df.to_excel(output_excel, index=False)

print("\n✔ ARCHIVO EXCEL GENERADO CORRECTAMENTE")

print(output_excel)

print("\n🎉 PROCESO FINALIZADO")