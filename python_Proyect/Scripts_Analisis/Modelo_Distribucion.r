# ============================================================
# DESCARGA Y LIMPIEZA DE DATOS DE OCURRENCIA DESDE GBIF
# Autor: [Tu nombre]
# Especie: Sternoclyta cyanopectus (Gould, 1846)
# ============================================================

# Paquetes
if(!require("spocc")) install.packages("spocc")
if(!require("dplyr")) install.packages("dplyr")
library(spocc)
library(dplyr)

# Descargar datos de GBIF
df <- occ(query = "Erythrolamprus zweifeli (Roze, 1959)", from = "gbif")

# Extraer el primer objeto de datos (evita problemas con nombres largos)
sp1 <- df$gbif$data[[1]]

# Filtrar coordenadas válidas
esp_geo1 <- subset(sp1,
                   !is.na(name) & !is.na(longitude) & !is.na(latitude) &
                     longitude >= -180 & longitude <= 180 &
                     latitude >= -90 & latitude <= 90)

# Crear data frame limpio
especie1 <- data.frame(
  name = esp_geo1$name,
  longitude = as.numeric(esp_geo1$longitude),
  latitude = as.numeric(esp_geo1$latitude)
)

# Eliminar duplicados
especie_limpia <- especie1 %>%
  distinct(longitude, latitude, .keep_all = TRUE)

# Añadir nombre científico
nombcientif <- "Erythrolamprus zweifeli"
midatos <- data.frame(
  nombcientif = nombcientif,
  longitude = especie_limpia$longitude,
  latitude = especie_limpia$latitude
)

# Exportar CSV
write.csv(midatos, "Erythrolamprus zweifeli.csv", row.names = FALSE)

# Mostrar resumen
cat("Datos limpios guardados correctamente.\n")
cat("Número de registros finales:", nrow(midatos), "\n")
head(midatos)


# ============================================================
# SDM WORKFLOW: desde CSV de GBIF -> Maxent (maxnet) con tuning ENMeval
# Especie: Sternoclyta cyanopectus
# Archivo de entrada esperado: "Sternoclyta_cyanopectus_GBIF.csv"
# Salidas: raster de predicción, CSV con ocurrencias usadas, resultados de ENMeval
# ============================================================

# 0) Paquetes ------------------------------------------------
pkgs <- c("terra","sf","dplyr","maxnet","ENMeval","blockCV","spThin","usdm","ggplot2","readr")
install.packages(setdiff(pkgs, installed.packages()[,"Package"]), dependencies = TRUE)
library(terra); library(sf); library(dplyr); library(maxnet)
library(ENMeval); library(blockCV); library(spThin); library(usdm); library(ggplot2); library(readr)

set.seed(42)

# 1) Cargar ocurrencias limpias (CSV que generaste) ---------------------------
occ_csv <- "Erythrolamprus zweifeli.csv"
occ <- read_csv(occ_csv, show_col_types = FALSE)

# Revisar columnas esperadas
if(!all(c("nombcientif","longitude","latitude") %in% names(occ))){
  stop("El CSV debe tener columnas: nombcientif, longitude, latitude")
}

# Convertir a sf (WGS84)
occs_wgs <- st_as_sf(occ, coords = c("longitude","latitude"), crs = 4326, remove = FALSE)

# Guardar copia
write_csv(occs_wgs %>% st_drop_geometry(), "occurrences_cleaned_for_SDM.csv")

# 2) Cargar capas ambientales (carpeta 'env/' con .tif) -------------------------
env_files <- list.files("env/", pattern = "\\.tif$", full.names = TRUE)
if(length(env_files) == 0) stop("No se encontraron .tif en la carpeta 'env/'. Coloca tus capas allí.")
env_stack <- rast(env_files)  # objeto SpatRaster (terra)







# Instala si no lo tienes
install.packages("geodata")
library(geodata)
library(terra)

# Descargar variables bioclimáticas para Colombia (resolución ~1 km)
env_stack <- geodata::worldclim_global(var = "bio", res = 0.5, path = "env") 
# 0.5 = 30 arc-seconds (~1 km)
# Si quieres todo el mundo, usa worldclim_global(); para país, usa crop()

# Recorta al país (Colombia)
colombia <- geodata::gadm(country = "COL", level = 0, path = "env")
env_col <- crop(env_stack, ext(colombia))
env_col <- mask(env_col, colombia)

# Guarda las capas recortadas
dir.create("env", showWarnings = FALSE)
for (i in 1:nlyr(env_col)) {
  writeRaster(env_col[[i]], filename = paste0("env/BIO", i, ".tif"), overwrite = TRUE)
}

# Revisa
plot(env_col[[1]])


#2) DESCARGAR VARIABLES BIOCLIMÁTICAS Y ELEVACIÓN
# -------------------------------------------------------

cat("🌡️ Descargando variables bioclimáticas y elevación para Sudamérica...\n")

# Descargar todas las variables BIO1–BIO19 (~1 km)
bio <- geodata::worldclim_global(var = "bio", res = 0.5, path = "env")

# Descargar capa de elevación (altitud)
elev <- geodata::elevation_global(res = 0.5, path = "env")

# -------------------------------------------------------
# 3) RECORTAR LAS CAPAS A SUDAMÉRICA
# -------------------------------------------------------

cat("✂️ Recortando capas a Sudamérica...\n")

# Polígono de Sudamérica (nivel continental)
south_america <- geodata::gadm("South America", level = 0, path = "env")

# Recortar y enmascarar
bio_sa <- crop(bio, ext(south_america))
bio_sa <- mask(bio_sa, south_america)

elev_sa <- crop(elev, ext(south_america))
elev_sa <- mask(elev_sa, south_america)

# Añadir elevación al stack
env_stack <- c(bio_sa, elev_sa)

# -------------------------------------------------------
# 4) GUARDAR TODAS LAS CAPAS .TIF
# -------------------------------------------------------

cat("💾 Guardando capas recortadas...\n")

dir.create("env_southamerica", showWarnings = FALSE)
for (i in 1:nlyr(env_stack)) {
  writeRaster(env_stack[[i]], filename = paste0("env_southamerica/", names(env_stack)[i], ".tif"), overwrite = TRUE)
}

# -------------------------------------------------------
# 5) VISUALIZACIÓN RÁPIDA
# -------------------------------------------------------

plot(env_stack[[1]], main = "BIO1 - Temperatura media anual (Sudamérica)")
points(occ_points, pch = 20, col = "red", cex = 0.6)

cat("\n✅ Todo listo. Capas guardadas en 'env_southamerica/' y ocurrencias en 'data/'.\n")
cat("Puedes usar estos objetos directamente en MaxEnt o ENMeval.\n")


#✅ Qué obtendrás


library(dismo)

# Cargar capas
env_files <- list.files("env_southamerica", pattern = "\\.tif$", full.names = TRUE)
env_stack <- rast(env_files)

# Leer puntos
occ <- read.csv("data/Erythrolamprus zweifeli_GBIF.csv")

# Crear modelo MaxEnt (asegúrate de tener java instalado)
mx <- maxent(x = env_stack, p = occ[, c("decimalLongitude", "decimalLatitude")])

# Predecir distribución
pred <- predict(env_stack, mx)

# Ver el mapa
plot(pred)
points(occ[, c("decimalLongitude", "decimalLatitude")], pch = 20, col = "red")



























# 3) Asegurar CRS coincidente: reproyectar ocurrencias a CRS de las capas -----
env_crs <- crs(env_stack)  # puede ser PROJ string
occs <- st_transform(occs_wgs, env_crs)  # ahora en misma CRS que env_stack

# 4) Eliminar duplicados espaciales (idénticas coordenadas en raster) -------
# Convertimos a coordenadas de las capas (para evitar duplicados por distinta precisión)
coords <- st_coordinates(occs)
coords_df <- as.data.frame(coords)
names(coords_df) <- c("X","Y")
occs <- bind_cols(occs, coords_df)

occs <- occs %>% distinct(X, Y, .keep_all = TRUE)
cat("Registros únicos espacialmente:", nrow(occs), "\n")

# 5) Verificar NAs al extraer valores ambientales ------------------------------
vals_pres <- terra::extract(env_stack, vect(occs))
na_check <- apply(vals_pres, 1, function(x) any(is.na(x)))
if(any(na_check)){
  warning(sum(na_check), " presencias caen en celdas con NA en al menos una variable ambiental.")
  # eliminar o revisar
  occs <- occs[!na_check, ]
  vals_pres <- vals_pres[!na_check, ]
  cat("Registros restantes tras quitar NA en ambientales:", nrow(occs), "\n")
}

# 6) Reproyectar a UTM local para operaciones en metros (thinning, buffer) ---
# Función para obtener UTM zona desde longitud media (en WGS84)
mean_lon <- mean(st_coordinates(st_transform(occs, 4326))[,1])
utm_zone <- floor((mean_lon + 180) / 6) + 1
utm_crs <- paste0("+proj=utm +zone=", utm_zone, ifelse(mean(st_coordinates(st_transform(occs,4326))[,2])<0," +south",""), " +datum=WGS84 +units=m +no_defs")

occs_utm <- st_transform(occs, crs = utm_crs)

# 7) Thinning espacial (distancia en metros) ----------------------------------
# Ajusta thin_dist a la escala ecológica de la especie; por ejemplo 1000 m
thin_dist <- 1000  # metros; cambia si tu especie tiene otro rango
# spThin espera data.frame con columnas 'Latitude' y 'Longitude' en grados o en las unidades que le des.
# Aquí usamos las coordenadas UTM (X,Y en metros) y el paquete spThin requiere columnas llamadas "Longitude","Latitude" (pero acepta cualquier numerico)
coords_thin_df <- as.data.frame(st_coordinates(occs_utm))
colnames(coords_thin_df) <- c("Longitude","Latitude")
thin_res <- spThin::thin(loc.data = coords_thin_df,
                         lat.col = "Latitude", long.col = "Longitude",
                         spec.col = NULL,
                         thin.par = thin_dist,
                         reps = 1,
                         locs.thinned.list.return = TRUE)

if(length(thin_res) == 0) stop("Thinning no retornó puntos. Reduce thin_dist.")

coords_thinned <- thin_res[[1]]
occs_thinned_utm <- st_as_sf(as.data.frame(coords_thinned), coords = c("Longitude","Latitude"), crs = utm_crs)
# Volvemos a la proyección de las capas ambientales
occs_thinned <- st_transform(occs_thinned_utm, crs = env_crs)

cat("Registros después de thinning:", nrow(occs_thinned), "\n")

# 8) Definir área M (background) - ejemplo: buffer de 50 km alrededor de presencias
m_buffer <- 50000  # 50 km; ajusta según biología / accesibilidad
m_poly_utm <- st_union(occs_thinned_utm) %>% st_buffer(dist = m_buffer)
m_poly <- st_transform(m_poly_utm, crs = env_crs)

# Visual check opcional (ggplot)
# plot(m_poly); plot(st_geometry(occs_thinned), add=TRUE)

# 9) Generar puntos de background (aleatorio dentro de M)
n_bg <- 10000  # número de background; reduce si problemas de memoria
bg_pts <- st_sample(m_poly, size = n_bg, type = "random")
bg_pts <- st_as_sf(bg_pts)
bg_coords <- st_coordinates(bg_pts)

# 10) Extraer valores ambientales para background (para VIF y modelado)
env_vals_bg <- terra::extract(env_stack, vect(bg_pts))
env_vals_bg_df <- as.data.frame(env_vals_bg)
env_vals_bg_df <- env_vals_bg_df[complete.cases(env_vals_bg_df), ]  # quitar filas con NA

# 11) Selección de variables - VIF (usdm::vifstep)
# Convertir a data.frame y evitar columnas no numéricas
env_vals_bg_df2 <- env_vals_bg_df[, sapply(env_vals_bg_df, is.numeric)]
vif_res <- usdm::vifstep(env_vals_bg_df2, th = 10)  # th=10 típico; puedes bajar a 5 para mayor strictness
selected_vars <- vif_res@results$Variables
cat("Variables seleccionadas por VIF:\n"); print(selected_vars)

env_sel <- env_stack[[selected_vars]]

# 12) Particionado espacial con blockCV ---------------------------------------
# blockCV necesita un data.frame con coordenadas en el mismo CRS que especies; pasamos coords en env_crs
pres_coords <- st_coordinates(occs_thinned)
pres_df_for_block <- data.frame(x = pres_coords[,1], y = pres_coords[,2])

# crear bloques espaciales (theRange en metros — usar m_buffer ó similar)
theRange <- 50000  # 50 km; ajustar según escala
sb <- spatialBlock(speciesData = pres_df_for_block,
                   species = rep(1, nrow(pres_df_for_block)),
                   theRange = theRange,
                   k = 5,
                   selection = "random",
                   progress = FALSE)

# sb$folds contiene las particiones con indices
# Preparamos objecto partitions para ENMeval: ENMevaluate acepta 'method="block"' y usa internamente
# pero también puede recibir bg.coords y partitions. Para simplificar usaremos method="block" y bg.coords.
cat("BlockCV creado. Número de folds:", length(sb$folds), "\n")

# 13) Preparar datos para ENMeval/maxnet: occs & bg coords (en matriz)
occ_xy <- st_coordinates(occs_thinned)           # occ en CRS de env
bg_xy <- bg_coords                               # background coords (ya en env_crs)
# convertir a matrices numéricas (x,y)
occ_xy_mat <- as.matrix(occ_xy[,1:2])
bg_xy_mat <- as.matrix(bg_xy[,1:2])

# 14) Tuning con ENMeval (maxnet backend) -------------------------------------
# Ajusta RMvalues y fc según tiempo/recursos
rm_vals <- seq(0.5, 3, 0.5)
fc_options <- c("L","LQ","H","LQH","LQHP")

# ENMevaluate puede ser costoso; parallel=TRUE si tienes cores
tune <- ENMevaluate(occ = occ_xy_mat,
                    env = env_sel,
                    bg.coords = bg_xy_mat,
                    method = "block",
                    RMvalues = rm_vals,
                    fc = fc_options,
                    algorithm = "maxnet",
                    parallel = FALSE)  # cambia a TRUE si configuras multicore

# Guardar resultados
write.csv(tune@results, "ENMeval_results.csv", row.names = FALSE)
cat("ENMeval completado. Resultados guardados en ENMeval_results.csv\n")

# 15) Seleccionar mejor modelo (por AICc si existe, sino por AUCtrain)
res_df <- tune@results
# Si AICc está presente, seleccionar por AICc; sino por avg.test.AUC
best_idx <- if("AICc" %in% names(res_df)) which.min(res_df$AICc) else which.max(res_df$avg.test.AUC)
best_row <- res_df[best_idx, ]
cat("Mejor configuración seleccionada:\n"); print(best_row)

best_fc <- as.character(best_row$fc)
best_rm <- as.numeric(best_row$RM)

# 16) Preparar tabla de entrenamiento para maxnet ------------------------------
# Extraer valores ambientales para presencias y background (usando env_sel)
pres_vals <- terra::extract(env_sel, vect(occs_thinned))
bg_vals <- terra::extract(env_sel, vect(bg_pts))

pres_df <- as.data.frame(pres_vals); pres_df$occ <- 1
bg_df2 <- as.data.frame(bg_vals); bg_df2$occ <- 0

train_df <- rbind(pres_df, bg_df2)
train_df <- train_df[complete.cases(train_df), ]  # quitar filas con NA

# 17) Ajustar modelo maxnet final con parámetros óptimos ---------------------
# formula según feature classes
f_formula <- maxnet::maxnet.formula(p = train_df$occ, data = train_df[, selected_vars], classes = best_fc)
model_maxnet <- maxnet::maxnet(p = train_df$occ, data = train_df[, selected_vars], f = f_formula, regmult = best_rm)

# 18) Importancia de variables y curvas de respuesta --------------------------
varimp <- maxnet::var.importance(model_maxnet)
print(varimp)

# Graficar respuesta para cada variable (ejemplo: primeros 4)
par(mfrow = c(2,2))
for(i in seq_len(min(4, length(selected_vars)))){
  plot(maxnet::response(model_maxnet, var = selected_vars[i]), main = selected_vars[i])
}
par(mfrow = c(1,1))

# 19) Predicción raster (creando vector de valores y luego reconstruyendo raster)
# Obtener valores de env_sel como data.frame (por celdas)
env_vals_all <- as.data.frame(values(env_sel, dataframe=TRUE))
# Predecir con maxnet (type = "cloglog" recomendado)
pred_values <- predict(model_maxnet, env_vals_all, type = "cloglog")

# Reconstruir raster de predicción
# Tomamos una capa base de env_sel
base_r <- env_sel[[1]]
pred_r <- base_r
# setValues espera un vector con longitud igual al número de celdas
pred_r[] <- pred_values

# Guardar raster
terra::writeRaster(pred_r, filename = "maxent_cloglog_prediction.tif", overwrite = TRUE)
cat("Raster de predicción guardado: maxent_cloglog_prediction.tif\n")

# 20) Evaluación final simple: calcular AUC sobre particiones (ENMeval ya lo hizo)
# podemos guardar las predicciones por particion (ENMeval tiene tune@predictions)
# Guardar modelo y objetos relevantes
saveRDS(model_maxnet, file = "maxnet_model_final.rds")
saveRDS(tune, file = "ENMeval_object.rds")
write.csv(as.data.frame(pred_r, xy = TRUE), "prediction_values_xy.csv", row.names = FALSE)

cat("Workflow completado. Archivos generados:\n",
    "- occurrences_cleaned_for_SDM.csv\n",
    "- ENMeval_results.csv\n",
    "- maxent_cloglog_prediction.tif\n",
    "- maxnet_model_final.rds\n",
    "- prediction_values_xy.csv\n")