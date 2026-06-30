# -------------------------------------------------------
# MODELO DE DISTRIBUCIÓN DE ESPECIES - Sternoclyta cyanopectus
# Descarga de datos de ocurrencia y variables ambientales (Sudamérica)
# -------------------------------------------------------

# Instalar y cargar librerías necesarias
if(!require(rgbif)) install.packages("rgbif")
if(!require(terra)) install.packages("terra")
if(!require(geodata)) install.packages("geodata")
if(!require(dplyr)) install.packages("dplyr")

library(rgbif)
library(terra)
library(geodata)
library(dplyr)

# -------------------------------------------------------
# 1) DESCARGAR DATOS DE OCURRENCIA DESDE GBIF
# -------------------------------------------------------

species_name <- "Anadia pamplonensis"

cat("📥 Descargando ocurrencias de GBIF...\n")
gbif_data <- occ_search(scientificName = species_name, hasCoordinate = TRUE, limit = 20000)

# Extraer coordenadas y limpiar datos
occ <- gbif_data$data %>%
  select(species, decimalLongitude, decimalLatitude, countryCode) %>%
  filter(!is.na(decimalLongitude) & !is.na(decimalLatitude))

nrow(gbif_data$data)

table(is.na(gbif_data$data$decimalLongitude))
table(is.na(gbif_data$data$countryCode))

# Eliminar duplicados geográficos
occ_unique <- occ %>%
  distinct(decimalLongitude, decimalLatitude, .keep_all = TRUE)

cat("✅ Registros únicos descargados:", nrow(occ_unique), "\n")

# Guardar los datos de ocurrencia limpios
dir.create("data", showWarnings = FALSE)
write.csv(occ_unique, "data/Sternoclyta_cyanopectus_GBIF.csv", row.names = FALSE)

# Crear objeto SpatVector para visualización
occ_points <- vect(occ_unique, geom = c("decimalLongitude", "decimalLatitude"), crs = "EPSG:4326")

# -------------------------------------------------------
# 2) DESCARGAR VARIABLES BIOCLIMÁTICAS Y ELEVACIÓN
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
south_america <- geodata::gadm("COL", level = 0, path = "env")

library(geodata)
library(terra)
library(sf)

cat("✂️ Recortando capas a Sudamérica...\n")

# --- 1. Definir países de Sudamérica ---
paises_SA <- c("COL", "VEN")

# --- 2. Descargar límites administrativos y unirlos ---
paises_list <- lapply(paises_SA, function(p) gadm(country = p, level = 0, path = "env"))
south_america <- do.call(rbind, paises_list) |> st_as_sf()


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




library(rJava)
options(java.parameters = "-Xmx16g")


rjava -version

# Sustituye la ruta si tu JDK tiene otro número
Sys.setenv(JAVA_HOME = "C:/Program Files/Java/jdk-25")
library(rJava)

library(rJava)
.jinit()
.jcall("java/lang/System", "S", "getProperty", "java.version")








library(dismo)

# Cargar capas
env_files <- list.files("env_southamerica", pattern = "\\.tif$", full.names = TRUE)
env_stack <- rast(env_files)

# Leer puntos
occ <- read.csv("data/Sternoclyta_cyanopectus_GBIF.csv")

# Crear modelo MaxEnt (asegúrate de tener java instalado)
mx <- maxent(x = env_stack, p = occ[, c("decimalLongitude", "decimalLatitude")])

# Convertir a RasterStack
library(raster)
env_stack_raster <- raster::stack(env_stack)

# Correr el modelo Maxent
library(dismo)

# Asumiendo que tus coordenadas están en 'occ'
mx <- dismo::maxent(x = env_stack_raster,
                    p = occ[, c("decimalLongitude", "decimalLatitude")])

# Corregir nombres
names(env_stack) <- gsub("~", "_30s_", names(env_stack))

# Verifica que ahora se vean bien
names(env_stack)
# [1] "wc2.1_30s_bio_1" "wc2.1_30s_bio_2" ... "wc2.1_30s_bio_19" "wc2.1_30s_elev"

# Luego predice

pred <- terra::predict(env_stack, mx, na.rm = TRUE, progress = TRUE, filename="maxent_pred_sa.tif", overwrite=TRUE)



# Ver el mapa
plot(pred)
points(occ[, c("decimalLongitude", "decimalLatitude")], pch = 20, col = "red")

filename = "maxent_pred_sa.tif"


list.files(pattern = "maxent_pred_sa.tif")

class(pred)

library(terra)
summary(pred)

library(terra)

# Guarda explícitamente en GeoTIFF
writeRaster(pred, "maxent_pred_sa.tif", overwrite = TRUE)


names(mx)
colnames(mx$betas)

library(terra)

# Ver CRS, extensión y min/max reales
crs(pred)
ext(pred)
minmax(pred)        # devuelve min y max (excluyendo NA)
terra::global(pred, fun = "mean", na.rm = TRUE)  # ejemplo estadístico




library(terra)

# Reescalar entre 0 y 1 para exportar a Qgis
pred_rescaled <- (pred - 0) / (0.9986 - 0)
writeRaster(pred_rescaled, "maxent_pred_rescaled.tif", overwrite = TRUE)


#Guardar los puntos 


# Extraer solo las columnas de coordenadas
occ_points <- occ[, c("decimalLongitude", "decimalLatitude")]

# Guardar en CSV
write.csv(occ_points, "coordenadas_Anadia.csv", row.names = FALSE)

cat("✅ Archivo guardado como 'coordenadas_Sternoclyta.csv'\n")

