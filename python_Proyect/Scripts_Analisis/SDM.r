# ============================================================
# SDM AUTOMATIZADO (GBIF -> MAXENT con ENMeval)
# Solo cambia el nombre de la especie en la sección 1
# ============================================================

# =========================
# 1) CONFIGURACIÓN
# =========================
species_name <- "Erythrolamprus zweifeli"  # <-- CAMBIAR AQUÍ
output_dir <- "SDM_output"
dir.create(output_dir, showWarnings = FALSE)

# =========================
# 2) PAQUETES
# =========================
pkgs <- c("spocc","dplyr","terra","sf","maxnet","ENMeval","spThin","usdm","geodata","readr")
install.packages(setdiff(pkgs, installed.packages()[,"Package"]), dependencies = TRUE)
lapply(pkgs, library, character.only = TRUE)

set.seed(42)

# =========================
# 3) DESCARGA Y LIMPIEZA GBIF
# =========================
cat("Descargando datos de GBIF...\n")
df <- occ(query = species_name, from = "gbif")
sp1 <- df$gbif$data[[1]]

# Filtrar coordenadas válidas
esp_geo <- subset(sp1,
                  !is.na(name) & !is.na(longitude) & !is.na(latitude) &
                  longitude >= -180 & longitude <= 180 &
                  latitude >= -90 & latitude <= 90)

# Data limpio
occ_clean <- data.frame(
  species = species_name,
  longitude = as.numeric(esp_geo$longitude),
  latitude = as.numeric(esp_geo$latitude)
) %>%
  distinct(longitude, latitude, .keep_all = TRUE)

# Guardar CSV
occ_file <- file.path(output_dir, "occurrences.csv")
write.csv(occ_clean, occ_file, row.names = FALSE)
cat("Registros limpios:", nrow(occ_clean), "\n")

# =========================
# 4) VARIABLES AMBIENTALES
# =========================
cat("Descargando variables ambientales...\n")

bio <- geodata::worldclim_global(var = "bio", res = 0.5, path = output_dir)
elev <- geodata::elevation_global(res = 0.5, path = output_dir)

# Recorte a Sudamérica (extensión manual)
ext <- ext(-90, -30, -60, 15)

bio_crop <- crop(bio, ext)
bio_crop <- mask(bio_crop, bio_crop[[1]])

elev_crop <- crop(elev, ext)
elev_crop <- mask(elev_crop, elev_crop[[1]])

env_stack <- c(bio_crop, elev_crop)

# =========================
# 5) PREPARACIÓN ESPACIAL
# =========================
occ_sf <- st_as_sf(occ_clean, coords = c("longitude","latitude"), crs = 4326)

# Eliminar puntos con NA en variables
vals <- terra::extract(env_stack, vect(occ_sf))
occ_sf <- occ_sf[complete.cases(vals), ]

cat("Registros tras limpieza ambiental:", nrow(occ_sf), "\n")

# =========================
# 6) THINNING
# =========================
coords <- st_coordinates(occ_sf)
thin_df <- data.frame(Longitude = coords[,1], Latitude = coords[,2])

thin_res <- spThin::thin(loc.data = thin_df,
                        lat.col = "Latitude", long.col = "Longitude",
                        thin.par = 1,  # ~1 km
                        reps = 1,
                        locs.thinned.list.return = TRUE)

coords_thin <- thin_res[[1]]
occ_thin <- st_as_sf(coords_thin, coords = c("Longitude","Latitude"), crs = 4326)

cat("Registros tras thinning:", nrow(occ_thin), "\n")

# =========================
# 7) BACKGROUND
# =========================
bg <- spatSample(env_stack, size = 10000, method = "random", na.rm = TRUE, xy = TRUE)
bg_coords <- as.matrix(bg[,1:2])

# =========================
# 8) SELECCIÓN DE VARIABLES (VIF)
# =========================
env_vals <- as.data.frame(values(env_stack))
env_vals <- env_vals[complete.cases(env_vals), ]

vif_res <- usdm::vifstep(env_vals, th = 10)
vars <- vif_res@results$Variables

env_sel <- env_stack[[vars]]
cat("Variables seleccionadas:", vars, "\n")

# =========================
# 9) PREPARAR DATOS
# =========================
occ_xy <- st_coordinates(occ_thin)
occ_xy <- as.matrix(occ_xy[,1:2])

# =========================
# 10) ENMEVAL (TUNING)
# =========================
cat("Ejecutando ENMeval...\n")

tune <- ENMevaluate(
  occ = occ_xy,
  env = env_sel,
  bg.coords = bg_coords,
  method = "block",
  RMvalues = seq(0.5, 3, 0.5),
  fc = c("L","LQ","H","LQH"),
  algorithm = "maxnet",
  parallel = FALSE
)

write.csv(tune@results, file.path(output_dir, "ENMeval_results.csv"), row.names = FALSE)

# =========================
# 11) MEJOR MODELO
# =========================
res <- tune@results
best <- res[which.min(res$AICc), ]

best_rm <- best$RM
best_fc <- as.character(best$fc)

cat("Mejor modelo:", best_fc, "RM=", best_rm, "\n")

# =========================
# 12) MODELO FINAL
# =========================
pres_vals <- terra::extract(env_sel, vect(occ_thin))
bg_vals <- terra::extract(env_sel, bg_coords)

pres_df <- as.data.frame(pres_vals); pres_df$occ <- 1
bg_df <- as.data.frame(bg_vals); bg_df$occ <- 0

train <- rbind(pres_df, bg_df)
train <- train[complete.cases(train), ]

formula <- maxnet.formula(train$occ, train[,vars], classes = best_fc)
model <- maxnet(train$occ, train[,vars], f = formula, regmult = best_rm)

# =========================
# 13) PREDICCIÓN
# =========================
pred_vals <- predict(model, as.data.frame(values(env_sel)), type = "cloglog")

pred_r <- env_sel[[1]]
pred_r[] <- pred_vals

pred_file <- file.path(output_dir, "prediction.tif")
writeRaster(pred_r, pred_file, overwrite = TRUE)

# =========================
# 14) GUARDAR RESULTADOS
# =========================
saveRDS(model, file.path(output_dir, "model.rds"))

cat("\n✅ SDM COMPLETADO\n")
cat("Archivos en:", output_dir, "\n")
















































#----------------------------------------------------------------------------

# ============================================================
# SDM AUTOMATIZADO (GBIF -> MAXENT con ENMeval)
# Solo cambia el nombre de la especie en la sección 1
# ============================================================

# =========================
# 1) CONFIGURACIÓN
# =========================
species_name <- "Erythrolamprus zweifeli"  # <-- CAMBIAR AQUÍ
output_dir <- "SDM_output"
dir.create(output_dir, showWarnings = FALSE)

# =========================
# 2) PAQUETES
# =========================
pkgs <- c("spocc","dplyr","terra","sf","maxnet","ENMeval","spThin","usdm","geodata","readr")
install.packages(setdiff(pkgs, installed.packages()[,"Package"]), dependencies = TRUE)
lapply(pkgs, library, character.only = TRUE)

set.seed(42)

# =========================
# 3) DESCARGA Y LIMPIEZA GBIF
# =========================
cat("Descargando datos de GBIF...\n")
df <- occ(query = species_name, from = "gbif")
sp1 <- df$gbif$data[[1]]

# Filtrar coordenadas válidas
esp_geo <- subset(sp1,
                  !is.na(name) & !is.na(longitude) & !is.na(latitude) &
                  longitude >= -180 & longitude <= 180 &
                  latitude >= -90 & latitude <= 90)

# Data limpio
occ_clean <- data.frame(
  species = species_name,
  longitude = as.numeric(esp_geo$longitude),
  latitude = as.numeric(esp_geo$latitude)
) %>%
  distinct(longitude, latitude, .keep_all = TRUE)

# Guardar CSV
occ_file <- file.path(output_dir, "occurrences.csv")
write.csv(occ_clean, occ_file, row.names = FALSE)
cat("Registros limpios:", nrow(occ_clean), "\n")

# =========================
# 4) VARIABLES AMBIENTALES
# =========================
cat("Descargando variables ambientales...\n")

bio <- geodata::worldclim_global(var = "bio", res = 0.5, path = output_dir)
elev <- geodata::elevation_global(res = 0.5, path = output_dir)

# Recorte a Sudamérica (extensión manual)
ext <- ext(-90, -30, -60, 15)

bio_crop <- crop(bio, ext)
bio_crop <- mask(bio_crop, bio_crop[[1]])

elev_crop <- crop(elev, ext)
elev_crop <- mask(elev_crop, elev_crop[[1]])

env_stack <- c(bio_crop, elev_crop)



#---------------------------------
# =========================
# 4) VARIABLES AMBIENTALES (DESDE ARCHIVOS EXISTENTES)
# =========================
cat("Cargando variables ambientales desde archivos locales...\n")

# Ruta donde ya se descargaron (la misma que usaste antes)
env_path <- output_dir

# Buscar archivos BIO
bio_files <- list.files(env_path, pattern = "wc2.1_30s_bio_.*\\.tif$", full.names = TRUE)

# Verificar que existan
if(length(bio_files) == 0){
  stop("No se encontraron archivos bioclimáticos en la carpeta. Verifica la ruta.")
}

# Cargar variables BIO
bio <- rast(bio_files)

# Cargar elevación (si ya la descargaste)
elev_file <- list.files(env_path, pattern = "elev.*\\.tif$", full.names = TRUE)

if(length(elev_file) > 0){
  elev <- rast(elev_file)
} else {
  cat("⚠️ No se encontró elevación, continuando solo con variables BIO\n")
  elev <- NULL
}

# =========================
# RECORTE A ÁREA DE ESTUDIO (Sudamérica)
# =========================
ext_sa <- ext(-90, -30, -60, 15)

bio_crop <- crop(bio, ext_sa)
bio_crop <- mask(bio_crop, bio_crop[[1]])

if(!is.null(elev)){
  elev_crop <- crop(elev, ext_sa)
  elev_crop <- mask(elev_crop, elev_crop[[1]])
  env_stack <- c(bio_crop, elev_crop)
} else {
  env_stack <- bio_crop
}

cat("Capas ambientales listas:", nlyr(env_stack), "variables\n")





# =========================
# 5) PREPARACIÓN ESPACIAL
# =========================
occ_sf <- st_as_sf(occ_clean, coords = c("longitude","latitude"), crs = 4326)

# Eliminar puntos con NA en variables
vals <- terra::extract(env_stack, vect(occ_sf))
occ_sf <- occ_sf[complete.cases(vals), ]

cat("Registros tras limpieza ambiental:", nrow(occ_sf), "\n")

# =========================
# 6) THINNING
# =========================
coords <- st_coordinates(occ_sf)
thin_df <- data.frame(Longitude = coords[,1], Latitude = coords[,2])

thin_res <- spThin::thin(loc.data = thin_df,
                        lat.col = "Latitude", long.col = "Longitude",
                        thin.par = 1,  # ~1 km
                        reps = 1,
                        locs.thinned.list.return = TRUE)

coords_thin <- thin_res[[1]]
occ_thin <- st_as_sf(coords_thin, coords = c("Longitude","Latitude"), crs = 4326)

cat("Registros tras thinning:", nrow(occ_thin), "\n")

# =========================
# 7) BACKGROUND
# =========================
bg <- spatSample(env_stack, size = 10000, method = "random", na.rm = TRUE, xy = TRUE)
bg_coords <- as.matrix(bg[,1:2])

# =========================
# 8) SELECCIÓN DE VARIABLES (VIF)
# =========================
env_vals <- as.data.frame(values(env_stack))
env_vals <- env_vals[complete.cases(env_vals), ]

vif_res <- usdm::vifstep(env_vals, th = 10)
vars <- vif_res@results$Variables

env_sel <- env_stack[[vars]]
cat("Variables seleccionadas:", vars, "\n")

# =========================
# 9) PREPARAR DATOS
# =========================
occ_xy <- st_coordinates(occ_thin)
occ_xy <- as.matrix(occ_xy[,1:2])

# =========================
# 10) ENMEVAL (TUNING)
# =========================
cat("Ejecutando ENMeval...\n")

tune <- ENMevaluate(
  occ = occ_xy,
  env = env_sel,
  bg.coords = bg_coords,
  method = "block",
  RMvalues = seq(0.5, 3, 0.5),
  fc = c("L","LQ","H","LQH"),
  algorithm = "maxnet",
  parallel = FALSE
)

write.csv(tune@results, file.path(output_dir, "ENMeval_results.csv"), row.names = FALSE)

# =========================
# 11) MEJOR MODELO
# =========================
res <- tune@results
best <- res[which.min(res$AICc), ]

best_rm <- best$RM
best_fc <- as.character(best$fc)

cat("Mejor modelo:", best_fc, "RM=", best_rm, "\n")

# =========================
# 12) MODELO FINAL
# =========================
pres_vals <- terra::extract(env_sel, vect(occ_thin))
bg_vals <- terra::extract(env_sel, bg_coords)

pres_df <- as.data.frame(pres_vals); pres_df$occ <- 1
bg_df <- as.data.frame(bg_vals); bg_df$occ <- 0

train <- rbind(pres_df, bg_df)
train <- train[complete.cases(train), ]

formula <- maxnet.formula(train$occ, train[,vars], classes = best_fc)
model <- maxnet(train$occ, train[,vars], f = formula, regmult = best_rm)

# =========================
# 13) PREDICCIÓN
# =========================
pred_vals <- predict(model, as.data.frame(values(env_sel)), type = "cloglog")

pred_r <- env_sel[[1]]
pred_r[] <- pred_vals

pred_file <- file.path(output_dir, "prediction.tif")
writeRaster(pred_r, pred_file, overwrite = TRUE)

# =========================
# 14) MAPA LISTO PARA INFORME
# =========================
cat("Generando mapa para informe...
")

# Convertir raster a data.frame para ggplot
pred_df <- as.data.frame(pred_r, xy = TRUE, na.rm = TRUE)
names(pred_df)[3] <- "Suitability"

# Convertir puntos de ocurrencia
occ_plot <- as.data.frame(st_coordinates(occ_thin))
names(occ_plot) <- c("X","Y")

# Crear mapa bonito
mapa <- ggplot() +
  geom_raster(data = pred_df, aes(x = x, y = y, fill = Suitability)) +
  scale_fill_viridis_c(name = "Idoneidad", option = "C") +
  geom_point(data = occ_plot, aes(x = X, y = Y), color = "red", size = 1, alpha = 0.7) +
  coord_equal() +
  labs(
    title = paste("Distribución potencial de", species_name),
    subtitle = "Modelo MaxEnt (cloglog)",
    x = "Longitud",
    y = "Latitud"
  ) +
  theme_minimal(base_size = 12)

# Guardar imagen
map_file <- file.path(output_dir, "mapa_distribucion.png")
ggsave(map_file, mapa, width = 10, height = 8, dpi = 300)

# =========================
# 15) GUARDAR RESULTADOS
# =========================
saveRDS(model, file.path(output_dir, "model.rds"))

cat("
✅ SDM COMPLETADO
")
cat("Archivos en:", output_dir, "
")
cat("Mapa generado:", map_file, "
")
cat("Archivos en:", output_dir, "\n")

