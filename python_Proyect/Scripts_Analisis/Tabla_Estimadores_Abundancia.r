#------------------------------------Análisis de estimadores clásicos de abundancia y bootstrap manual---------------------

#--------------------------------Cargar paquetes y documentos---------------------------------------
paquetes <- c("readxl", "vegan", "dplyr")

# Instala los que falten
instalar <- paquetes[!(paquetes %in% installed.packages()[,"Package"])]
if(length(instalar)) install.packages(instalar)

# Cargar paquetes
lapply(paquetes, library, character.only = TRUE)

ruta <- "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/Tabla_Abundancia_Semanal.xlsx"
datos <- read_excel(ruta, sheet = 1)

# 🔹 Paquete necesario
library(SpadeR)
library(tidyverse)

#------------------------------------Crear matriz de abundancia y calcular estimadores clásicos de abundancia---------------------
# 🔹 Convertir tu dataframe 'datos' en una matriz de abundancia
# Primera columna = ESPECIE
matriz <- datos %>%
  column_to_rownames("ESPECIE") %>%
  as.matrix()

str(matriz)
colnames(matriz) <- paste0("Unidad", 1:ncol(matriz))
colnames(matriz)

str(matriz)

library(SpadeR)

# Vector de abundancias totales
abund_total <- rowSums(matriz)

str(abund_total)

# Calcular estimadores clásicos de riqueza (tipo abundance)
spade_result <- ChaoSpecies(abund_total, datatype = "abundance")

# Mostrar el resumen
spade_result

#------------------------------------Calcular estimamdores acomulados por semana---------------

library(dplyr)
library(tidyr)
library(openxlsx)
library(SpadeR)

resultados_list <- list()

for (i in 1:ncol(matriz)) {
  cat("\n--- Unidad acumulada:", i, "---\n")
  
  abund_acumulada <- rowSums(matriz[, 1:i, drop = FALSE])
  resultado <- ChaoSpecies(abund_acumulada, datatype = "abundance")
 
tabla <- as.data.frame(resultado$Species_table)
tabla$Estimador <- rownames(resultado$Species_table)
tabla$Unidad <- paste0("Unidad", i)

  
  resultados_list[[i]] <- tabla
}

# --- Unir todas las semanas ---
resultados_totales <- bind_rows(resultados_list)

# --- Limpiar nombres ---
resultados_totales <- resultados_totales %>%
  rename(
    Mean = Estimate,
    SD = s.e.,
    Low = `95%Lower`,
    Upp = `95%Upper`
  ) %>%
  select(Unidad, Estimador, Mean, SD, Low, Upp)

print(resultados_totales)

library(dplyr)
library(tidyr)
library(stringr)

# --- 1. Limpiar nombres de los estimadores ---
resultados_limpios <- resultados_totales %>%
  mutate(
    Unidad = trimws(Unidad),
    Estimador = trimws(Estimador),
    # Reemplazar espacios, paréntesis, comas, etc. por "_"
    Estimador = str_replace_all(Estimador, "[^A-Za-z0-9]+", "_"),
    Estimador = str_replace_all(Estimador, "_+", "_"),
    Estimador = str_remove_all(Estimador, "^_|_$")
  )

# --- 2. Pivotar para agrupar Low, Mean y Upp por estimador ---
resultados_wide <- resultados_limpios %>%
  pivot_wider(
    names_from = Estimador,
    values_from = c(Low, Mean, SD, Upp),
    names_glue = "{Estimador}_{.value}"
  )

# --- 3. Reordenar columnas para que queden Low, Mean, Upp juntos ---
orden_columnas <- resultados_wide %>%
  select(Unidad, sort(tidyselect::peek_vars())) %>%
  names()

# Reordenamos columnas para que cada estimador tenga Low, Mean, Upp juntos
orden_ordenado <- c("Unidad")
for (est in unique(resultados_limpios$Estimador)) {
  orden_ordenado <- c(
    orden_ordenado,
    paste0(est, c("_Low", "_Mean", "_SD", "_Upp"))
  )
}
# Filtrar solo las columnas que existen
orden_ordenado <- intersect(orden_ordenado, names(resultados_wide))
resultados_wide <- resultados_wide[, orden_ordenado]

# --- Calcular Observadas, Singletons y Doubletons por semana ---


library(boot)
library(dplyr)

# --- Crear lista vacía ---
resumen_list <- list()

# --- Definir una función auxiliar para bootstrap ---
calc_stats <- function(abund, R = 1000) {
  # Bootstrap simple
  boot_res <- boot(data = abund, statistic = function(x, i) {
    xi <- x[i]
    c(
      Observadas = sum(xi > 0),
      Singletons = sum(xi == 1),
      Doubletons = sum(xi == 2)
    )
  }, R = R)
  
  # Calcular media, sd y quantiles
  resumen <- apply(boot_res$t, 2, function(x) {
    c(
      Mean = mean(x, na.rm = TRUE),
      SD = sd(x, na.rm = TRUE),
      Low = quantile(x, 0.025, na.rm = TRUE),
      Upp = quantile(x, 0.975, na.rm = TRUE)
    )
  })
  
  # Convertir a data.frame con nombres
  resumen_df <- as.data.frame(t(resumen))
  resumen_df$Metrica <- rownames(resumen_df)
  return(resumen_df)
}

# --- Bucle para acumular semanas ---
for (i in 1:ncol(matriz)) {
  abund_acumulada <- rowSums(matriz[, 1:i, drop = FALSE])
  
  # Calcular estadísticas bootstrap
  boot_df <- calc_stats(abund_acumulada, R = 1000)
  boot_df$Unidad <- paste0("Unidad", i)
  
  resumen_list[[i]] <- boot_df
}


# --- Unir todos los resultados del resumen ---
resumen_total <- bind_rows(resumen_list) %>%
  # Renombrar correctamente las columnas antes de seleccionar
  rename(
    Low = `Low.2.5%`,
    Upp = `Upp.97.5%`
  ) %>%
  select(Unidad, Metrica, Mean, SD, Low, Upp)

# --- Revisar resultado final ---
print(resumen_total)

library(dplyr)
library(tidyr)

# --- Paso 1: reemplazar Metrica numérica por nombres claros ---
resumen_total <- resumen_total %>%
  mutate(
    Metrica = case_when(
      Metrica == 1 ~ "Observadas",
      Metrica == 2 ~ "Singletons",
      Metrica == 3 ~ "Doubletons"
    )
  )

# --- Paso 2: pivotar para dejar una fila por unidad ---
resumen_pivot <- resumen_total %>%
  pivot_wider(
    names_from = Metrica,
    values_from = c(Mean, SD, Low, Upp),
    names_glue = "{Metrica}_{.value}"
  )

# --- Ver resultado final ---
print(resumen_pivot)

# --- Reordenar columnas por bloque lógico ---
resumen_pivot <- resumen_pivot %>%
  select(
    Unidad,
    Observadas_Low, Observadas_Mean, Observadas_SD, Observadas_Upp,
    Singletons_Low, Singletons_Mean, Singletons_SD, Singletons_Upp,
    Doubletons_Low, Doubletons_Mean, Doubletons_SD, Doubletons_Upp
  )

# --- Verificar ---
print(resumen_pivot)

# --- Integrar resumen con resultados_wide ---

resultados_Final <- left_join(resumen_pivot, resultados_wide, by = "Unidad")





#----------------------------------Agregar Bootstraping-----------------------------------------------------

library(boot)
library(dplyr)
library(SpadeR)

# --- Función para calcular el estimador Chao1 (corrige el nombre automáticamente) ---
calc_chao1 <- function(data, indices) {
  muestra <- data[indices]
  resultado <- SpadeR::ChaoSpecies(muestra, datatype = "abundance")
  
  # Buscar el nombre que contenga "Chao1"
  fila_chao <- grep("Chao1", rownames(resultado$Species_table), value = TRUE)
  
  if (length(fila_chao) == 0) return(NA)
  
  valor <- resultado$Species_table[fila_chao[1], "Estimate"]
  return(valor)
}

# --- Lista para guardar resultados ---
bootstrap_list <- list()

# --- Bucle por unidad ---
for (i in 1:ncol(matriz)) {
  cat("\n--- Bootstrap para Unidad", i, "---\n")
  
  abund_acumulada <- rowSums(matriz[, 1:i, drop = FALSE])
  
  # Bootstrap con 1000 repeticiones
  boot_res <- boot(data = abund_acumulada, statistic = calc_chao1, R = 100)
  
  # Media, desviación y percentiles
  media <- mean(boot_res$t, na.rm = TRUE)
  sd <- sd(boot_res$t, na.rm = TRUE)
  low <- quantile(boot_res$t, 0.025, na.rm = TRUE)
  upp <- quantile(boot_res$t, 0.975, na.rm = TRUE)
  
  # Guardar resultados
  bootstrap_list[[i]] <- data.frame(
    Unidad = paste0("Unidad", i),
    Bootstrap_Mean = media,
    Bootstrap_SD = sd,
    Bootstrap_Low = low,
    Bootstrap_Upp = upp
  )
}

# --- Unir todos los resultados ---
bootstrap_total <- bind_rows(bootstrap_list)

print(bootstrap_total)

# --- Integrar con resultados_Final -------------------------------------------------------
resultados_Final1 <- left_join(resultados_Final, bootstrap_total, by = "Unidad")

# --- 4. Verificar estructura final ---
glimpse(resultados_Final1)


library(openxlsx)
write.xlsx(resultados_Final1, "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/Estimadores_SpadeR_Semanal_Wide.xlsx")
cat("\n✅ Archivo exportado correctamente con los estimadores agrupados por semana.\n")









































# Número total de unidades de muestreo
n_unidades <- ncol(matriz)

# Calcular cuántas veces aparece cada especie (en cuántas semanas)
frecuencia_especies <- rowSums(matriz > 0)

# Crear el vector en formato "incidence_freq"
incidencia_freq <- c(n_unidades, frecuencia_especies)

library(SpadeR)

spade_incid <- SpadeR::ChaoSpecies(incidencia_freq, datatype = "incidence_freq")

spade_incid


































































# matriz: tu tabla de abundancias (especies en filas, semanas en columnas)
incidencia <- ifelse(matriz > 0, 1, 0)

# Número de unidades de muestreo
T <- ncol(incidencia)

# Número de especies observadas
S_obs <- nrow(incidencia)

# Proporción de ocurrencia de cada especie
p_i <- rowSums(incidencia) / T

# Estimador Bootstrap clásico (Smith & van Belle, 1984)
S_boot <- S_obs + sum((1 - p_i)^T)

# Error estándar aproximado (Chao & Shen, 2003)
var_boot <- sum(((1 - p_i)^T) * (1 - (1 - p_i)^T))
se_boot <- sqrt(var_boot)

# Intervalos de confianza (95%)
lower_boot <- S_boot - 1.96 * se_boot
upper_boot <- S_boot + 1.96 * se_boot

# Crear un dataframe con el resultado
bootstrap_result <- data.frame(
  Estimador = "Bootstrap",
  Valor = S_boot,
  Error = se_boot,
  IC_Inf = lower_boot,
  IC_Sup = upper_boot
)

bootstrap_result




# --- 2. Crear matriz de incidencia ---
incidencia <- ifelse(matriz > 0, 1, 0)
T <- ncol(incidencia) # número de unidades de muestreo
S_obs <- nrow(incidencia)

# --- 3. Calcular Q1 (singleton) y Q2 (doubleton) ---
freq_incid <- rowSums(incidencia)
Q1 <- sum(freq_incid == 1)
Q2 <- sum(freq_incid == 2)

cat("Singletons (Q1):", Q1, "\n")
cat("Doubletons (Q2):", Q2, "\n")

# --- 4. Calcular estimadores de incidencia con SpadeR ---
spade_incid <- SpadeR::ChaoSpecies(incidencia, datatype = "incidence_freq")

# Extraer tabla de resultados
tabla_est <- as.data.frame(spade_incid$Species_table)
colnames(tabla_est) <- c("Estimador", "Estimado", "Error", "IC_Inf", "IC_Sup")

# --- 5. Calcular Bootstrap manualmente ---
p_i <- rowSums(incidencia) / T
S_boot <- S_obs + sum((1 - p_i)^T)
var_boot <- sum(((1 - p_i)^T) * (1 - (1 - p_i)^T))
se_boot <- sqrt(var_boot)
lower_boot <- S_boot - 1.96 * se_boot
upper_boot <- S_boot + 1.96 * se_boot

bootstrap_result <- data.frame(
  Estimador = "Bootstrap",
  Estimado = S_boot,
  Error = se_boot,
  IC_Inf = lower_boot,
  IC_Sup = upper_boot
)

# --- 6. Integrar Bootstrap con el resto ---
tabla_completa <- bind_rows(tabla_est, bootstrap_result)

# --- 7. Añadir Q1 y Q2 al encabezado como atributos ---
info_basica <- data.frame(
  Variable = c("Número de especies observadas (S_obs)",
               "Número de unidades (T)",
               "Singletons (Q1)",
               "Doubletons (Q2)"),
  Valor = c(S_obs, T, Q1, Q2)
)

# --- 8. Mostrar resultados ---
cat("\n(1) INFORMACIÓN BÁSICA:\n")
print(info_basica)
cat("\n(2) TABLA DE ESTIMADORES CLÁSICA:\n")
print(tabla_completa)



# ---------------------------------------------------------
# 🔹 Cálculo de Bootstrap, singletons y doubletons (datos de abundancia)
# ---------------------------------------------------------

# 1️⃣ Calcular abundancia total por especie
abundancia_por_especie <- rowSums(datos)

# 2️⃣ Calcular singletons (especies con 1 individuo)
singletons <- sum(abundancia_por_especie == 1)

# 3️⃣ Calcular doubletons (especies con 2 individuos)
doubletons <- sum(abundancia_por_especie == 2)

# 4️⃣ Calcular estimador Bootstrap (Chao et al. 2014)
n <- sum(abundancia_por_especie)  # total de individuos
pi_hat <- abundancia_por_especie / n
bootstrap_estimate <- length(pi_hat) + sum((1 - pi_hat)^n)

# 5️⃣ Mostrar resultados en consola
cat("\n==================== RESULTADOS DE RARE SPECIES ====================\n")
cat("Número de especies observadas (Sobs):", length(pi_hat), "\n")
cat("Número de singletons:", singletons, "\n")
cat("Número de doubletons:", doubletons, "\n")
cat("Estimador Bootstrap (riqueza esperada):", round(bootstrap_estimate, 2), "\n")
cat("=====================================================================\n")

