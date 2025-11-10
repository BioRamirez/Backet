#---------------------------------- Análisis de estimadores clásicos de incidencia (frecuencia/presencia) -----------------------

# --- Paquetes necesarios ---
paquetes <- c("readxl", "dplyr", "tidyr", "boot", "SpadeR", "openxlsx", "stringr")
instalar <- paquetes[!(paquetes %in% installed.packages()[,"Package"])]
if(length(instalar)) install.packages(instalar)
lapply(paquetes, library, character.only = TRUE)

# --- Cargar datos ---
ruta <- "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/Tabla_Abundancia_Semanal.xlsx"
datos <- read_excel(ruta, sheet = 1)

# --- Convertir a matriz de presencia/ausencia ---
matriz <- datos %>%
  column_to_rownames("ESPECIE") %>%
  as.matrix()

# Forzar 0/1 por si hay conteos
matriz[matriz > 0] <- 1
colnames(matriz) <- paste0("Unidad", 1:ncol(matriz))

# --- Verificar estructura ---
str(matriz)

#------------------------------------ Calcular estimadores clásicos de incidencia ---------------------



library(dplyr)
library(tidyr)
library(stringr)
library(SpadeR)

# --- Estimadores esperados (los que devuelve ChaoSpecies) ---
expected_estimators <- c(
  "Homogeneous Model", "Chao2 (Chao, 1987)", "Chao2-bc",
  "iChao2 (Chiu et al. 2014)", "ICE (Lee & Chao, 1994)",
  "ICE-1 (Lee & Chao, 1994)", "1st order jackknife", "2nd order jackknife"
)

resultados_list <- vector("list", ncol(matriz))

for (i in seq_len(ncol(matriz))) {
  cat("\n--- Unidad acumulada:", i, "---\n")
  freq_acumulada <- matriz[, 1:i, drop = FALSE]
  
  if (ncol(freq_acumulada) < 2) {
    # Datos insuficientes → crear tabla con NA pero incluir número de especies observadas
    tabla <- data.frame(
      Estimador = expected_estimators,
      Estimate = NA_real_,
      s.e. = NA_real_,
      `95%Lower` = NA_real_,
      `95%Upper` = NA_real_,
      Unidad = paste0("Unidad", i),
      Observadas = sum(rowSums(freq_acumulada) > 0),
      stringsAsFactors = FALSE
    )
  } else {
    # Ejecutar ChaoSpecies en formato de incidencia cruda
    df_result <- tryCatch({
      res <- ChaoSpecies(freq_acumulada, datatype = "incidence_raw")
      tabla_tmp <- as.data.frame(res$Species_table, stringsAsFactors = FALSE)
      if (!"Estimador" %in% names(tabla_tmp)) tabla_tmp$Estimador <- rownames(tabla_tmp)
      
      # Asegurar columnas estándar
      for (col in c("Estimate", "s.e.", "95%Lower", "95%Upper")) {
        if (!col %in% names(tabla_tmp)) tabla_tmp[[col]] <- NA_real_
      }
      tabla <- tabla_tmp %>%
        select(Estimador, Estimate, `s.e.`, `95%Lower`, `95%Upper`) %>%
        mutate(
          Unidad = paste0("Unidad", i),
          Observadas = sum(rowSums(freq_acumulada) > 0)
        )
      tabla
    }, error = function(e) {
      warning("ChaoSpecies falló en Unidad ", i, " → se rellenan NA: ", e$message)
      data.frame(
        Estimador = expected_estimators,
        Estimate = NA_real_,
        s.e. = NA_real_,
        `95%Lower` = NA_real_,
        `95%Upper` = NA_real_,
        Unidad = paste0("Unidad", i),
        Observadas = sum(rowSums(freq_acumulada) > 0),
        stringsAsFactors = FALSE
      )
    })
  }
  
  tabla$Estimador <- trimws(as.character(tabla$Estimador))
  
  # 🔁 Reemplazar NA en Estimate por Observadas solo donde falta
  tabla <- tabla %>%
    mutate(Estimate = ifelse(is.na(Estimate), Observadas, Estimate))
  
  resultados_list[[i]] <- tabla
}

print(resultados_list)





#----------------FALTA UNIR TODAS LAS SEMANAS EN UNA SOLA TABLA--------------------
























































































# Número total de unidades de muestreo
n_unidades <- ncol(matriz)

# Calcular cuántas veces aparece cada especie (en cuántas semanas)
frecuencia_especies <- rowSums(matriz > 0)

# Crear el vector en formato "incidence_freq"
incidencia_freq <- c(n_unidades, frecuencia_especies)

library(SpadeR)

spade_incid <- SpadeR::ChaoSpecies(incidencia_freq, datatype = "incidence_freq")

spade_incid


