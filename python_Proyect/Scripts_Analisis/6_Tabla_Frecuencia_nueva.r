# ======================================================================
# ANÁLISIS DE ESTIMADORES CLÁSICOS DE INCIDENCIA (FRECUENCIA/PRESENCIA)
# Script corregido y optimizado
# ======================================================================

# =========================
# 1️⃣ PAQUETES
# =========================
paquetes <- c(
  "readxl", "dplyr", "tidyr", "boot",
  "SpadeR", "openxlsx", "stringr",
  "tibble", "purrr", "zoo"
)

instalar <- paquetes[!(paquetes %in% installed.packages()[, "Package"])]

if(length(instalar) > 0){
  install.packages(instalar)
}

invisible(lapply(paquetes, library, character.only = TRUE))

# =========================
# 2️⃣ CARGAR DATOS
# =========================

ruta_frecuencia <- "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/3_Tabla_Abundancia_Semanal.xlsx"

datos <- read_excel(ruta_frecuencia, sheet = 1)

# =========================
# 3️⃣ CONVERTIR A MATRIZ
# =========================

matriz <- datos %>%
  column_to_rownames("ESPECIE") %>%
  as.matrix()

# Convertir a presencia/ausencia
matriz[matriz > 0] <- 1

# Renombrar unidades
colnames(matriz) <- paste0("Unidad", seq_len(ncol(matriz)))

# =========================
# 4️⃣ VERIFICACIONES
# =========================

cat("\n============================\n")
cat("VERIFICACIÓN DE MATRIZ\n")
cat("============================\n")

cat("Número de especies:", nrow(matriz), "\n")
cat("Número de unidades:", ncol(matriz), "\n")
cat("Especies observadas:", sum(rowSums(matriz) > 0), "\n")
cat("Total incidencias:", sum(matriz), "\n")
cat("Valores NA:", sum(is.na(matriz)), "\n")

# =========================
# 5️⃣ CARGAR DATOS DE ABUNDANCIA
# =========================

ruta_abundancia <- "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/5_Estimadores_Abundancia.xlsx"

datos_abundancia <- read_excel(ruta_abundancia, sheet = 1)

# =========================
# 6️⃣ ESTIMADORES ESPERADOS
# =========================

expected_estimators <- c(
  "Homogeneous Model",
  "Chao2 (Chao, 1987)",
  "Chao2-bc",
  "iChao2 (Chiu et al. 2014)",
  "ICE (Lee & Chao, 1994)",
  "ICE-1 (Lee & Chao, 1994)",
  "1st order jackknife",
  "2nd order jackknife"
)

# =========================
# 7️⃣ CALCULAR ESTIMADORES
# =========================

resultados_list <- vector("list", ncol(matriz))

for(i in seq_len(ncol(matriz))){

  cat("\nProcesando Unidad:", i, "\n")

  freq_acumulada <- matriz[, 1:i, drop = FALSE]

  observadas <- sum(rowSums(freq_acumulada) > 0)

  # ==========================================================
  # Si hay menos de 2 unidades no se pueden calcular estimadores
  # ==========================================================

  if(ncol(freq_acumulada) < 2){

    tabla <- data.frame(
      Estimador = expected_estimators,
      Estimate = NA_real_,
      SE = NA_real_,
      Low = NA_real_,
      Upp = NA_real_,
      Unidad = paste0("Unidad", i),
      Observadas = observadas,
      stringsAsFactors = FALSE
    )

  } else {

    tabla <- tryCatch({

      res <- ChaoSpecies(
        data = freq_acumulada,
        datatype = "incidence_raw"
      )

      tabla_tmp <- as.data.frame(
        res$Species_table,
        stringsAsFactors = FALSE
      )

      tabla_tmp$Estimador <- rownames(tabla_tmp)

      # Normalizar nombres
      names(tabla_tmp) <- gsub(" ", ".", names(tabla_tmp))

      # Crear columnas faltantes
      if(!"Estimate" %in% names(tabla_tmp)){
        tabla_tmp$Estimate <- NA_real_
      }

      if(!"s.e." %in% names(tabla_tmp)){
        tabla_tmp$s.e. <- NA_real_
      }

      if(!"95%Lower" %in% names(tabla_tmp)){
        tabla_tmp$`95%Lower` <- NA_real_
      }

      if(!"95%Upper" %in% names(tabla_tmp)){
        tabla_tmp$`95%Upper` <- NA_real_
      }

      tabla_tmp %>%
        select(
          Estimador,
          Estimate,
          `s.e.`,
          `95%Lower`,
          `95%Upper`
        ) %>%
        rename(
          SE = `s.e.`,
          Low = `95%Lower`,
          Upp = `95%Upper`
        ) %>%
        mutate(
          Unidad = paste0("Unidad", i),
          Observadas = observadas
        )

    }, error = function(e){

      warning(
        paste0(
          "Error en Unidad ",
          i,
          ": ",
          e$message
        )
      )

      data.frame(
        Estimador = expected_estimators,
        Estimate = NA_real_,
        SE = NA_real_,
        Low = NA_real_,
        Upp = NA_real_,
        Unidad = paste0("Unidad", i),
        Observadas = observadas,
        stringsAsFactors = FALSE
      )
    })
  }

  tabla$Estimador <- trimws(as.character(tabla$Estimador))

  resultados_list[[i]] <- tabla
}

# =========================
# 8️⃣ UNIR RESULTADOS
# =========================

datos_combinados <- bind_rows(resultados_list)

# =========================
# 9️⃣ RENOMBRAR COLUMNAS
# =========================

datos_combinados <- datos_combinados %>%
  rename(
    Mean = Estimate,
    SD = SE
  )

# =========================
# 🔟 PASAR A FORMATO ANCHO
# =========================

datos_frecuencia <- datos_combinados %>%
  mutate(
    Estimador = str_replace_all(
      Estimador,
      "[^A-Za-z0-9]+",
      "_"
    )
  ) %>%
  pivot_wider(
    names_from = Estimador,
    values_from = c(Low, Mean, SD, Upp),
    names_glue = "{Estimador}_{.value}"
  )

# =========================
# 1️⃣1️⃣ AGREGAR OBSERVADAS
# =========================

observadas_df <- data.frame(
  Unidad = paste0("Unidad", seq_len(ncol(matriz))),
  Observadas = sapply(
    seq_len(ncol(matriz)),
    function(i){
      sum(rowSums(matriz[, 1:i, drop = FALSE]) > 0)
    }
  )
)

# =========================
# AGREGAR OBSERVADAS
# =========================

observadas_df <- data.frame(
  Unidad = paste0("Unidad", seq_len(ncol(matriz))),
  Observadas = sapply(
    seq_len(ncol(matriz)),
    function(i){
      sum(rowSums(matriz[, 1:i, drop = FALSE]) > 0)
    }
  ),
  stringsAsFactors = FALSE
)

# Eliminar columnas Observadas previas si existen
datos_frecuencia <- datos_frecuencia %>%
  select(-matches("^Observadas$|^Observadas\\.x$|^Observadas\\.y$"))

# Unir nuevamente
datos_frecuencia <- left_join(
  observadas_df,
  datos_frecuencia,
  by = "Unidad"
)

# =========================
# 1️⃣2️⃣ CALCULAR SD FALTANTES
# =========================

cols_low <- names(datos_frecuencia)[str_detect(names(datos_frecuencia), "_Low$")]

for(low_col in cols_low){

  base <- str_remove(low_col, "_Low$")

  upp_col <- paste0(base, "_Upp")
  sd_col  <- paste0(base, "_SD")

  if(all(c(low_col, upp_col) %in% names(datos_frecuencia))){

    datos_frecuencia[[sd_col]] <-
      (datos_frecuencia[[upp_col]] -
         datos_frecuencia[[low_col]]) / (2 * 1.96)
  }
}

# =========================
# 1️⃣3️⃣ UNIR OBSERVADAS DE ABUNDANCIA
# =========================

datos_frecuencia <- datos_frecuencia %>%
  left_join(
    datos_abundancia %>%
      select(
        Unidad,
        Observadas_Low,
        Observadas_Mean,
        Observadas_SD,
        Observadas_Upp
      ),
    by = "Unidad"
  )

# =========================
# 1️⃣4️⃣ REEMPLAZAR NA EN UNIDAD1
# =========================

cols_estimadores <- names(datos_frecuencia)[
  str_detect(names(datos_frecuencia), "(_Low|_Mean|_SD|_Upp)$")
]

for(col in cols_estimadores){

  fila1 <- which(datos_frecuencia$Unidad == "Unidad1")

  if(is.na(datos_frecuencia[[col]][fila1])){

    if(str_detect(col, "_Low$")){

      datos_frecuencia[[col]][fila1] <-
        datos_frecuencia$Observadas_Low[fila1]

    } else if(str_detect(col, "_Mean$")){

      datos_frecuencia[[col]][fila1] <-
        datos_frecuencia$Observadas_Mean[fila1]

    } else if(str_detect(col, "_SD$")){

      datos_frecuencia[[col]][fila1] <-
        datos_frecuencia$Observadas_SD[fila1]

    } else if(str_detect(col, "_Upp$")){

      datos_frecuencia[[col]][fila1] <-
        datos_frecuencia$Observadas_Upp[fila1]
    }
  }
}

# =========================
# 1️⃣5️⃣ INTERPOLAR NA
# =========================

for(col in cols_estimadores){

  valores <- datos_frecuencia[[col]]

  if(any(is.na(valores))){

    idx <- which(!is.na(valores))

    if(length(idx) >= 2){

      datos_frecuencia[[col]] <- approx(
        x = idx,
        y = valores[idx],
        xout = seq_along(valores),
        method = "linear",
        rule = 2
      )$y
    }
  }
}

# =========================
# 1️⃣6️⃣ SINGLETONS Y DOUBLETONS
# =========================

resultados_extra <- map_dfr(
  seq_len(ncol(matriz)),
  function(i){

    submatriz <- matriz[, 1:i, drop = FALSE]

    incidencias <- rowSums(submatriz)

    Q1 <- sum(incidencias == 1)
    Q2 <- sum(incidencias == 2)

    t_obs <- sum(incidencias > 0)

    n <- ncol(submatriz)

    p1 <- Q1 / n

    bootstrap_mean <- t_obs + p1 * (1 - p1 / n)

    bootstrap_sd <- bootstrap_mean * 0.05

    bootstrap_low <- bootstrap_mean - 1.96 * bootstrap_sd

    bootstrap_upp <- bootstrap_mean + 1.96 * bootstrap_sd

    sing_sd <- Q1 * 0.05
    sing_low <- Q1 - 1.96 * sing_sd
    sing_upp <- Q1 + 1.96 * sing_sd

    doub_sd <- Q2 * 0.05
    doub_low <- Q2 - 1.96 * doub_sd
    doub_upp <- Q2 + 1.96 * doub_sd

    tibble(
      Unidad = paste0("Unidad", i),

      Singletons_Low = sing_low,
      Singletons_Mean = Q1,
      Singletons_SD = sing_sd,
      Singletons_Upp = sing_upp,

      Doubletons_Low = doub_low,
      Doubletons_Mean = Q2,
      Doubletons_SD = doub_sd,
      Doubletons_Upp = doub_upp,

      Bootstrap_Low = bootstrap_low,
      Bootstrap_Mean = bootstrap_mean,
      Bootstrap_SD = bootstrap_sd,
      Bootstrap_Upp = bootstrap_upp
    )
  }
)

# =========================
# 1️⃣7️⃣ UNIR RESULTADOS EXTRA
# =========================

datos_frecuencia <- left_join(
  datos_frecuencia,
  resultados_extra,
  by = "Unidad"
)

# =========================
# 1️⃣8️⃣ CORREGIR VALORES NEGATIVOS
# =========================

cols_estimadores <- names(datos_frecuencia)[
  str_detect(names(datos_frecuencia), "(_Low|_Mean|_SD|_Upp)$")
]

for(col in cols_estimadores){

  valores <- datos_frecuencia[[col]]

  valores[valores < 0] <- NA

  valores <- zoo::na.approx(
    valores,
    na.rm = FALSE,
    rule = 2
  )

  datos_frecuencia[[col]] <- valores
}

# =========================
# 1️⃣9️⃣ REORDENAR COLUMNAS
# =========================

datos_frecuencia <- datos_frecuencia %>%
  select(
    Unidad,
    Observadas,
    starts_with("Observadas_"),
    starts_with("Singletons_"),
    starts_with("Doubletons_"),
    starts_with("Bootstrap_"),
    everything()
  )

# =========================
# 2️⃣0️⃣ EXPORTAR RESULTADOS
# =========================

ruta_salida <- "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/6_Estimadores_frecuencia.xlsx"

write.xlsx(
  datos_frecuencia,
  ruta_salida,
  overwrite = TRUE
)

cat("\n=====================================\n")
cat("ARCHIVO EXPORTADO CORRECTAMENTE\n")
cat("=====================================\n")
cat(ruta_salida, "\n")
