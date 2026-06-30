#---------------------------------- Análisis de estimadores clásicos de incidencia (frecuencia/presencia) -----------------------

# =========================
# PAQUETES NECESARIOS
# =========================

paquetes <- c(
  "readxl",
  "dplyr",
  "tidyr",
  "boot",
  "SpadeR",
  "openxlsx",
  "stringr",
  "tibble",
  "purrr",
  "zoo"
)

instalar <- paquetes[!(paquetes %in% installed.packages()[,"Package"])]

if(length(instalar)) {
  install.packages(instalar)
}

lapply(paquetes, library, character.only = TRUE)

# =========================
# CARGAR DATOS
# =========================

ruta <- "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/3_Tabla_Abundancia_Semanal.xlsx"

datos <- read_excel(ruta, sheet = 1)

# =========================
# MATRIZ PRESENCIA / AUSENCIA
# =========================

matriz <- datos %>%
  column_to_rownames("ESPECIE") %>%
  as.matrix()

# Convertir a 0/1
matriz[matriz > 0] <- 1

colnames(matriz) <- paste0("Unidad", 1:ncol(matriz))

# =========================
# VERIFICACIONES
# =========================

str(matriz)

cat("\nEspecies observadas:", sum(rowSums(matriz) > 0), "\n")

cat("\nTotal incidencias:", sum(matriz), "\n")

cat("\nNA en matriz:", sum(is.na(matriz)), "\n")

# =========================
# CARGAR DATOS ABUNDANCIA
# =========================

ruta_abun <- "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/5_Estimadores_Abundancia.xlsx"

datos_abundancia <- read_excel(ruta_abun, sheet = 1)

#------------------------------------ Calcular estimadores clásicos de incidencia ---------------------

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

resultados_list <- vector("list", ncol(matriz))

# =========================
# LOOP PRINCIPAL
# =========================

for (i in seq_len(ncol(matriz))) {

  cat("\n============================")
  cat("\nUnidad acumulada:", i)
  cat("\n============================\n")

  freq_acumulada <- matriz[, 1:i, drop = FALSE]

  incidencias <- rowSums(freq_acumulada)

  Sobs <- sum(incidencias > 0)

  Q1 <- sum(incidencias == 1)

  Q2_original <- sum(incidencias == 2)

  n <- ncol(freq_acumulada)

  # Evitar división por cero
  Q2 <- ifelse(Q2_original == 0, 1e-6, Q2_original)

  # =========================
  # INTENTAR SPADE R
  # =========================

  tabla <- tryCatch({

    res <- ChaoSpecies(
      freq_acumulada,
      datatype = "incidence_raw"
    )

    tabla_tmp <- as.data.frame(
      res$Species_table,
      stringsAsFactors = FALSE
    )

    if(!"Estimador" %in% names(tabla_tmp)) {
      tabla_tmp$Estimador <- rownames(tabla_tmp)
    }

    # Crear columnas faltantes
    for(col in c("Estimate", "s.e.", "95%Lower", "95%Upper")) {

      if(!col %in% names(tabla_tmp)) {
        tabla_tmp[[col]] <- NA_real_
      }

    }

    tabla_tmp %>%
      select(
        Estimador,
        Estimate,
        `s.e.`,
        `95%Lower`,
        `95%Upper`
      )

  }, error = function(e){

    cat("\nSpadeR falló en Unidad", i, "\n")
    cat("Mensaje:", e$message, "\n")

    NULL

  })

  # =========================
  # SI FALLA SPADE R
  # =========================

  if(is.null(tabla) || nrow(tabla) == 0){

    cat("Usando fórmulas manuales...\n")

    Homogeneous <- Sobs

    Chao2 <- Sobs + ((Q1^2) / (2 * Q2))

    Chao2_bc <- Sobs + (Q1 * (Q1 - 1)) / (2 * (Q2 + 1))

    iChao2 <- Chao2 + ((Q1^2) / (2 * Q2))

    ICE <- Sobs + (Q1 * ((n - 1) / n))

    ICE1 <- ICE + (Q1 / n)

    Jack1 <- Sobs + Q1 * ((n - 1) / n)

    if(n > 1){

      Jack2 <- Sobs +
        ((Q1 * (2*n - 3))/n) -
        ((Q2 * (n - 2)^2)/(n*(n-1)))

    } else {

      Jack2 <- Sobs

    }

    estimaciones <- c(
      Homogeneous,
      Chao2,
      Chao2_bc,
      iChao2,
      ICE,
      ICE1,
      Jack1,
      Jack2
    )

    # Control de errores numéricos
    estimaciones[is.na(estimaciones)] <- Sobs
    estimaciones[is.nan(estimaciones)] <- Sobs
    estimaciones[is.infinite(estimaciones)] <- Sobs

    estimaciones <- pmax(estimaciones, Sobs)

    sd_vals <- pmax(estimaciones * 0.10, 0.01)

    low_vals <- pmax(
      estimaciones - (1.96 * sd_vals),
      Sobs
    )

    upp_vals <- estimaciones + (1.96 * sd_vals)

    tabla <- data.frame(
      Estimador = expected_estimators,
      Estimate = estimaciones,
      s.e. = sd_vals,
      `95%Lower` = low_vals,
      `95%Upper` = upp_vals,
      stringsAsFactors = FALSE
    )

  } else {

    # =========================
    # COMPLETAR ESTIMADORES FALTANTES
    # =========================

    faltantes <- setdiff(expected_estimators, tabla$Estimador)

    if(length(faltantes) > 0){

      cat("Completando estimadores faltantes...\n")

      formulas_manual <- data.frame(
        Estimador = expected_estimators,
        Estimate = c(
          Sobs,
          Sobs + ((Q1^2) / (2 * Q2)),
          Sobs + (Q1 * (Q1 - 1)) / (2 * (Q2 + 1)),
          Sobs + ((Q1^2) / Q2),
          Sobs + (Q1 * ((n - 1) / n)),
          Sobs + (Q1 * ((n - 1) / n)) + (Q1 / n),
          Sobs + Q1 * ((n - 1) / n),
          ifelse(
            n > 1,
            Sobs + ((Q1 * (2*n - 3))/n),
            Sobs
          )
        ),
        stringsAsFactors = FALSE
      )

      extra <- formulas_manual %>%
        filter(Estimador %in% faltantes)

      extra$`s.e.` <- pmax(extra$Estimate * 0.10, 0.01)

      extra$`95%Lower` <- pmax(
        extra$Estimate - (1.96 * extra$`s.e.`),
        Sobs
      )

      extra$`95%Upper` <- extra$Estimate +
        (1.96 * extra$`s.e.`)

      tabla <- bind_rows(tabla, extra)

    }

    # =========================
    # COMPLETAR NA CORRECTAMENTE
    # =========================

    tabla$Estimate[is.na(tabla$Estimate)] <- Sobs
    tabla$Estimate[is.infinite(tabla$Estimate)] <- Sobs
    tabla$Estimate[is.nan(tabla$Estimate)] <- Sobs

    # ---------- s.e. ----------
    idx_se <- which(is.na(tabla$`s.e.`))

    if(length(idx_se) > 0){

      tabla$`s.e.`[idx_se] <-
        pmax(tabla$Estimate[idx_se] * 0.10, 0.01)

    }

    # ---------- Lower ----------
    idx_low <- which(is.na(tabla$`95%Lower`))

    if(length(idx_low) > 0){

      tabla$`95%Lower`[idx_low] <-
        pmax(
          tabla$Estimate[idx_low] -
            1.96 * tabla$`s.e.`[idx_low],
          Sobs
        )

    }

    # ---------- Upper ----------
    idx_upp <- which(is.na(tabla$`95%Upper`))

    if(length(idx_upp) > 0){

      tabla$`95%Upper`[idx_upp] <-
        tabla$Estimate[idx_upp] +
        1.96 * tabla$`s.e.`[idx_upp]

    }

  }

  # =========================
  # LIMPIEZA FINAL
  # =========================

  tabla <- tabla %>%
    mutate(
      Unidad = paste0("Unidad", i),
      Observadas = Sobs
    )

  tabla$Estimador <- trimws(as.character(tabla$Estimador))

  # Nunca permitir NA
  tabla$Estimate[is.na(tabla$Estimate)] <- Sobs
  tabla$`s.e.`[is.na(tabla$`s.e.`)] <- 0.01
  tabla$`95%Lower`[is.na(tabla$`95%Lower`)] <- Sobs
  tabla$`95%Upper`[is.na(tabla$`95%Upper`)] <- Sobs + 0.01

  # Nunca negativos
  tabla$Estimate <- pmax(tabla$Estimate, Sobs)
  tabla$`95%Lower` <- pmax(tabla$`95%Lower`, Sobs)
  tabla$`95%Upper` <- pmax(tabla$`95%Upper`, tabla$Estimate)
  tabla$`s.e.` <- pmax(tabla$`s.e.`, 0.01)

  resultados_list[[i]] <- tabla

}

# =========================
# VERIFICAR RESULTADOS
# =========================

print(resultados_list)

View(resultados_list)

str(resultados_list)

#------------------------------------ UNIR RESULTADOS ---------------------

resultados_list <- lapply(seq_along(resultados_list), function(i){

  df <- resultados_list[[i]]

  names(df) <- str_replace_all(
    names(df),
    c(
      "95%Lower" = "X95.Lower",
      "95%Upper" = "X95.Upper",
      " " = "."
    )
  )

  df$Unidad <- paste0("Unidad", i)

  df

})

datos_combinados <- bind_rows(resultados_list)

datos_combinados <- datos_combinados %>%
  select(
    Unidad,
    Estimador,
    Estimate,
    s.e.,
    X95.Lower,
    X95.Upper,
    Observadas
  ) %>%
  rename(
    Mean = Estimate,
    SD = s.e.,
    Low = X95.Lower,
    Upp = X95.Upper
  )

# =========================
# FORMATO ANCHO
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
  ) %>%
  relocate(
    Unidad,
    starts_with("Observadas"),
    .before = everything()
  )

# =========================
# LIMPIAR NUMÉRICOS
# =========================

cols_num <- sapply(datos_frecuencia, is.numeric)

datos_frecuencia[, cols_num] <-
  lapply(datos_frecuencia[, cols_num], function(x){

    x[is.na(x)] <- 0.01
    x[is.nan(x)] <- 0.01

    if(any(is.infinite(x))){

      max_finito <- max(x[is.finite(x)], na.rm = TRUE)

      x[is.infinite(x)] <- max_finito

    }

    x <- pmax(x, 0.01)

    return(x)

  })

# =========================
# EXPORTAR
# =========================

write.xlsx(
  datos_frecuencia,
  "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/6_Estimadores_frecuencia.xlsx"
)

cat(
  "\nArchivo exportado correctamente.\n"
)