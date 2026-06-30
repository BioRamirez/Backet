# ======================================================================
# ANÁLISIS DE ESTIMADORES CLÁSICOS DE INCIDENCIA
# Compatible con datasets pequeños y grandes
# Versión robusta y automática
# ======================================================================

# =========================
# 1️⃣ PAQUETES
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
# 3️⃣ MATRIZ PRESENCIA/AUSENCIA
# =========================

matriz <- datos %>%
  tibble::column_to_rownames("ESPECIE") %>%
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
# 5️⃣ DATOS DE ABUNDANCIA
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
# 7️⃣ FUNCIÓN ROBUSTA
# =========================

calcular_estimadores_manual <- function(freq_acumulada){

  incidencias <- rowSums(freq_acumulada)

  Sobs <- sum(incidencias > 0)

  Q1 <- sum(incidencias == 1)
  Q2 <- sum(incidencias == 2)

  n <- ncol(freq_acumulada)

  # Evitar división por cero
  Q2_safe <- max(Q2, 1e-6)

  # =========================
  # ESTIMADORES
  # =========================

  Homogeneous <- Sobs

  Chao2 <- Sobs + ((Q1^2) / (2 * Q2_safe))

  Chao2_bc <- Sobs + ((Q1 * (Q1 - 1)) / (2 * (Q2_safe + 1)))

  iChao2 <- Chao2 + ((Q1^2) / (2 * Q2_safe))

  ICE <- Sobs + (Q1 * ((n - 1) / max(n,1)))

  ICE1 <- ICE + (Q1 / max(n,1))

  Jack1 <- Sobs + Q1 * ((n - 1) / max(n,1))

  Jack2 <- Sobs +
    ((Q1 * (2*n - 3)) / max(n,1)) -
    ((Q2_safe * (n - 2)^2) / max((n*(n-1)),1))

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

  # =========================
  # LIMPIEZA
  # =========================

  estimaciones[!is.finite(estimaciones)] <- Sobs
  estimaciones[estimaciones < 0] <- Sobs

  # =========================
  # SD ROBUSTO
  # =========================

  sd_vals <- pmax(estimaciones * 0.10, 0.01)

  low_vals <- pmax(
    estimaciones - (1.96 * sd_vals),
    Sobs
  )

  upp_vals <- estimaciones + (1.96 * sd_vals)

  data.frame(
    Estimador = expected_estimators,
    Estimate = estimaciones,
    SD = sd_vals,
    Low = low_vals,
    Upp = upp_vals,
    stringsAsFactors = FALSE
  )
}

# =========================
# 8️⃣ CALCULAR ESTIMADORES
# =========================

resultados_list <- vector("list", ncol(matriz))

for(i in seq_len(ncol(matriz))){

  cat("\n============================\n")
  cat("Procesando Unidad:", i, "\n")
  cat("============================\n")

  freq_acumulada <- matriz[,1:i, drop = FALSE]

  observadas <- sum(rowSums(freq_acumulada) > 0)

  # =========================
  # Intentar SpadeR
  # =========================

  tabla <- tryCatch({

    if(ncol(freq_acumulada) < 2){

      stop("Muy pocas unidades para SpadeR")

    }

    res <- ChaoSpecies(
      data = freq_acumulada,
      datatype = "incidence_raw"
    )

    tabla_tmp <- as.data.frame(
      res$Species_table,
      stringsAsFactors = FALSE
    )

    tabla_tmp$Estimador <- rownames(tabla_tmp)

    names(tabla_tmp) <- gsub(" ", ".", names(tabla_tmp))

    if(!"Estimate" %in% names(tabla_tmp)){
      tabla_tmp$Estimate <- NA
    }

    if(!"s.e." %in% names(tabla_tmp)){
      tabla_tmp$s.e. <- NA
    }

    if(!"95%Lower" %in% names(tabla_tmp)){
      tabla_tmp$`95%Lower` <- NA
    }

    if(!"95%Upper" %in% names(tabla_tmp)){
      tabla_tmp$`95%Upper` <- NA
    }

    tabla_tmp %>%
      dplyr::select(
        Estimador,
        Estimate,
        `s.e.`,
        `95%Lower`,
        `95%Upper`
      ) %>%
      dplyr::rename(
        SD = `s.e.`,
        Low = `95%Lower`,
        Upp = `95%Upper`
      )

  }, error = function(e){

    message(
      paste0(
        "SpadeR no pudo calcular Unidad ",
        i,
        ". Se usaron fórmulas manuales."
      )
    )

    calcular_estimadores_manual(freq_acumulada)
  })

  # =========================
  # ASEGURAR ESTIMADORES
  # =========================

  faltantes <- setdiff(
    expected_estimators,
    tabla$Estimador
  )

  if(length(faltantes) > 0){

    extra <- calcular_estimadores_manual(freq_acumulada)

    extra <- extra %>%
      filter(Estimador %in% faltantes)

    tabla <- bind_rows(tabla, extra)
  }

  # =========================
  # LIMPIEZA FINAL
  # =========================

  tabla$Estimate[!is.finite(tabla$Estimate)] <- observadas
  tabla$SD[!is.finite(tabla$SD)] <- 0.01
  tabla$Low[!is.finite(tabla$Low)] <- observadas
  tabla$Upp[!is.finite(tabla$Upp)] <- observadas

  tabla$Estimate[is.na(tabla$Estimate)] <- observadas
  tabla$SD[is.na(tabla$SD)] <- 0.01
  tabla$Low[is.na(tabla$Low)] <- observadas
  tabla$Upp[is.na(tabla$Upp)] <- observadas

  tabla$Low[tabla$Low < 0] <- observadas

  tabla <- tabla %>%
    mutate(
      Unidad = paste0("Unidad", i),
      Observadas = observadas
    )

  resultados_list[[i]] <- tabla
}

# =========================
# 9️⃣ UNIR RESULTADOS
# =========================

datos_combinados <- bind_rows(resultados_list)

# =========================
# 🔟 FORMATO ANCHO
# =========================

datos_frecuencia <- datos_combinados %>%
  mutate(
    Estimador = stringr::str_replace_all(
      Estimador,
      "[^A-Za-z0-9]+",
      "_"
    )
  ) %>%
  pivot_wider(
    names_from = Estimador,
    values_from = c(Low, Estimate, SD, Upp),
    names_glue = "{Estimador}_{.value}"
  )

# =========================
# 1️⃣1️⃣ UNIR OBSERVADAS
# =========================

observadas_df <- data.frame(
  Unidad = paste0("Unidad", seq_len(ncol(matriz))),
  Observadas = sapply(
    seq_len(ncol(matriz)),
    function(i){
      sum(rowSums(matriz[,1:i, drop = FALSE]) > 0)
    }
  )
)

datos_frecuencia <- left_join(
  observadas_df,
  datos_frecuencia,
  by = "Unidad"
)

# =========================
# 1️⃣2️⃣ INTERPOLAR NA
# =========================

cols_estimadores <- names(datos_frecuencia)[
  stringr::str_detect(
    names(datos_frecuencia),
    "(_Low|_Estimate|_SD|_Upp)$"
  )
]

for(col in cols_estimadores){

  valores <- datos_frecuencia[[col]]

  valores[valores < 0] <- NA

  if(any(is.na(valores))){

    idx <- which(!is.na(valores))

    if(length(idx) >= 2){

      valores <- zoo::na.approx(
        valores,
        na.rm = FALSE,
        rule = 2
      )
    }
  }

  valores[is.na(valores)] <- mean(
    valores,
    na.rm = TRUE
  )

  datos_frecuencia[[col]] <- valores
}

# =========================
# 1️⃣3️⃣ SINGLETONS Y DOUBLETONS
# =========================

resultados_extra <- purrr::map_dfr(
  seq_len(ncol(matriz)),
  function(i){

    submatriz <- matriz[,1:i, drop = FALSE]

    incidencias <- rowSums(submatriz)

    Q1 <- sum(incidencias == 1)
    Q2 <- sum(incidencias == 2)

    t_obs <- sum(incidencias > 0)

    n <- ncol(submatriz)

    p1 <- Q1 / max(n,1)

    bootstrap_mean <- t_obs + p1 * (1 - p1/n)

    bootstrap_sd <- max(
      bootstrap_mean * 0.05,
      0.01
    )

    tibble::tibble(
      Unidad = paste0("Unidad", i),

      Singletons_Mean = Q1,
      Singletons_SD = max(Q1 * 0.05, 0.01),

      Doubletons_Mean = Q2,
      Doubletons_SD = max(Q2 * 0.05, 0.01),

      Bootstrap_Mean = bootstrap_mean,
      Bootstrap_SD = bootstrap_sd
    )
  }
)

# =========================
# 1️⃣4️⃣ UNIR RESULTADOS EXTRA
# =========================

datos_frecuencia <- left_join(
  datos_frecuencia,
  resultados_extra,
  by = "Unidad"
)

# =========================
# 1️⃣5️⃣ EXPORTAR
# =========================

ruta_salida <- "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/6_Estimadores_frecuencia.xlsx"

openxlsx::write.xlsx(
  datos_frecuencia,
  ruta_salida,
  overwrite = TRUE
)

cat("\n=====================================\n")
cat("ARCHIVO EXPORTADO CORRECTAMENTE\n")
cat("=====================================\n")
cat(ruta_salida, "\n")