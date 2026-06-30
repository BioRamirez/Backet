# ------------------------------------
# Análisis de estimadores clásicos de abundancia y bootstrap (versión robusta)
# ------------------------------------

# --------------------------
# Paquetes
# --------------------------
paquetes <- c("readxl", "vegan", "dplyr", "tidyr", "stringr", "SpadeR",
              "boot", "openxlsx", "tibble")

instalar <- paquetes[!(paquetes %in% installed.packages()[, "Package"])]
if (length(instalar)) install.packages(instalar, dependencies = TRUE)

invisible(lapply(paquetes, library, character.only = TRUE))

# --------------------------
# Leer datos
# --------------------------
ruta <- "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/3_Tabla_Abundancia_Semanal.xlsx"
datos <- readxl::read_excel(ruta, sheet = 1)

# --------------------------
# Preparar matriz de abundancia (primera columna = ESPECIE)
# --------------------------
if (!"ESPECIE" %in% names(datos)) stop("La columna 'ESPECIE' no existe en el archivo de entrada.")

matriz <- datos %>%
  tibble::column_to_rownames("ESPECIE") %>%
  as.matrix()

# Renombrar columnas a Unidad1, Unidad2, ...
colnames(matriz) <- paste0("Unidad", seq_len(ncol(matriz)))

# --------------------------
# Funciones auxiliares robustas
# --------------------------

# Wrapper seguro para ChaoSpecies: devuelve la tabla completa o NA si falla
safe_chao_species <- function(x) {
  # x: vector de abundancias por especie
  # Validación mínima: al menos un individuo y al menos una especie con >0
  if (length(x) == 0 || sum(x, na.rm = TRUE) <= 0 || length(which(x > 0)) < 1) {
    return(list(error = "Insufficient data", result = NA))
  }

  res <- tryCatch({
    out <- SpadeR::ChaoSpecies(x, datatype = "abundance")
    list(error = NULL, result = out)
  }, error = function(e) {
    list(error = conditionMessage(e), result = NA)
  }, warning = function(w) {
    # capturar warnings como no fatales pero retornar resultado si existe
    invokeRestart("muffleWarning")
    # volver a intentar pero permitiendo warning mudo (o devolver NA si falla)
    tryCatch({
      out <- SpadeR::ChaoSpecies(x, datatype = "abundance")
      list(error = NULL, result = out)
    }, error = function(e) list(error = conditionMessage(e), result = NA))
  })

  return(res)
}

# Función para calcular Observadas, Singletons, Doubletons de forma segura con bootstrap
calc_stats_safe <- function(abund, R = 100) {
  # abund: vector de abundancias por especie (no por individuo)
  # Si no hay datos suficientes, devolver NAs pero manteniendo formato
  if (length(abund) == 0 || sum(abund, na.rm = TRUE) <= 0) {
    df_na <- data.frame(
      Metrica = c("Observadas", "Singletons", "Doubletons"),
      Mean = NA_real_, SD = NA_real_, Low = NA_real_, Upp = NA_real_,
      stringsAsFactors = FALSE
    )
    return(df_na)
  }

  # Definir función statistic para boot
  stat_fun <- function(x, i) {
    xi <- x[i]
    c(
      Observadas = sum(xi > 0, na.rm = TRUE),
      Singletons = sum(xi == 1, na.rm = TRUE),
      Doubletons = sum(xi == 2, na.rm = TRUE)
    )
  }

  boot_res <- tryCatch({
    boot::boot(data = abund, statistic = stat_fun, R = R)
  }, error = function(e) {
    return(NULL)
  })

  if (is.null(boot_res) || is.null(boot_res$t)) {
    df_na <- data.frame(
      Metrica = c("Observadas", "Singletons", "Doubletons"),
      Mean = NA_real_, SD = NA_real_, Low = NA_real_, Upp = NA_real_,
      stringsAsFactors = FALSE
    )
    return(df_na)
  }

  resumen <- apply(boot_res$t, 2, function(col) {
    c(Mean = mean(col, na.rm = TRUE),
      SD = sd(col, na.rm = TRUE),
      Low = as.numeric(quantile(col, 0.025, na.rm = TRUE)),
      Upp = as.numeric(quantile(col, 0.975, na.rm = TRUE)))
  })

  resumen_df <- as.data.frame(t(resumen))
  resumen_df$Metrica <- c("Observadas", "Singletons", "Doubletons")
  rownames(resumen_df) <- NULL
  resumen_df <- resumen_df[, c("Metrica", "Mean", "SD", "Low", "Upp")]
  return(resumen_df)
}

# Función segura para obtener Chao1 mediante bootstrap (si se desea)
calc_chao1_safe <- function(abund, R = 100) {
  # Si no hay suficiente info, devolver NA set
  if (length(abund) == 0 || sum(abund, na.rm = TRUE) <= 1 || length(which(abund > 0)) < 2) {
    return(data.frame(
      Bootstrap_Mean = NA_real_,
      Bootstrap_SD = NA_real_,
      Bootstrap_Low = NA_real_,
      Bootstrap_Upp = NA_real_,
      stringsAsFactors = FALSE
    ))
  }

  # Statistic wrapper para boot que llama a safe_chao_species
  stat_chao <- function(data, indices) {
    muestra <- data[indices]
    sc <- safe_chao_species(muestra)
    if (!is.null(sc$error) || is.na(sc$result)[1]) {
      return(NA_real_)
    }
    # intentar extraer Chao1 (buscar fila con "Chao1" sin sensibilidad a mayus)
    st <- sc$result$Species_table
    fila <- grep("Chao1", rownames(st), ignore.case = TRUE, value = TRUE)
    if (length(fila) == 0) return(NA_real_)
    val <- st[fila[1], "Estimate"]
    # Asegurar valor numérico
    as.numeric(val)
  }

  boot_res <- tryCatch({
    boot::boot(data = abund, statistic = stat_chao, R = R)
  }, error = function(e) {
    return(NULL)
  })

  if (is.null(boot_res) || is.null(boot_res$t)) {
    return(data.frame(
      Bootstrap_Mean = NA_real_,
      Bootstrap_SD = NA_real_,
      Bootstrap_Low = NA_real_,
      Bootstrap_Upp = NA_real_,
      stringsAsFactors = FALSE
    ))
  }

  valores <- as.numeric(boot_res$t[, 1])
  df_out <- data.frame(
    Bootstrap_Mean = mean(valores, na.rm = TRUE),
    Bootstrap_SD = sd(valores, na.rm = TRUE),
    Bootstrap_Low = as.numeric(quantile(valores, 0.025, na.rm = TRUE)),
    Bootstrap_Upp = as.numeric(quantile(valores, 0.975, na.rm = TRUE)),
    stringsAsFactors = FALSE
  )
  return(df_out)
}

# --------------------------
# Estimadores clásicos acumulados por unidad (semana)
# --------------------------
resultados_list <- list()
log_insuficientes <- list()

for (i in seq_len(ncol(matriz))) {
  cat("\n--- Unidad acumulada:", i, "---\n")
  abund_acumulada <- rowSums(matriz[, 1:i, drop = FALSE])

  sc <- safe_chao_species(abund_acumulada)
  if (!is.null(sc$error)) {
    # Registrar el error y crear una tabla con NA para mantener la estructura
    message(sprintf("Unidad %d: ChaoSpecies fallo -> %s", i, sc$error))
    log_insuficientes[[paste0("Unidad", i)]] <- sc$error
    # Generar tabla NA compatible
    tabla <- data.frame(Estimate = NA_real_, s.e. = NA_real_, `95%Lower` = NA_real_, `95%Upper` = NA_real_,
                        row.names = "Chao1", stringsAsFactors = FALSE)
  } else {
    tabla <- as.data.frame(sc$result$Species_table)
  }

  # Preparar tabla con nombres
  tabla_df <- as.data.frame(tabla)
  tabla_df$Estimador <- rownames(tabla_df)
  tabla_df$Unidad <- paste0("Unidad", i)
  rownames(tabla_df) <- NULL

  resultados_list[[i]] <- tabla_df
}

resultados_totales <- dplyr::bind_rows(resultados_list)

# Normalizar nombres de columnas si faltan
if (!"Estimate" %in% names(resultados_totales)) resultados_totales$Estimate <- NA_real_
if (!"s.e." %in% names(resultados_totales)) resultados_totales$`s.e.` <- NA_real_
if (!"95%Lower" %in% names(resultados_totales)) resultados_totales$`95%Lower` <- NA_real_
if (!"95%Upper" %in% names(resultados_totales)) resultados_totales$`95%Upper` <- NA_real_

resultados_totales <- resultados_totales %>%
  rename(
    Mean = Estimate,
    SD = `s.e.`,
    Low = `95%Lower`,
    Upp = `95%Upper`
  ) %>%
  select(Unidad, Estimador, Mean, SD, Low, Upp)

# Limpiar nombres de estimadores
resultados_limpios <- resultados_totales %>%
  mutate(
    Unidad = trimws(Unidad),
    Estimador = trimws(Estimador),
    Estimador = stringr::str_replace_all(Estimador, "[^A-Za-z0-9]+", "_"),
    Estimador = stringr::str_replace_all(Estimador, "_+", "_"),
    Estimador = stringr::str_remove_all(Estimador, "^_|_$")
  )

# Pivotar wide
resultados_wide <- resultados_limpios %>%
  tidyr::pivot_wider(
    names_from = Estimador,
    values_from = c(Low, Mean, SD, Upp),
    names_glue = "{Estimador}_{.value}"
  )

# Asegurar orden de columnas
orden_ordenado <- c("Unidad")
for (est in unique(resultados_limpios$Estimador)) {
  orden_ordenado <- c(orden_ordenado, paste0(est, c("_Low", "_Mean", "_SD", "_Upp")))
}
orden_ordenado <- intersect(orden_ordenado, names(resultados_wide))
resultados_wide <- resultados_wide[, orden_ordenado, drop = FALSE]

# --------------------------
# Bootstrap para Observadas/Singletons/Doubletons por semana (robusto)
# --------------------------
resumen_list <- list()

for (i in seq_len(ncol(matriz))) {
  abund_acumulada <- rowSums(matriz[, 1:i, drop = FALSE])
  stats_df <- calc_stats_safe(abund_acumulada, R = 100)  # ajustar R si se desea
  stats_df$Unidad <- paste0("Unidad", i)
  resumen_list[[i]] <- stats_df
}

resumen_total <- bind_rows(resumen_list)

# Ajustar nombres y pivotar
resumen_total <- resumen_total %>%
  select(Unidad, Metrica, Mean, SD, Low, Upp)

# Asegurar mapeo de métricas (si por algún motivo vienen numéricas)
resumen_total <- resumen_total %>%
  mutate(Metrica = as.character(Metrica))

# Sumarizar en una sola fila por unidad
resumen_pivot <- resumen_total %>%
  tidyr::pivot_wider(
    names_from = Metrica,
    values_from = c(Mean, SD, Low, Upp),
    names_glue = "{Metrica}_{.value}"
  )

# Reordenar columnas lógicamente (permitiendo que falten algunas)
cols_req <- c(
  "Unidad",
  "Observadas_Low", "Observadas_Mean", "Observadas_SD", "Observadas_Upp",
  "Singletons_Low", "Singletons_Mean", "Singletons_SD", "Singletons_Upp",
  "Doubletons_Low", "Doubletons_Mean", "Doubletons_SD", "Doubletons_Upp"
)
cols_exist <- intersect(cols_req, names(resumen_pivot))
resumen_pivot <- resumen_pivot %>% select(all_of(cols_exist))

# Integrar resumen con resultados_wide
resultados_Final <- dplyr::left_join(resumen_pivot, resultados_wide, by = "Unidad")

# --------------------------
# Bootstrap seguro para Chao1 (si se desea)
# --------------------------
bootstrap_list <- list()
for (i in seq_len(ncol(matriz))) {
  abund_acumulada <- rowSums(matriz[, 1:i, drop = FALSE])
  chao_boot <- calc_chao1_safe(abund_acumulada, R = 100)
  chao_boot$Unidad <- paste0("Unidad", i)
  bootstrap_list[[i]] <- chao_boot
}
bootstrap_total <- bind_rows(bootstrap_list)

# Integrar con resultados_Final
resultados_Final1 <- dplyr::left_join(resultados_Final, bootstrap_total, by = "Unidad")

# --------------------------
# Revisar estructura final y guardar
# --------------------------
print("Resumen de unidades con problemas (si hay):")
if (length(log_insuficientes) == 0) {
  print("Ninguna unidad reportó fallo en ChaoSpecies.")
} else {
  print(log_insuficientes)
}

# Vista rápida
print("Vista previa de resultados finales:")
print(dplyr::glimpse(resultados_Final1))

# Guardar a Excel
output_file <- "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/5_Estimadores_Abundancia.xlsx"
openxlsx::write.xlsx(resultados_Final1, output_file, overwrite = TRUE)
cat("\n Archivo exportado correctamente en:\n", output_file, "\n")
