# ==========================================================
# ANÁLISIS DE ESTIMADORES DE DIVERSIDAD (PARAMÉTRICOS Y NO PARAMÉTRICOS)
# ============================================================
# ESTIMADORES DE RIQUEZA TIPO ESTIMATES + IC95%
# ============================================================

# ============================================================
# 1️⃣ Cargar o instalar paquetes necesarios
# ============================================================

paquetes <- c("readxl", "vegan", "dplyr", "tibble", "SpadeR")

# Instala los que falten
instalar <- paquetes[!(paquetes %in% installed.packages()[,"Package"])]

if(length(instalar) > 0){
  install.packages(instalar)
}

# Cargar paquetes
lapply(paquetes, library, character.only = TRUE)

# ============================================================
# 2️⃣ Cargar datos desde Excel
# ============================================================

ruta <- "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/3_Tabla_Abundancia_Semanal.xlsx"

datos <- read_excel(ruta, sheet = 1)

# ============================================================
# Evaluar el coeficiente de variación (CV) antes de estimar riqueza
# ============================================================

# ============================================================
# Convertir dataframe a matriz de abundancia
# ============================================================

matriz <- datos %>%
  tibble::column_to_rownames("ESPECIE") %>%
  mutate(across(everything(), as.numeric)) %>%
  as.matrix()

# Reemplazar NA por 0
matriz[is.na(matriz)] <- 0

# Eliminar especies sin registros
matriz <- matriz[rowSums(matriz) > 0, ]

# ============================================================
# Preparar datos de incidencia
# ============================================================

# Convierte la matriz de abundancias a matriz de incidencias
# (1 = presente, 0 = ausente)

incidencia <- ifelse(matriz > 0, 1, 0)

# ============================================================
# Evaluar el coeficiente de variación (CV)
# ============================================================

T <- ncol(incidencia)
S_obs <- nrow(incidencia)

f_i <- rowSums(incidencia)

# Q1 y Q2
Q1 <- sum(f_i == 1)
Q2 <- sum(f_i == 2)

# Advertencia para Q2
if(Q2 == 0){

  warning(
    "Q2 = 0. Chao2 puede ser inestable ",
    "o no calculable con estos datos."
  )

}

# ============================================================
# Crear vector en formato correcto para SpadeR
# ============================================================

incidencia_freq <- c(T, f_i)

# ============================================================
# Calcular el estimador
# ============================================================

spade_temp <- tryCatch({

  SpadeR::ChaoSpecies(
    incidencia_freq,
    datatype = "incidence_freq"
  )

}, error = function(e){

  message("\n⚠️ ERROR EN ChaoSpecies():")
  message(e$message)

  return(NULL)

})

# ============================================================
# Extraer el CV con el nombre correcto
# ============================================================

if(!is.null(spade_temp)){

  cv_valor <- suppressWarnings(
    as.numeric(
      spade_temp$Basic_data_information$Value[
        spade_temp$Basic_data_information$Variable == "CV"
      ]
    )
  )

} else {

  cv_valor <- NA

}

# ============================================================
# Si SpadeR no devuelve CV, calcularlo manualmente
# ============================================================

if(is.na(cv_valor) || length(cv_valor) == 0){

  cv_valor <- stats::sd(f_i, na.rm = TRUE) /
    mean(f_i, na.rm = TRUE)

  message(
    "⚠️ CV calculado manualmente a partir ",
    "de las frecuencias de incidencia."
  )

}

# ============================================================
# Interpretación automática del CV (mensaje estilo EstimateS)
# ============================================================

if (cv_valor <= 0.5) {

  mensaje <- paste0(
    "NOTA: El coeficiente de variación estimado para la distribución de incidencias es ",
    round(cv_valor, 3),
    ".\nComo CV ≤ 0.5, la comunidad es relativamente homogénea.\n",
    "Se recomienda utilizar la versión corregida por sesgo (bias-corrected) del estimador Chao2."
  )

} else {

  mensaje <- paste0(
    "⚠️ NOTA: El coeficiente de variación estimado para la distribución de incidencias es ",
    round(cv_valor, 3),
    ".\nDado que CV > 0.5, existe alta heterogeneidad en la detectabilidad de las especies.\n",
    "Anne Chao recomienda usar la versión clásica (Classic) del estimador Chao2.\n",
    "Posteriormente, compara los valores de Chao2 clásico e ICE, y reporta el mayor como mejor estimador de la riqueza basada en incidencias."
  )

}

# ============================================================
# Mostrar el mensaje y el valor numérico
# ============================================================

cat(
  mensaje,
  "\n\nValor numérico del CV:",
  round(cv_valor, 4),
  "\n"
)

# ============================================================
# Exportar avisos del CV a un archivo de texto
# ============================================================

ruta_salida <- "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/3.1_Analisis_CV.txt"

writeLines(
  c(
    mensaje,
    "",
    paste0(
      "Valor numérico del CV: ",
      round(cv_valor, 4)
    )
  ),
  con = ruta_salida
)

# ============================================================