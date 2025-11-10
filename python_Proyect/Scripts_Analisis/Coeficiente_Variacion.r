# ==========================================================
# ANÁLISIS DE ESTIMADORES DE DIVERSIDAD (PARAMÉTRICOS Y NO PARAMÉTRICOS)
# ============================================================
# 🔹 ESTIMADORES DE RIQUEZA TIPO ESTIMATES + IC95%
# ============================================================

# ============================================================
# 🔧 1️⃣ Cargar o instalar paquetes necesarios
# ============================================================
paquetes <- c("readxl", "vegan", "dplyr")

# Instala los que falten
instalar <- paquetes[!(paquetes %in% installed.packages()[,"Package"])]
if(length(instalar)) install.packages(instalar)

# Cargar paquetes
lapply(paquetes, library, character.only = TRUE)

# ============================================================
# 📂 2️⃣ Cargar datos desde Excel
# ============================================================
ruta <- "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/Tabla_Abundancia_Semanal.xlsx"
datos <- read_excel(ruta, sheet = 1)


# ============================================================
# 📊 Evaluar el coeficiente de variación (CV) antes de estimar riqueza
# ============================================================

# 🔹 Paquete necesario
library(SpadeR)
library(tidyverse)

# 🔹 Convertir tu dataframe 'datos' en una matriz de abundancia
# Primera columna = ESPECIE
matriz <- datos %>%
  column_to_rownames("ESPECIE") %>%
  as.matrix()

# ============================================================
# 🔹 Preparar datos de incidencia
# ============================================================
# Convierte la matriz de abundancias a matriz de incidencias (1 = presente, 0 = ausente)
incidencia <- ifelse(matriz > 0, 1, 0)

# ============================================================
# 📊 Evaluar el coeficiente de variación (CV) antes de estimar riqueza
# ============================================================

library(SpadeR)

# 1️⃣ Asegúrate de tener creada la matriz de incidencias
# (si ya la tienes no repitas esta parte)
# incidencia <- ifelse(matriz > 0, 1, 0)

T <- ncol(incidencia)
S_obs <- nrow(incidencia)
f_i <- rowSums(incidencia)

# 2️⃣ Crear el vector en formato correcto para SpadeR
incidencia_freq <- c(T, f_i)

# 3️⃣ Calcular el estimador
spade_temp <- SpadeR::ChaoSpecies(incidencia_freq, datatype = "incidence_freq")

# 4️⃣ Extraer el CV con el nombre correcto
cv_valor <- as.numeric(spade_temp$Basic_data_information$Value[
  spade_temp$Basic_data_information$Variable == "CV"
])

# 5️⃣ Si SpadeR no devuelve CV (raro, pero posible), calcularlo manualmente
if (is.na(cv_valor) || length(cv_valor) == 0) {
  cv_valor <- sd(f_i) / mean(f_i)
  message("⚠️ CV calculado manualmente a partir de las frecuencias de incidencia.")
}

# ============================================================
# 🧭 Interpretación automática del CV (mensaje estilo EstimateS)
# ============================================================

if (cv_valor <= 0.5) {
  mensaje <- paste0(
    "📘 NOTA: El coeficiente de variación estimado para la distribución de incidencias es ",
    round(cv_valor, 3),
    ".\nComo CV ≤ 0.5, la comunidad es relativamente homogénea.\n",
    "➡️ Se recomienda utilizar la versión **corregida por sesgo (bias-corrected)** del estimador Chao2."
  )
} else {
  mensaje <- paste0(
    "⚠️ NOTA: El coeficiente de variación estimado para la distribución de incidencias es ",
    round(cv_valor, 3),
    ".\nDado que CV > 0.5, existe alta heterogeneidad en la detectabilidad de las especies.\n",
    "➡️ Anne Chao recomienda **usar la versión clásica (Classic)** del estimador Chao2 en lugar de la bias-corrected.\n",
    "Posteriormente, compara los valores de **Chao2 clásico** y **ICE**, y reporta el mayor como mejor estimador de la riqueza basada en incidencias."
  )
}

# ============================================================
# 📋 Mostrar el mensaje y el valor numérico
# ============================================================
cat(mensaje, "\n\nValor numérico del CV:", round(cv_valor, 4), "\n")


# ============================================================  Falta exportar avisos del CV a un archivo de texto  
# 📂 3️⃣ Exportar el mensaje a un archivo de text
# ============================================================
ruta_salida <- "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/Analisis_CV.txt"
writeLines(mensaje, con = ruta_salida)      
# ============================================================