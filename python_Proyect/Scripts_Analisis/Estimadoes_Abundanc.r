# ==========================================================
# ANÁLISIS DE ESTIMADORES DE DIVERSIDAD (PARAMÉTRICOS Y NO PARAMÉTRICOS)
# Versión consolidada para Abundancia e Incidencia
# ============================================================

# ============================================================
# 🔧 1️⃣ Cargar o instalar paquetes necesarios
# ============================================================

# Función para asegurar que los paquetes estén instalados y cargados
cargar_paquetes <- function(pkgs) {
  nuevos_paquetes <- pkgs[!(pkgs %in% installed.packages()[, "Package"])]
  if (length(nuevos_paquetes)) {
    install.packages(nuevos_paquetes, dependencies = TRUE)
  }
  sapply(pkgs, require, character.only = TRUE)
}

paquetes_necesarios <- c("readxl", "dplyr", "SpadeR", "openxlsx")
cargar_paquetes(paquetes_necesarios)

# ============================================================
# 📂 2️⃣ Cargar datos desde Excel
# ============================================================

# --- Parámetros de entrada y salida ---
ruta_entrada <- "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/Tabla_Abundancia_Semanal.xlsx"
ruta_salida <- "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/Reporte_Estimadores_Riqueza.xlsx"

# --- Carga y preparación de la matriz de abundancia ---
datos <- read_excel(ruta_entrada, sheet = 1)

# Asignar nombres de especies a las filas y eliminar la primera columna
rownames(datos) <- datos[[1]]
matriz <- as.data.frame(datos[, -1])

# Asegurar que todos los datos son numéricos y los NA se convierten en 0
matriz <- as.data.frame(sapply(matriz, as.numeric))
matriz[is.na(matriz)] <- 0

cat("✅ Datos cargados y matriz preparada con", nrow(matriz), "especies y", ncol(matriz), "muestras.\n")

# ============================================================
# 🧮 3️⃣ Análisis basado en ABUNDANCIA
# ============================================================

cat("\n--- Calculando estimadores basados en Abundancia ---\n")

# Vector de abundancias totales (suma de individuos por especie en todas las muestras)
abund_total <- rowSums(matriz)

# Calcular estimadores con SpadeR para datos de abundancia
spade_abund_result <- SpadeR::ChaoSpecies(abund_total, datatype = "abundance")

# Extraer y formatear la tabla de resultados
tabla_abund <- as.data.frame(spade_abund_result$Species_table)
colnames(tabla_abund) <- c("Estimador", "Estimacion", "Error_Std", "IC_Inferior", "IC_Superior")
tabla_abund$Tipo_Dato <- "Abundancia"

print(tabla_abund)

# ============================================================
# 📊 4️⃣ Análisis basado en INCIDENCIA (Presencia/Ausencia)
# ============================================================

cat("\n--- Calculando estimadores basados en Incidencia ---\n")

# Crear matriz de incidencia (1 si la especie está presente, 0 si no)
incidencia <- ifelse(matriz > 0, 1, 0)

# --- Parámetros básicos de incidencia ---
T_muestras <- ncol(incidencia)      # Número total de unidades de muestreo (semanas)
S_obs <- nrow(incidencia)           # Número de especies observadas
freq_incid <- rowSums(incidencia)   # Frecuencia de cada especie (en cuántas semanas aparece)
Q1 <- sum(freq_incid == 1)          # Singletons: especies que aparecen en 1 sola muestra
Q2 <- sum(freq_incid == 2)          # Doubletons: especies que aparecen en 2 muestras

# --- Estimadores de SpadeR para incidencia ---
spade_incid_result <- SpadeR::ChaoSpecies(incidencia, datatype = "incidence_raw")
tabla_incid <- as.data.frame(spade_incid_result$Species_table)
colnames(tabla_incid) <- c("Estimador", "Estimacion", "Error_Std", "IC_Inferior", "IC_Superior")

# --- Cálculo manual del estimador Bootstrap ---
p_i <- freq_incid / T_muestras
S_boot <- S_obs + sum((1 - p_i)^T_muestras)
var_boot <- sum(((1 - p_i)^T_muestras) * (1 - (1 - p_i)^T_muestras))
se_boot <- sqrt(var_boot)

bootstrap_result <- data.frame(
  Estimador = "Bootstrap",
  Estimacion = S_boot,
  Error_Std = se_boot,
  IC_Inferior = S_boot - 1.96 * se_boot,
  IC_Superior = S_boot + 1.96 * se_boot
)

# --- Combinar resultados de incidencia ---
tabla_incid_completa <- bind_rows(tabla_incid, bootstrap_result)
tabla_incid_completa$Tipo_Dato <- "Incidencia"

print(tabla_incid_completa)

# ============================================================
# 📋 5️⃣ Consolidar y exportar resultados
# ============================================================

# --- Tabla de resumen de parámetros básicos ---
info_basica <- data.frame(
  Parametro = c("Especies Observadas (S.obs)", "Unidades de Muestreo (T)", "Singletons de Incidencia (Q1)", "Doubletons de Incidencia (Q2)"),
  Valor = c(S_obs, T_muestras, Q1, Q2)
)

# --- Tabla final con todos los estimadores ---
tabla_final_estimadores <- bind_rows(tabla_abund, tabla_incid_completa) %>%
  select(Tipo_Dato, Estimador, Estimacion, Error_Std, IC_Inferior, IC_Superior) %>%
  mutate(across(where(is.numeric), round, 3))

# --- Crear un libro de Excel y añadir las hojas ---
wb <- createWorkbook()
addWorksheet(wb, "Resumen_Parametros")
addWorksheet(wb, "Estimadores_Riqueza")

writeData(wb, "Resumen_Parametros", info_basica)
writeData(wb, "Estimadores_Riqueza", tabla_final_estimadores)

saveWorkbook(wb, ruta_salida, overwrite = TRUE)

cat("\n✅ Reporte de estimadores exportado con éxito a:\n", ruta_salida, "\n")

cat("\n--- Resumen de Parámetros ---\n")
print(info_basica)
cat("\n--- Tabla Consolidada de Estimadores ---\n")
print(tabla_final_estimadores)
