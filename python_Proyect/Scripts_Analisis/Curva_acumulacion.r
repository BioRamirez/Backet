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

# Preparar matriz
rownames(datos) <- datos[[1]]
matriz <- as.data.frame(datos[, -1])
matriz <- as.data.frame(sapply(matriz, as.numeric))
matriz[is.na(matriz)] <- 0

# ============================================================
# 🧮 3️⃣ Función Chao1 modificada
# ============================================================
calc_chao1_mod <- function(abundancias) {
  abund <- abundancias[abundancias > 0]
  Sobs <- length(abund)
  F1 <- sum(abund == 1)
  F2 <- sum(abund == 2)
  
  if (F2 == 0) {
    est <- Sobs + (F1 * (F1 - 1)) / (2 * (F2 + 1))
  } else {
    est <- Sobs + (F1^2) / (2 * F2)
  }
  
  var_chao <- F2 * ((0.5 * (F1 / F2)^2) + (F1 / F2)^3 + 0.25 * (F1 / F2)^4)
  se <- sqrt(var_chao)
  
  lower <- est - 1.96 * se
  upper <- est + 1.96 * se
  lower <- ifelse(lower < Sobs, Sobs, lower)
  
  return(c(Chao1 = est, LCL = lower, UCL = upper))
}

# ============================================================
# 📊 4️⃣ Curva de acumulación y estimadores
# ============================================================
riqueza_obs <- specaccum(t(matriz), method = "collector")
n_muestras <- length(riqueza_obs$sites)
tabla_est <- data.frame(Muestra = 1:n_muestras, S.obs = riqueza_obs$richness)

for (i in 1:n_muestras) {
  submat <- matriz[, 1:i, drop = FALSE]
  if (sum(submat) == 0) next
  
  sp_pool <- specpool(t(submat))
  chao_corr <- calc_chao1_mod(rowSums(submat))
  
  tabla_est$ACE[i]         <- sp_pool$ACE
  tabla_est$ACE_LCL[i]     <- sp_pool$ACE - 1.96 * sp_pool$sd
  tabla_est$ACE_UCL[i]     <- sp_pool$ACE + 1.96 * sp_pool$sd
  
  tabla_est$Chao1[i]       <- chao_corr["Chao1"]
  tabla_est$Chao1_LCL[i]   <- chao_corr["LCL"]
  tabla_est$Chao1_UCL[i]   <- chao_corr["UCL"]
  
  tabla_est$Jackknife1[i]  <- sp_pool$jack1
  tabla_est$Jackknife2[i]  <- sp_pool$jack2
  tabla_est$Bootstrap[i]   <- sp_pool$boot
}

# ============================================================
# ✨ 5️⃣ Redondear y exportar
# ============================================================
tabla_est <- tabla_est %>% mutate(across(-Muestra, round, 3))

print(tabla_est)
write.xlsx(tabla_est, "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/Tabla_Estimadores_IC.xlsx", rowNames = FALSE)

cat("\n✅ Tabla exportada con éxito.\n")



#---------------------------------------------------------

install.packages(c("readxl", "vegan", "SpadeR", "iNEXT", "tidyverse"))
library(readxl)
library(vegan)
library(SpadeR)
library(iNEXT)
library(tidyverse)

# ============================================================
# 🔹 ESTIMADORES DE RIQUEZA TIPO ESTIMATES + IC95%
# ============================================================

# 1️⃣ Cargar datos
ruta <- "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/Tabla_Abundancia_Semanal.xlsx"
datos <- read_excel(ruta, sheet = 1)

rownames(datos) <- datos[[1]]
matriz <- as.data.frame(datos[, -1])
matriz <- as.data.frame(sapply(matriz, as.numeric))
matriz[is.na(matriz)] <- 0

# 2️⃣ Función para estimar Chao1 con corrección (caso F2 = 0)
calc_chao1_mod <- function(abundancias) {
  abund <- abundancias[abundancias > 0]
  Sobs <- length(abund)
  F1 <- sum(abund == 1)
  F2 <- sum(abund == 2)
  
  if (F2 == 0) {
    est <- Sobs + (F1 * (F1 - 1)) / (2 * (F2 + 1))
  } else {
    est <- Sobs + (F1^2) / (2 * F2)
  }
  
  var_chao <- F2 * ((0.5 * (F1 / F2)^2) + (F1 / F2)^3 + 0.25 * (F1 / F2)^4)
  se <- sqrt(var_chao)
  
  lower <- est - 1.96 * se
  upper <- est + 1.96 * se
  lower <- ifelse(lower < Sobs, Sobs, lower)
  
  return(c(Chao1 = est, LCL = lower, UCL = upper))
}

# 3️⃣ Crear tabla base
riqueza_obs <- specaccum(t(matriz), method = "collector")
n_muestras <- length(riqueza_obs$sites)
tabla_est <- data.frame(Muestra = 1:n_muestras, S.obs = riqueza_obs$richness)

# 4️⃣ Calcular estimadores
for (i in 1:n_muestras) {
  submat <- matriz[, 1:i, drop = FALSE]
  if (sum(submat) == 0) next
  
  # Estimadores de vegan
  sp_pool <- specpool(t(submat))
  
  # Chao1 con corrección manual
  chao_corr <- calc_chao1_mod(rowSums(submat))
  
  # Guardar resultados
  tabla_est$ACE[i]         <- sp_pool$ACE
  tabla_est$ACE_LCL[i]     <- sp_pool$ACE - 1.96 * sp_pool$sd
  tabla_est$ACE_UCL[i]     <- sp_pool$ACE + 1.96 * sp_pool$sd
  
  tabla_est$Chao1[i]       <- chao_corr["Chao1"]
  tabla_est$Chao1_LCL[i]   <- chao_corr["LCL"]
  tabla_est$Chao1_UCL[i]   <- chao_corr["UCL"]
  
  tabla_est$Jackknife1[i]  <- sp_pool$jack1
  tabla_est$Jackknife2[i]  <- sp_pool$jack2
  tabla_est$Bootstrap[i]   <- sp_pool$boot
}

# 5️⃣ Redondear y limpiar
tabla_est <- tabla_est %>%
  mutate(across(-Muestra, round, 3))

# 6️⃣ Mostrar y guardar tabla final
print(tabla_est)
write.xlsx(tabla_est, "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/Tabla_Estimadores2_IC.xlsx", rowNames = FALSE)


cat("\n✅ Tabla exportada con éxito.\n")


#................................................

# ============================================================
# Script: Estimadores tipo EstimateS (S.obs, Chao1, ACE, Jack1, Jack2, Bootstrap)
# Con IC95% (Chao1 analítico corregido + IC bootstrap para ACE y demás)
# ============================================================

# Paquetes necesarios
install_if_missing <- function(pkgs){
  to_install <- pkgs[!(pkgs %in% installed.packages()[,"Package"])]
  if(length(to_install)) install.packages(to_install)
}
install_if_missing(c("readxl", "vegan", "tidyverse"))
library(readxl)
library(vegan)
library(tidyverse)

# ----------------------
# Parámetros del usuario
# ----------------------
ruta <- "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/Tabla_Abundancia_Semanal.xlsx"  # <-- ajusta si hace falta
hoja <- 1
B <- 1000     # número de réplicas bootstrap (ajusta según tiempo/recursos)
seed <- 12345 # semilla para reproducibilidad

set.seed(seed)

# ----------------------
# 1. Cargar datos y preparar matriz
# ----------------------
df <- read_excel(ruta, sheet = hoja)
# Primera columna = especies
rownames(df) <- df[[1]]
mat <- as.data.frame(df[, -1])
mat <- as.data.frame(sapply(mat, as.numeric))   # forzar numérico
mat[is.na(mat)] <- 0

# Transponer si prefieres muestras en filas: aquí mantenemos (especies x muestras)
# mat: filas = especies, columnas = muestras (semanas)

# ----------------------
# 2. Funciones auxiliares
# ----------------------

# Chao1 (clásico) con corrección cuando F2 = 0, y varianza/IC según Chao (1984) / Colwell logic
calc_chao1_and_var <- function(abund_vector) {
  # abund_vector: abundancia por especie (vector)
  x <- abund_vector[abund_vector > 0]
  Sobs <- length(x)
  F1 <- sum(x == 1)
  F2 <- sum(x == 2)

  # estimador corregido
  if (F2 == 0) {
    chao1 <- Sobs + (F1 * (F1 - 1)) / (2 * (F2 + 1))
  } else {
    chao1 <- Sobs + (F1^2) / (2 * F2)
  }

  # varianza según Chao (1984) (cuando F2>0) - usar forma robusta
  if (F2 == 0) {
    # aproximación estable cuando F2=0 (evitar NaN)
    # usar var aproximada de (F1*(F1-1))/(2*(F2+1)) con delta method simplificado
    var_chao <- (F1 * (F1 - 1)) / 4  # aproximación conservadora
  } else {
    var_chao <- F2 * ( ( (F1 / F2)^2 ) / 2 + ( (F1 / F2)^3 ) + ( (F1 / F2)^4 ) / 4 )
  }

  se_chao <- sqrt(max(var_chao, 0))
  # Intervalo simétrico en escala original (EstimateS usa log-normal para algunos casos;
  # aquí calculamos tanto simétrico como log-based y devolvemos ambos)
  lcl_sym <- max(Sobs, chao1 - 1.96 * se_chao)
  ucl_sym <- chao1 + 1.96 * se_chao

  # log-normal CI (más estable para asimetría)
  if (chao1 > Sobs && se_chao > 0) {
    z <- 1.96
    log_mean <- log(chao1)
    log_se <- sqrt(log(1 + (se_chao^2) / (chao1^2)))
    lcl_log <- exp(log_mean - z * log_se)
    ucl_log <- exp(log_mean + z * log_se)
    # asegurar lcl >= Sobs
    lcl_log <- max(lcl_log, Sobs)
  } else {
    lcl_log <- lcl_sym
    ucl_log <- ucl_sym
  }

  return(list(
    Sobs = Sobs,
    Chao1 = chao1,
    Var = var_chao,
    SE = se_chao,
    LCL_sym = lcl_sym,
    UCL_sym = ucl_sym,
    LCL_log = lcl_log,
    UCL_log = ucl_log
  ))
}

# Calcula ACE, Jack1, Jack2, Bootstrap (est.) usando vegan::specpool on pooled data
calc_specpool <- function(submat) {
  # submat: especies x muestras
  # specpool espera muestras en filas: por eso transponemos
  res <- tryCatch(specpool(t(submat)), error = function(e) NULL)
  if (is.null(res)) {
    return(list(ACE = NA, jack1 = NA, jack2 = NA, boot = NA, ACE_sd = NA))
  } else {
    # specpool devuelve un data.frame con columnas: Species, chao, se.chao, jack1, jack2, ACE, SD
    # en algunas versiones los nombres pueden diferir; manejamos con fallback
    ACE <- if("ACE" %in% names(res)) res$ACE else if("ace" %in% names(res)) res$ace else NA
    jack1 <- if("jack1" %in% names(res)) res$jack1 else NA
    jack2 <- if("jack2" %in% names(res)) res$jack2 else NA
    boot  <- if("boot" %in% names(res)) res$boot else NA
    sdACE <- if("sd" %in% names(res)) res$sd else if("SD" %in% names(res)) res$SD else NA

    return(list(ACE = as.numeric(ACE),
                jack1 = as.numeric(jack1),
                jack2 = as.numeric(jack2),
                boot = as.numeric(boot),
                ACE_sd = as.numeric(sdACE)))
  }
}

# Funcion para calcular todos los estimadores sobre un submat (y opcional bootstrap)
calc_estimators_for_submat <- function(submat, B = 1000) {
  # submat: especies x muestras
  # 1) S.obs y Chao1 (analítico)
  chao_info <- calc_chao1_and_var(rowSums(submat))

  # 2) ACE, Jack1, Jack2, Bootstrap (vegan::specpool)
  sp <- calc_specpool(submat)

  # 3) Bootstrap por remuestreo de columnas para IC (resample columns with replacement)
  #    Calculamos distribuciones bootstrap para S.obs, Chao1, ACE
  ncol_sub <- ncol(submat)
  if (ncol_sub <= 1 || sum(submat) == 0) {
    # no es posible bootstrap con <2 muestras o sin registros
    boot_summary <- list(
      Sobs_mean = NA, Sobs_lci = NA, Sobs_uci = NA,
      Chao1_mean = NA, Chao1_lci = NA, Chao1_uci = NA,
      ACE_mean = NA, ACE_lci = NA, ACE_uci = NA
    )
  } else {
    # prealloc
    boot_Sobs <- numeric(B)
    boot_Chao1 <- numeric(B)
    boot_ACE <- numeric(B)

    for (b in 1:B) {
      cols_sample <- sample(1:ncol_sub, replace = TRUE)
      mat_b <- submat[, cols_sample, drop = FALSE]

      # S.obs bootstrap
      x_sum <- rowSums(mat_b)
      boot_Sobs[b] <- sum(x_sum > 0)

      # Chao1 bootstrap (usar la función definida)
      ch_b <- calc_chao1_and_var(x_sum)
      boot_Chao1[b] <- ch_b$Chao1

      # ACE bootstrap: usar specpool si posible
      sp_b <- tryCatch(specpool(t(mat_b)), error = function(e) NULL)
      if (!is.null(sp_b)) {
        ACE_b <- if("ACE" %in% names(sp_b)) sp_b$ACE else if("ace" %in% names(sp_b)) sp_b$ace else NA
      } else {
        ACE_b <- NA
      }
      boot_ACE[b] <- as.numeric(ACE_b)
    }

    # resumir (media y percentiles 2.5/97.5)
    boot_summary <- list(
      Sobs_mean = mean(boot_Sobs, na.rm = TRUE),
      Sobs_lci  = quantile(boot_Sobs, probs = 0.025, na.rm = TRUE),
      Sobs_uci  = quantile(boot_Sobs, probs = 0.975, na.rm = TRUE),

      Chao1_mean = mean(boot_Chao1, na.rm = TRUE),
      Chao1_lci  = quantile(boot_Chao1, probs = 0.025, na.rm = TRUE),
      Chao1_uci  = quantile(boot_Chao1, probs = 0.975, na.rm = TRUE),

      ACE_mean = mean(boot_ACE, na.rm = TRUE),
      ACE_lci  = quantile(boot_ACE, probs = 0.025, na.rm = TRUE),
      ACE_uci  = quantile(boot_ACE, probs = 0.975, na.rm = TRUE)
    )
  }

  # ensamblar salida
  out <- list(
    Sobs = chao_info$Sobs,
    Chao1 = chao_info$Chao1,
    Chao1_SE = chao_info$SE,
    Chao1_LCL_sym = chao_info$LCL_sym,
    Chao1_UCL_sym = chao_info$UCL_sym,
    Chao1_LCL_log = chao_info$LCL_log,
    Chao1_UCL_log = chao_info$UCL_log,

    ACE = sp$ACE,
    ACE_sd = sp$ACE_sd,
    jack1 = sp$jack1,
    jack2 = sp$jack2,
    bootstrap_est = sp$boot,

    # bootstrap CI summaries
    boot_Sobs = boot_summary$Sobs_mean,
    boot_Sobs_lci = boot_summary$Sobs_lci,
    boot_Sobs_uci = boot_summary$Sobs_uci,

    boot_Chao1 = boot_summary$Chao1_mean,
    boot_Chao1_lci = boot_summary$Chao1_lci,
    boot_Chao1_uci = boot_summary$Chao1_uci,

    boot_ACE = boot_summary$ACE_mean,
    boot_ACE_lci = boot_summary$ACE_lci,
    boot_ACE_uci = boot_summary$ACE_uci
  )

  return(out)
}

# ----------------------
# 3. Loop: calcular para cada nivel acumulado (1..ncol)
# ----------------------
acc <- specaccum(t(mat), method = "collector")
n_muestras <- length(acc$sites)

# Prepara dataframe resultado
res_df <- tibble(
  Muestras = 1:n_muestras,
  S.obs = NA_real_,
  S.obs_boot_mean = NA_real_,
  S.obs_boot_LCI = NA_real_,
  S.obs_boot_UCI = NA_real_,

  Chao1 = NA_real_,
  Chao1_SE = NA_real_,
  Chao1_LCL_sym = NA_real_,
  Chao1_UCL_sym = NA_real_,
  Chao1_LCL_log = NA_real_,
  Chao1_UCL_log = NA_real_,
  Chao1_boot_mean = NA_real_,
  Chao1_boot_LCI = NA_real_,
  Chao1_boot_UCI = NA_real_,

  ACE = NA_real_,
  ACE_sd = NA_real_,
  ACE_boot_mean = NA_real_,
  ACE_boot_LCI = NA_real_,
  ACE_boot_UCI = NA_real_,

  Jackknife1 = NA_real_,
  Jackknife2 = NA_real_,
  Bootstrap_est = NA_real_
)

# Bucle principal
for (i in 1:n_muestras) {
  submat <- mat[, 1:i, drop = FALSE]
  # saltar si no hay datos
  if (sum(submat) == 0) {
    next
  }

  cat("Procesando nivel acumulación:", i, "de", n_muestras, "\n")
  est <- calc_estimators_for_submat(submat, B = B)

  res_df$S.obs[i] <- est$Sobs
  res_df$S.obs_boot_mean[i] <- est$boot_Sobs
  res_df$S.obs_boot_LCI[i] <- est$boot_Sobs_lci
  res_df$S.obs_boot_UCI[i] <- est$boot_Sobs_uci

  res_df$Chao1[i] <- est$Chao1
  res_df$Chao1_SE[i] <- est$Chao1_SE
  res_df$Chao1_LCL_sym[i] <- est$Chao1_LCL_sym
  res_df$Chao1_UCL_sym[i] <- est$Chao1_UCL_sym
  res_df$Chao1_LCL_log[i] <- est$Chao1_LCL_log
  res_df$Chao1_UCL_log[i] <- est$Chao1_UCL_log
  res_df$Chao1_boot_mean[i] <- est$boot_Chao1
  res_df$Chao1_boot_LCI[i] <- est$boot_Chao1_lci
  res_df$Chao1_boot_UCI[i] <- est$boot_Chao1_uci

  res_df$ACE[i] <- est$ACE
  res_df$ACE_sd[i] <- est$ACE_sd
  res_df$ACE_boot_mean[i] <- est$boot_ACE
  res_df$ACE_boot_LCI[i] <- est$boot_ACE_lci
  res_df$ACE_boot_UCI[i] <- est$boot_ACE_uci

  res_df$Jackknife1[i] <- est$jack1
  res_df$Jackknife2[i] <- est$jack2
  res_df$Bootstrap_est[i] <- est$bootstrap_est
}

# ----------------------
# 4. Redondear, mostrar y exportar
# ----------------------
res_df_out <- res_df %>%
  mutate(across(where(is.numeric), ~ round(., 3)))

print(res_df_out)

# Exportar
write.xlsx(res_df_out, "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/Tabla_Estimadores3_IC.xlsx", rowNames = FALSE)
cat("Exportado: Tabla_Estimadores_EstimateS_Extendido_conIC.xlsx\n")




library(readxl)
library(iNEXT)

# --- Leer la tabla ---
ruta <- "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/Tabla_Abundancia_Semanal.xlsx"
datos <- read_excel(ruta, sheet = 1)

# --- Preparar la matriz de abundancia ---
rownames(datos) <- datos[[1]]       # nombres de especies
datos <- datos[ , -1]               # eliminar columna de nombres
matriz <- as.matrix(datos)

# Vector de abundancias totales por especie
abund_total <- rowSums(matriz)

# Crear el objeto iNEXT
inext_result <- iNEXT(abund_total, q = 0, datatype = "abundance")

# Información básica de los datos (análoga a “Data Summary” de EstimateS)
info <- iNEXT:::DataInfo(abund_total, datatype = "abundance")
info


# Tabla de estimadores asintóticos
asy <- inext_result$AsyEst
asy


ggiNEXT(inext_result, type = 1, se = TRUE, color.var = "site") +
  labs(
    title = "Curva de riqueza de especies (S-estimates)",
    x = "Número de individuos o unidades de muestreo",
    y = "Número de especies (S)"
  ) +
  theme_minimal()

library(SpadeR)

# Vector de abundancias totales
abund_total <- rowSums(matriz)

# Calcular estimadores clásicos de riqueza (tipo abundance)
spade_result <- ChaoSpecies(abund_total, datatype = "abundance")

# Mostrar el resumen
spade_result

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
