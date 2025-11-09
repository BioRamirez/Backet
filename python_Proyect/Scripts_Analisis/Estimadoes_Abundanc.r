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




#--------------------------------------------------------------Calcular la efectividad del muestreo iNEXT---------------------



library(readxl)
library(iNEXT)
library(dplyr)
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


interpretar_inext <- function(iNEXT_result, sitio = "Muestreo") {
  info <- info
  asy <- inext_result$AsyEst
  
  n <- info$n
  s_obs <- info$S.obs
  sc <- info$SC
  f1 <- info$f1
  f2 <- info$f2
  
# Convertir las filas a una columna de nombres
asy_df <- tibble::rownames_to_column(as.data.frame(asy), var = "Indice")

# Acceso seguro por nombre del índice
s_obs <- asy_df$Observed[asy_df$Indice == "Species Richness"]
s_est <- asy_df$Estimator[asy_df$Indice == "Species Richness"]

shannon_obs <- asy_df$Observed[asy_df$Indice == "Shannon diversity"]
shannon_est <- asy_df$Estimator[asy_df$Indice == "Shannon diversity"]

simpson_obs <- asy_df$Observed[asy_df$Indice == "Simpson diversity"]
simpson_est <- asy_df$Estimator[asy_df$Indice == "Simpson diversity"]

# Diferencia de riqueza
s_diff <- s_est - s_obs

  
  # --- Interpretaciones dinámicas ---
  cobertura_txt <- case_when(
    sc >= 0.95 ~ "altamente representativo del ensamble analizado",
    sc >= 0.85 ~ "moderadamente representativo del ensamble",
    TRUE ~ "poco representativo; el muestreo podría no haber capturado toda la diversidad existente"
  )
  
  rareza_txt <- case_when(
    f1 / s_obs > 0.3 ~ "una alta proporción de especies raras, lo que sugiere que aún hay especies poco detectadas",
    f1 / s_obs > 0.15 ~ "una proporción moderada de especies raras",
    TRUE ~ "una baja proporción de especies raras, lo que indica una buena caracterización del ensamble"
  )
  
  completitud_txt <- case_when(
    s_diff <= 5 ~ "lo que confirma la alta completitud del inventario",
    s_diff <= 20 ~ "lo que sugiere que podrían existir algunas especies adicionales por detectar",
    TRUE ~ "lo que indica que el esfuerzo de muestreo debería incrementarse para capturar adecuadamente la riqueza total"
  )
  
  texto <- paste0(
    "En el ", sitio, " se registraron un total de ", n, " individuos y ",
    s_obs, " especies observadas. El valor de cobertura de muestra (SC = ",
    round(sc, 4), ") indica que el muestreo fue ", cobertura_txt, ". ",
    "Se detectaron ", f1, " especies con un solo individuo (singletons) y ",
    f2, " con dos individuos (doubletons), lo que representa ", rareza_txt, ". ",
    "El estimador asintótico de riqueza (Chao1) predice aproximadamente ",
    round(s_est, 2), " especies, es decir, unas ",
    round(s_diff, 2), " más de las observadas, ", completitud_txt, ". ",
    "El índice de Shannon aumentó de ", round(shannon_obs, 2), " a ",
    round(shannon_est, 2), ", y el índice de Simpson pasó de ",
    round(simpson_obs, 2), " a ", round(simpson_est, 2), 
    ", lo que refleja la estructura de abundancias y equidad de la comunidad."
  )
  
  return(texto)
}

# Ejemplo
interpretacion <- interpretar_inext(inext_result, sitio = "Muestreo")
cat(interpretacion)


#-------------------------------------------------------------- Gráfica de curvas de acumulación iNEXT---------------------

# Información básica de los datos (análoga a “Data Summary” de EstimateS)
info <- iNEXT:::DataInfo(abund_total, datatype = "abundance")
info

# Tabla de estimadores asintóticos
asy <- inext_result$AsyEst
asy


library(ggplot2)

ggiNEXT(inext_result, type = 1, se = TRUE) +
  labs(
    title = "Curva de acumulación de especies",
    subtitle = "Riqueza observada, interpolada y extrapolada",
    x = "Número de individuos muestreados",
    y = "Número de especies (S)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    legend.position = "bottom"
  )



ggsave(
  "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/Curva_Acumulacion_iNEXT.png",
  width = 10,
  height = 6,
  dpi = 300,
  bg = "white"   # 👈 asegura fondo blanco
)



#------------------------------------Graficar he interpetar demas estimadores de iNEXT---------------------

inext_todos <- iNEXT(abund_total, q = c(0, 1, 2), datatype = "abundance")

# ======================================
# 📦 Paquetes necesarios
# ======================================
library(iNEXT)
library(dplyr)
library(ggplot2)
library(glue)

# ======================================
# 🧮 1. Cargar tu objeto iNEXT
# ======================================
# (Ya generado previamente)
# inext_todos <- iNEXT(datos, q = c(0,1,2), datatype = "abundance")

# ======================================
# 📊 2. Extraer la tabla asintótica
# ======================================
resumen_asy <- inext_todos$AsyEst %>%
  mutate(
    Orden_q = c(0, 1, 2),
    Tipo = c("Riqueza (q=0)", "Diversidad de Shannon (q=1)", "Diversidad de Simpson (q=2)")
  ) %>%
  select(Tipo, Observed, Estimator, Est_s.e., `95% Lower`, `95% Upper`)

print(resumen_asy)

# ======================================
# 📈 3. Generar gráfico resumen de curvas
# ======================================
grafico_inext <- ggiNEXT(inext_todos, type = 1, se = TRUE, color.var = "Order.q") +
  labs(
    title = "Curvas de Diversidad de Hill (q = 0, 1, 2)",
    subtitle = "Riqueza observada, interpolada y extrapolada con intervalos de confianza (95%)",
    x = "Número de individuos (m)",
    y = "Número efectivo de especies (qD)",
    color = "Orden de diversidad (q)"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 11),
    legend.position = "top"
  )

print(grafico_inext)

# ======================================
# 🧩 4. Cálculo automático de suficiencia e interpretación
# ======================================
interpretacion2 <- resumen_asy %>%
  mutate(
    Diferencia = Estimator - Observed,
    Porcentaje_completitud = (Observed / Estimator) * 100,
    Conclusion = case_when(
      Porcentaje_completitud > 95 ~ "El muestreo fue suficiente; las curvas alcanzan una tendencia asintótica.",
      Porcentaje_completitud > 85 ~ "El muestreo fue adecuado, aunque podrían registrarse algunas especies adicionales.",
      TRUE ~ "El muestreo fue insuficiente; se espera mayor riqueza con un esfuerzo adicional."
    )
  )

print(interpretacion2)

# ======================================
# 📝 5. Generar texto interpretativo automático
# ======================================
texto_interpretacion <- glue(
  "🔹 En el análisis de diversidad basado en números de Hill:
  
  - Para q = 0 (riqueza de especies), se observaron {round(interpretacion2$Observed[1],1)} especies,
    con una estimación asintótica de {round(interpretacion2$Estimator[1],1)} ± {round(interpretacion2$Est_s.e.[1],2)}.
    Esto representa un {round(interpretacion2$Porcentaje_completitud[1],2)}% de completitud, lo que indica que {interpretacion2$Conclusion[1]}.

  - Para q = 1 (diversidad de Shannon), el número efectivo de especies fue de {round(interpretacion2$Observed[2],1)} 
    y la estimación asintótica alcanzó {round(interpretacion2$Estimator[2],1)} ± {round(interpretacion2$Est_s.e.[2],2)} 
    ({round(interpretacion2$Porcentaje_completitud[2],2)}% de completitud). {interpretacion2$Conclusion[2]}.

  - Para q = 2 (diversidad de Simpson), el número efectivo de especies dominantes fue de {round(interpretacion2$Observed[3],1)},
    con un valor esperado de {round(interpretacion2$Estimator[3],1)} ± {round(interpretacion2$Est_s.e.[3],2)}, 
    alcanzando un {round(interpretacion2$Porcentaje_completitud[3],2)}% de completitud. {interpretacion2$Conclusion[3]}.
  
  En conjunto, las tres curvas muestran una tendencia asintótica, evidenciando que el esfuerzo de muestreo fue adecuado 
  para representar la mayoría de las especies presentes en la comunidad."
)

cat(texto_interpretacion)


# ======================================
# 💾 6. Guardar resultados
# ======================================

ggsave("D:/CORPONOR 2025/Backet/python_Proyect/Resultados/Grafico_Hill_iNEXT.png",
       plot = grafico_inext, width = 8, height = 6, dpi = 300,
  bg = "white"   # 👈 asegura fondo blanco
)


#-----------------Guardar documento word con resultados de iNEXT-----------------------

library(officer)
library(magrittr)

# --- Capturar la salida exacta del segundo texto ---
texto_final <- capture.output(cat(texto_interpretacion))

# --- Crear documento Word con ambos gráficos e interpretaciones ---
doc <- read_docx() %>%
  # --- Sección 1: Curva de acumulación ---
  body_add_par("Curva de acumulación de especies (iNEXT)", style = "heading 1") %>%
  body_add_img(
    src = "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/Curva_Acumulacion_iNEXT.png",
    width = 6.5, height = 4.5, style = "centered"
  ) %>%
  body_add_par("Interpretación del análisis", style = "heading 2") %>%
  body_add_fpar(fpar(interpretacion, fp_p = fp_par(text.align = "justify"))) %>%  # ← tu texto justificado original
  
  # --- Sección 2: Números de Hill ---
  body_add_par("Curvas de números de Hill (iNEXT)", style = "heading 1") %>%
  body_add_img(
    src = "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/Grafico_Hill_iNEXT.png",
    width = 6.5, height = 4.5, style = "centered"
  ) %>%
  body_add_par("Interpretación de los números de Hill", style = "heading 2") %>%
  body_add_fpar(fpar(
    paste(texto_final, collapse = "\n"),
    fp_p = fp_par(text.align = "justify")
  ))  # ← también justificado

# --- Guardar el documento ---
print(doc, target = "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/Informe_Completo_iNEXT.docx")

# --- Convertir a PDF usando Word (solo Windows) ---
list.dirs("C:/Program Files", full.names = TRUE, recursive = FALSE)

#----------------------Convertir el documento a pdf-----------------------------


























#------------------------------------Análisis de estimadores clásicos de abundancia y bootstrap manual---------------------






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
















#-------------------------Ejemplo no final------------------------------

# ---------------------------------------------------------
# 🔹 Cálculo de estimadores por unidad de muestreo (tipo EstimateS)
# ---------------------------------------------------------

library(vegan)

# Aseguramos que cada fila = especie, cada columna = unidad de muestreo
matriz <- as.matrix(datos)

# Crear lista vacía para guardar resultados
resultados <- data.frame(
  Muestras = 1:ncol(matriz),
  S_obs = NA,
  Chao1 = NA,
  ACE = NA,
  Jack1 = NA,
  Bootstrap = NA
)

# Bucle acumulando muestreos progresivamente
for (i in 1:ncol(matriz)) {
  # Submatriz hasta la i-ésima unidad de muestreo (mantener forma matricial)
  submatriz <- matriz[, 1:i, drop = FALSE]
  
  # Abundancia total por especie acumulada
  abund_total <- rowSums(submatriz)
  
  # Filtrar especies presentes
  abund_total <- abund_total[abund_total > 0]
  
  # Calcular estimadores
  S_obs <- length(abund_total)
  chao1 <- estimateR(abund_total)["S.chao1"]
  ace <- estimateR(abund_total)["S.ACE"]
  
  # Jackknife y Bootstrap
  jack1 <- specpool(submatriz)$jack1
  boot <- specpool(submatriz)$boot
  
  # Guardar resultados
  resultados[i, 2:6] <- c(S_obs, chao1, ace, jack1, boot)
}

# Mostrar resultados tipo EstimateS
print(resultados)

# ---------------------------------------------------------
# 🔹 (Opcional) Exportar a Excel
# ---------------------------------------------------------
library(openxlsx)
write.xlsx(resultados, "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/Estimadores_por_muestreo.xlsx")

cat("\n✅ Archivo exportado con éxito a: D:/CORPONOR 2025/Backet/python_Proyect/Resultados/Estimadores_por_muestreo.xlsx\n")

# --- Librerías ---
library(ggplot2)
library(dplyr)
library(reshape2)

# --- Reorganizar ---
resultados_long <- melt(resultados, id.vars = "Muestras", variable.name = "Estimador", value.name = "Riqueza")


library(ggplot2)
library(dplyr)

# --- Gráfico con líneas suaves y proporciones similares a Excel ---
grafico <- ggplot(resultados_long, aes(x = Muestras, y = Riqueza, color = Estimador)) +
  geom_smooth(se = FALSE, method = "loess", span = 0.9, linewidth = 1.2) +
  geom_point(size = 3) +
  geom_text(
    data = resultados_long %>% group_by(Estimador) %>% slice_tail(n = 1),
    aes(label = round(Riqueza, 0)),
    hjust = -0.3, vjust = 0.5, size = 5, show.legend = FALSE
  ) +
  scale_color_manual(values = c("S_obs" = "#0072B2", "Chao1" = "#D55E00", "ACE" = "#009E73")) +
  scale_x_continuous(breaks = resultados$Muestras, expand = expansion(mult = c(0.05, 0.15))) +
  scale_y_continuous(limits = c(80, 340), breaks = seq(100, 340, by = 20)) +
  labs(
    title = "Curva de acumulación de especies",
    x = "Unidades de muestreo",
    y = "Riqueza estimada de especies",
    color = "Estimador"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold", hjust = 0.5),
    panel.grid.minor = element_blank(),
    aspect.ratio = 0.3  # 🔹 eje Y más corto (más parecido a Excel)
  )

# --- Mostrar ---
print(grafico)

# --- Si deseas exportar igual que