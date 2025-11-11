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
    title = "Curvas de Diversidad de Hills (q = 0, 1, 2)",
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
  "🔹 En el análisis de diversidad basado en números de Hills:
  
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

ggsave("D:/CORPONOR 2025/Backet/python_Proyect/Resultados/Grafico_Hills_iNEXT.png",
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
  body_add_par("Curvas de números de Hills (iNEXT)", style = "heading 1") %>%
  body_add_img(
    src = "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/Grafico_Hills_iNEXT.png",
    width = 6.5, height = 4.5, style = "centered"
  ) %>%
  body_add_par("Interpretación de los números de Hills", style = "heading 2") %>%
  body_add_fpar(fpar(
    paste(texto_final, collapse = "\n"),
    fp_p = fp_par(text.align = "justify")
  ))  # ← también justificado

# --- Guardar el documento ---
print(doc, target = "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/Informe_Completo_iNEXT.docx")



#----------------------Convertir el documento a pdf-----------------------------