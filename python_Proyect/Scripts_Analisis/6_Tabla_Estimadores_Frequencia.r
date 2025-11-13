#---------------------------------- Análisis de estimadores clásicos de incidencia (frecuencia/presencia) -----------------------

# --- Paquetes necesarios ---
paquetes <- c("readxl", "dplyr", "tidyr", "boot", "SpadeR", "openxlsx", "stringr")
instalar <- paquetes[!(paquetes %in% installed.packages()[,"Package"])]
if(length(instalar)) install.packages(instalar)
lapply(paquetes, library, character.only = TRUE)
library(tibble)

# --- Cargar datos ---
ruta <- "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/Tabla_Abundancia_Semanal.xlsx"
datos <- read_excel(ruta, sheet = 1)


# --- Convertir a matriz de presencia/ausencia ---
matriz <- datos %>%
  column_to_rownames("ESPECIE") %>%
  as.matrix()

# Forzar 0/1 por si hay conteos
matriz[matriz > 0] <- 1
colnames(matriz) <- paste0("Unidad", 1:ncol(matriz))

# --- Verificar estructura ---
str(matriz)

# Total de especies observadas
sum(rowSums(matriz) > 0)

# Total de individuos o incidencias
sum(matriz)

# Revisión de valores faltantes
sum(is.na(matriz))

# Valores ya calculados necesarios para estimadores de incidencia cuando la variacion es baja---------

# --- Cargar datos ed abundancia ---
ruta <- "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/Estimadores_Abundancia.xlsx"
datos_abundancia <- read_excel(ruta, sheet = 1)




#------------------------------------ Calcular estimadores clásicos de incidencia por frecuencia ---------------------



library(dplyr)
library(tidyr)
library(stringr)
library(SpadeR)

# --- Estimadores esperados (los que devuelve ChaoSpecies) ---
expected_estimators <- c(
  "Homogeneous Model", "Chao2 (Chao, 1987)", "Chao2-bc",
  "iChao2 (Chiu et al. 2014)", "ICE (Lee & Chao, 1994)",
  "ICE-1 (Lee & Chao, 1994)", "1st order jackknife", "2nd order jackknife"
)

resultados_list <- vector("list", ncol(matriz))

for (i in seq_len(ncol(matriz))) {
  cat("\n--- Unidad acumulada:", i, "---\n")
  freq_acumulada <- matriz[, 1:i, drop = FALSE]
  
  if (ncol(freq_acumulada) < 2) {
    # Datos insuficientes → crear tabla con NA pero incluir número de especies observadas
    tabla <- data.frame(
      Estimador = expected_estimators,
      Estimate = NA_real_,
      s.e. = NA_real_,
      `95%Lower` = NA_real_,
      `95%Upper` = NA_real_,
      Unidad = paste0("Unidad", i),
      Observadas = sum(rowSums(freq_acumulada) > 0),
      stringsAsFactors = FALSE
    )
  } else {
    # Ejecutar ChaoSpecies en formato de incidencia cruda
    df_result <- tryCatch({
      res <- ChaoSpecies(freq_acumulada, datatype = "incidence_raw")
      tabla_tmp <- as.data.frame(res$Species_table, stringsAsFactors = FALSE)
      if (!"Estimador" %in% names(tabla_tmp)) tabla_tmp$Estimador <- rownames(tabla_tmp)
     
      # Asegurar columnas estándar
      for (col in c("Estimate", "s.e.", "95%Lower", "95%Upper")) {
        if (!col %in% names(tabla_tmp)) tabla_tmp[[col]] <- NA_real_
      }
      tabla <- tabla_tmp %>%
        select(Estimador, Estimate, `s.e.`, `95%Lower`, `95%Upper`) %>%
        mutate(
          Unidad = paste0("Unidad", i),
          Observadas = sum(rowSums(freq_acumulada) > 0)
        )
      tabla
    }, error = function(e) {
      warning("ChaoSpecies falló en Unidad ", i, " → se rellenan NA: ", e$message)
      data.frame(
        Estimador = expected_estimators,
        Estimate = NA_real_,
        s.e. = NA_real_,
        `95%Lower` = NA_real_,
        `95%Upper` = NA_real_,
        Unidad = paste0("Unidad", i),
        Observadas = sum(rowSums(freq_acumulada) > 0),
        stringsAsFactors = FALSE
      )
    })
  }
  
  tabla$Estimador <- trimws(as.character(tabla$Estimador))
  

  resultados_list[[i]] <- tabla
}

print(resultados_list)
view(resultados_list)
str(resultados_list) 

print(datos_abundancia)

str(datos_abundancia) 

#------------------------------------ Unir y armonizar resultados con estimadores de abundancia ---------------------

library(dplyr)
library(stringr)
library(tidyr)
library(purrr)

# --- 1️⃣ Armonizar columnas y corregir nombres de unidad ---
resultados_list <- lapply(seq_along(resultados_list), function(i) {
  df <- resultados_list[[i]]
  
  # Armonizar nombres
  names(df) <- str_replace_all(names(df), c(
    "95%Lower" = "X95.Lower",
    "95%Upper" = "X95.Upper",
    " " = "."
  ))
  
  # Si tiene columna Unidad, la reemplazamos con el nombre correcto
  if ("Unidad" %in% names(df)) {
    df$Unidad <- paste0("Unidad", i)
  } else {
    # Si no la tiene, la creamos
    df$Unidad <- paste0("Unidad", i)
  }
  
  df
})

# --- 2️⃣ Unir todo ---
datos_combinados <- bind_rows(resultados_list)

# --- 3️⃣ Seleccionar columnas relevantes ---
datos_combinados <- datos_combinados %>%
  select(Unidad, Estimador, Estimate, s.e., X95.Lower, X95.Upper, Observadas) %>%
  rename(
    Mean = Estimate,
    SD = s.e.,
    Low = X95.Lower,
    Upp = X95.Upper
  )

# --- 4️⃣ Pivotear a formato ancho ---
datos_frecuencia <- datos_combinados %>%
  mutate(Estimador = str_replace_all(Estimador, "[^A-Za-z0-9]+", "_")) %>%
  pivot_wider(
    names_from = Estimador,
    values_from = c(Low, Mean, SD, Upp),
    names_glue = "{Estimador}_{.value}"
  ) %>%
  relocate(Unidad, starts_with("Observadas"), .before = everything())

# --- 5️⃣ Revisar resultado ---
print(datos_frecuencia)
view(datos_frecuencia)
str(datos_frecuencia)

#--------------Calcular la desviación estándar de las estimaciones de frecuencia --------------------
library(dplyr)
library(stringr)

# --- Calcular SD para todas las columnas que terminan en _Low, _Upp y _SD ---
for (estimador in c("Homogeneous_Model", "Chao2_Chao_1987_", "Chao2_bc",
                     "iChao2_Chiu_et_al_2014_", "ICE_Lee_Chao_1994_",
                     "ICE_1_Lee_Chao_1994_", "1st_order_jackknife",
                     "2nd_order_jackknife")) {
  
  low_col <- paste0(estimador, "_Low")
  upp_col <- paste0(estimador, "_Upp")
  sd_col  <- paste0(estimador, "_SD")
  
  if (all(c(low_col, upp_col) %in% names(datos_frecuencia))) {
    datos_frecuencia[[sd_col]] <- (datos_frecuencia[[upp_col]] - datos_frecuencia[[low_col]]) / (2 * 1.96)
  }
}

# --- Resultado ---
datos_frecuencia %>%
  select(Unidad, ends_with("_SD")) %>%
  print()

print(datos_frecuencia)
view(datos_frecuencia)
str(datos_frecuencia)
view(datos_abundancia)
str(datos_abundancia)


library(dplyr)
library(stringr)

# ============================
# 1️⃣ Copia de seguridad
# ============================
datos_frecuencia_mod <- datos_frecuencia

# ============================
# 2️⃣ Unir los valores observados
# ============================
datos_frecuencia_mod <- datos_frecuencia_mod %>%
  left_join(
    datos_abundancia %>%
      select(Unidad, Observadas_Low, Observadas_Mean, Observadas_SD, Observadas_Upp),
    by = "Unidad"
  )

# ============================
# 3️⃣ Reemplazar valores NA de la primera unidad por los observados
#     (primer código nuevo)
# ============================
cols_estimadores <- names(datos_frecuencia_mod)[str_detect(names(datos_frecuencia_mod), "(_Low|_Mean|_SD|_Upp)$")]

for (col in cols_estimadores) {
  # Si la Unidad1 tiene NA, reemplazar por los valores observados equivalentes
  if (is.na(datos_frecuencia_mod[[col]][datos_frecuencia_mod$Unidad == "Unidad1"])) {
    if (str_detect(col, "_Low$")) {
      datos_frecuencia_mod[[col]][datos_frecuencia_mod$Unidad == "Unidad1"] <-
        datos_frecuencia_mod$Observadas_Low[datos_frecuencia_mod$Unidad == "Unidad1"]
    } else if (str_detect(col, "_Mean$")) {
      datos_frecuencia_mod[[col]][datos_frecuencia_mod$Unidad == "Unidad1"] <-
        datos_frecuencia_mod$Observadas_Mean[datos_frecuencia_mod$Unidad == "Unidad1"]
    } else if (str_detect(col, "_SD$")) {
      datos_frecuencia_mod[[col]][datos_frecuencia_mod$Unidad == "Unidad1"] <-
        datos_frecuencia_mod$Observadas_SD[datos_frecuencia_mod$Unidad == "Unidad1"]
    } else if (str_detect(col, "_Upp$")) {
      datos_frecuencia_mod[[col]][datos_frecuencia_mod$Unidad == "Unidad1"] <-
        datos_frecuencia_mod$Observadas_Upp[datos_frecuencia_mod$Unidad == "Unidad1"]
    }
  }
}

# ============================
# 4️⃣ Interpolación condicional
#     (segundo código nuevo: interpola valores NA a partir de la segunda unidad)
# ============================
for (col in cols_estimadores) {
  valores <- datos_frecuencia_mod[[col]]
  if (any(is.na(valores))) {
    # Interpola los NA usando linealmente las unidades con datos
    indices <- which(!is.na(valores))
    if (length(indices) >= 2) {
      datos_frecuencia_mod[[col]] <- approx(
        x = indices,
        y = valores[indices],
        xout = seq_along(valores),
        method = "linear",
        rule = 2
      )$y
    }
  }
}

# ============================
# 5️⃣ Verificación final
# ============================
datos_frecuencia_mod %>%
  select(Unidad, starts_with("Homogeneous_Model")) %>%
  print()

View(datos_frecuencia_mod)

# --- Reordenar columnas de datos_frecuencia_mod por sufijo (Low, Mean, SD, Upp) ---

library(stringr)
library(dplyr)

# Tomamos los nombres originales
nombres <- colnames(datos_frecuencia_mod)

# Extraer base y sufijo (Low, Mean, SD, Upp), permitiendo doble guion bajo
partes <- str_match(nombres, "^(.*?)(?:__|_)(Low|Mean|SD|Upp)$")

# Crear tabla auxiliar
orden_aux <- data.frame(
  nombre = nombres,
  base = partes[, 2],
  sufijo = partes[, 3],
  stringsAsFactors = FALSE
)

# Definir orden deseado
orden_sufijos <- c("Low", "Mean", "SD", "Upp")

# Las columnas sin sufijo se quedan con NA, las movemos al inicio
orden_aux <- orden_aux %>%
  mutate(sufijo = factor(sufijo, levels = orden_sufijos)) %>%
  arrange(is.na(base), base, sufijo)

# Reordenar el data frame
datos_frecuencia_mod <- datos_frecuencia_mod[, orden_aux$nombre]

# Verificar resultado
colnames(datos_frecuencia_mod)
view(datos_frecuencia_mod)

#---------------- Agregar demas datos--------------------



# --- Calcular Singletons, Doubletons y Bootstrap acumulados ---
library(dplyr)
library(purrr)

# --- Cálculo de acumulados incluyendo Low, Mean, SD, Upp para Singletons y Doubletons ---
resultados_acumulados <- map_dfr(1:ncol(matriz), function(i) {
  # Submatriz acumulada hasta la unidad i
  submatriz <- matriz[, 1:i, drop = FALSE]
  
  # Calcular frecuencia total por especie
  incidencias <- rowSums(submatriz)
  
  # Calcular singletons y doubletons
  Q1 <- sum(incidencias == 1)
  Q2 <- sum(incidencias == 2)
  
  # Número de especies observadas
  t <- sum(incidencias > 0)
  
  # Parámetros para Bootstrap
  n <- ncol(submatriz)
  p1 <- Q1 / n
  
  # --- Bootstrap ---
  bootstrap_mean <- t + p1 * (1 - p1 / n)
  bootstrap_sd <- bootstrap_mean * 0.05
  bootstrap_low <- bootstrap_mean - 1.96 * bootstrap_sd
  bootstrap_upp <- bootstrap_mean + 1.96 * bootstrap_sd
  
  # --- Agregar error estimado para singletons y doubletons ---
  # Usamos una aproximación del 5% del valor como SD (mismo criterio de referencia)
  sing_sd <- Q1 * 0.05
  sing_low <- Q1 - 1.96 * sing_sd
  sing_upp <- Q1 + 1.96 * sing_sd
  
  doub_sd <- Q2 * 0.05
  doub_low <- Q2 - 1.96 * doub_sd
  doub_upp <- Q2 + 1.96 * doub_sd
  
  # --- Resultado acumulado ---
  tibble(
    Unidad = i,
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
})


# --- Convertir a formato de columnas (una por estimador acumulado) ---
# Similar a tus otros resultados en datos_frecuencia_mod

resultados_wide <- resultados_acumulados %>%
  select(-Unidad) %>%
  bind_cols()

# --- Unir al data frame principal ---
datos_frecuencia_mod <- bind_cols(datos_frecuencia_mod, resultados_wide)

# Verificar resultado
print(datos_frecuencia_mod)

# --- Reordenar columnas: Observadas, Singletons, Doubletons al inicio ---

datos_frecuencia_mod <- datos_frecuencia_mod %>%
  select(Unidad, Observadas, Observadas_Low, Observadas_Mean, Observadas_SD, Observadas_Upp,
   Singletons_Low, Singletons_Mean, Singletons_SD, Singletons_Upp, Doubletons_Low, Doubletons_Mean,
   Doubletons_SD, Doubletons_Upp, everything())

# Verificar orden
colnames(datos_frecuencia_mod)

#----------------------------------Manejar valores negativos en los estimadores de frecuencia -----------------------
library(dplyr)
library(stringr)
library(zoo)  # para interpolación

# --- Copia de seguridad ---
datos_frecuencia_mod_cor <- datos_frecuencia_mod

# --- Columnas de estimadores a evaluar ---
cols_estimadores <- names(datos_frecuencia_mod_cor)[
  str_detect(names(datos_frecuencia_mod_cor), "(_Low|_Mean|_SD|_Upp)$")
]

# --- Aplicar corrección columna por columna ---
for (col in cols_estimadores) {
  
  # Extraer los valores actuales de esa columna
  valores <- datos_frecuencia_mod_cor[[col]]
  
  # Detectar índices de valores negativos
  negativos <- which(valores < 0)
  
  if (length(negativos) > 0) {
    # Si hay valores negativos:
    
    # --- 1️⃣ Si el primer valor es negativo, reemplazarlo por el observado correspondiente ---
    if (negativos[1] == 1) {
      unidad1 <- datos_frecuencia_mod_cor$Unidad[1]
      fila_abun <- which(datos_abundancia$Unidad == unidad1)
      
      if (length(fila_abun) > 0) {
        if (str_detect(col, "_Low$")) {
          valores[1] <- datos_abundancia$Observadas_Low[fila_abun]
        } else if (str_detect(col, "_Mean$")) {
          valores[1] <- datos_abundancia$Observadas_Mean[fila_abun]
        } else if (str_detect(col, "_SD$")) {
          valores[1] <- datos_abundancia$Observadas_SD[fila_abun]
        } else if (str_detect(col, "_Upp$")) {
          valores[1] <- datos_abundancia$Observadas_Upp[fila_abun]
        }
      }
    }
    
    # --- 2️⃣ Para los valores negativos posteriores, interpolar linealmente ---
    # Los valores positivos se mantienen, los negativos se reemplazan con NA
    valores_interp <- valores
    valores_interp[valores_interp < 0] <- NA
    
    # Interpolación lineal solo donde existan valores positivos
    valores_interp <- na.approx(valores_interp, na.rm = FALSE)
    
    # Reemplazar en el data frame corregido
    datos_frecuencia_mod_cor[[col]] <- valores_interp
  }
}

# --- Verificación ---
datos_frecuencia_mod_cor %>%
  select(Unidad, starts_with("Homogeneous_Model")) %>%
  print()

#-------------------------Imprimir documento--------------------


library(openxlsx)

write.xlsx(datos_frecuencia_mod_cor, "D:/CORPONOR 2025/Backet/python_Proyect/Resultados/Estimadores_frecuencia.xlsx")
cat("\n✅ Archivo exportado correctamente con los estimadores agrupados por semana.\n")




