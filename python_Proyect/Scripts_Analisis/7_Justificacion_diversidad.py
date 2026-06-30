# ==========================================
# GUARDAR TODO EN ARCHIVO .TXT
# ==========================================

import os

# --- Título ---
titulo = "Justificación metodológica del cálculo de índices de diversidad\n"
titulo += "=" * 70 + "\n\n"

# --- Cuerpo del texto ---
texto = """
Los índices de diversidad ecológica fueron calculados a partir de las abundancias de especies registradas en cada cobertura vegetal. 
Para ello, se empleó el paquete scikit-bio (Rideout et al., 2016), ampliamente reconocido en bioinformática y ecología computacional 
para el análisis de diversidad alfa, beta y filogenética en comunidades biológicas.

El índice de Shannon-Wiener (H′) y el índice de Simpson se calcularon mediante la función alpha_diversity() de skbio.diversity, 
la cual implementa las fórmulas clásicas propuestas por Shannon & Weaver (1949) y Simpson (1949):

H′ = −∑(pᵢ ln pᵢ)
D = ∑(pᵢ²)

donde pᵢ representa la proporción de individuos de la especie i respecto al total de individuos. 
El índice de Simpson fue transformado a su forma complementaria (1 − D) para expresar la diversidad efectiva.

Los índices de equidad, riqueza, dominancia, Margalef y Menhinick se derivaron a partir de expresiones tradicionales 
de la ecología cuantitativa (Magurran, 2004; Begon, Townsend & Harper, 2006) aplicadas sobre los conteos de especies en cada cobertura.

"""

# --- Tabla convertida a texto plano ---
tabla_txt = "TABLA RESUMEN DE ÍNDICES Y FÓRMULAS\n"
tabla_txt += "-" * 70 + "\n"
tabla_txt += f"{'Índice':25}  {'Fórmula':30}  {'Referencia'}\n"
tabla_txt += "-" * 70 + "\n"

indices_info = [
    ("Riqueza (S)", "Número de especies presentes", "Magurran (2004)"),
    ("Abundancia (N)", "Total individuos observados", "Begon et al. (2006)"),
    ("Shannon (H′)", "−∑ pᵢ ln(pᵢ)", "Shannon & Weaver (1949)"),
    ("Simpson (1-D)", "1 − ∑ pᵢ²", "Simpson (1949); Rideout et al. (2016)"),
    ("Dominancia (D)", "∑ pᵢ²", "McIntosh (1967)"),
    ("Equidad (J′)", "H′/ln(S)", "Pielou (1966)"),
    ("Margalef (DMg)", "(S−1)/ln(N)", "Margalef (1958)"),
    ("Menhinick (DMn)", "S/√N", "Menhinick (1964)"),
]

for indice, formula, ref in indices_info:
    tabla_txt += f"{indice:25}  {formula:30}  {ref}\n"

tabla_txt += "\n"

# --- Referencias ---
referencias = [
    "Begon, M., Townsend, C. R., & Harper, J. L. (2006). Ecología: de individuos a ecosistemas (4.ª ed.). Oxford University Press.",
    "Magurran, A. E. (2004). Measuring biological diversity. Blackwell Publishing.",
    "Margalef, R. (1958). Information theory in ecology. General Systems, 3, 36–71.",
    "McIntosh, R. P. (1967). An index of diversity and the relation of certain concepts to diversity. Ecology, 48(3), 392–404.",
    "Menhinick, E. F. (1964). A comparison of some species–individuals diversity indices applied to samples of field insects. Ecology, 45(4), 859–861.",
    "Pielou, E. C. (1966). The measurement of diversity in different types of biological collections. Journal of Theoretical Biology, 13, 131–144.",
    "Rideout, J. R., et al. (2016). The scikit-bio package for bioinformatics and ecology. Bioinformatics, 32(15), 2229–2231.",
    "Shannon, C. E., & Weaver, W. (1949). The mathematical theory of communication. University of Illinois Press.",
    "Simpson, E. H. (1949). Measurement of diversity. Nature, 163, 688."
]

refs_txt = "REFERENCIAS (Formato APA 7ª Ed.)\n"
refs_txt += "-" * 70 + "\n"
refs_txt += "\n".join(referencias)

# --- Unir todo ---
contenido_final = titulo + texto + tabla_txt + refs_txt


# --- Guardar TXT ---
ruta_salida = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\7_Justificacion_Indices_Diversidad_skbio.txt"

import textwrap

# Ajustar el contenido a un máximo de 90 caracteres por línea
contenido_líneas_90 = textwrap.fill(contenido_final, width=90)

# Guardar el archivo
with open(ruta_salida, "w", encoding="utf-8") as f:
    f.write(contenido_líneas_90)


print("\nDocumento TXT generado correctamente en:\n", ruta_salida)
