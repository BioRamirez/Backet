# ============================================================
#  ESTILO MINIMALISTA PROFESIONAL – EXCEL FRIENDLY
#  Autor: Juan Carlos Ramirez Gil (uso permanente)
#  Módulo universal para estandarizar todos los gráficos
# ============================================================

import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1️⃣  PALETA ÚNICA – Minimalista Profesional
# ------------------------------------------------------------

PALETA = [
    "#4A76A8",  # Azul profesional
    "#5A8F5E",  # Verde científico
    "#D29B5A",  # Naranja suave
    "#C76C5B",  # Rojo tenue
    "#7A6FA0",  # Morado grisáceo
    "#6AA3A1",  # Turquesa gris
    "#7C7C7C",  # Gris medio
    "#2F2F2F",  # Negro suave
]

def get_color(i):
    """Devuelve un color consistente de la paleta profesional."""
    return PALETA[i % len(PALETA)]


# ------------------------------------------------------------
# 2️⃣  ESTILO GENERAL PARA TODAS LAS FIGURAS
# ------------------------------------------------------------

def aplicar_estilo_minimalista(ax):
    """
    Aplica el estilo visual base a cualquier figura:
    - Fondo blanco limpio
    - Grid horizontal suave
    - Ejes con espinas grises
    - Tipografía homogénea
    """

    # Fondo blanco puro
    ax.set_facecolor("white")

    # Grilla SOLO horizontal
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Activar minor ticks
    ax.minorticks_on()

    # Espinas grises suaves
    for spine in ax.spines.values():
        spine.set_color("#6F6F6F")
        spine.set_linewidth(0.8)

    return ax


# ------------------------------------------------------------
# 3️⃣  ESTILO PARA GRÁFICOS DE BARRAS
# ------------------------------------------------------------

def estilo_barras(ax):
    """
    Ajustes profesionales para gráficos de barra:
    - Borde negro suave
    - Etiquetas numéricas con 2 decimales
    - Separación uniforme entre barras
    """

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.2f",
            fontsize=9,
            padding=3,
            color="#2F2F2F"
        )

    return ax


# ------------------------------------------------------------
# 4️⃣  FÁBRICA DE FIGURAS LISTA PARA PUBLICACIONES
# ------------------------------------------------------------

def figura_publicacion(width=8, height=5):
    """Crea una figura con parámetros óptimos para informes y artículos."""
    fig, ax = plt.subplots(figsize=(width, height))
    ax = aplicar_estilo_minimalista(ax)
    return fig, ax


# ------------------------------------------------------------
# 5️⃣  Estilo para curvas rango–abundancia
# ------------------------------------------------------------

def estilo_curvas(ax):
    """
    Añade parámetros predefinidos para curvas ecológicas:
    - Marcadores pequeños
    - Líneas moderadas
    """
    aplicar_estilo_minimalista(ax)

    ax.tick_params(axis='both', labelsize=10)
    return ax











fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(
    x=FD_resultados.index,
    y='FD',
    data=FD_resultados,
    palette=[get_color(i) for i in range(len(FD_resultados))]
)

aplicar_estilo_minimalista(ax)
estilo_barras(ax)

plt.xlabel("Cobertura")
plt.ylabel("Índice FD")
plt.title("")

plt.tight_layout()
plt.savefig("9.1_FVI_por_cobertura.png", dpi=300, bbox_inches="tight")
plt.show()
