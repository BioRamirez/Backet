# ============================================================
# 🔗 CONFIGURAR GIT Y GITHUB DESDE RSTUDIO
# ============================================================

# 1️⃣ Instalar e importar el paquete necesario
#install.packages("usethis")
library(usethis)

# ------------------------------------------------------------
# 2️⃣ CONFIGURAR TU IDENTIDAD DE GIT
# (Estos datos deben coincidir con tu cuenta de GitHub)
# ------------------------------------------------------------
use_git_config(
  user.name = "BioRamirez",                    # tu nombre de usuario de GitHub
  user.email = "bioramirezjuan@gmail.com"      # el correo vinculado a GitHub
)

# Puedes verificar si Git está correctamente instalado en tu PC
system("git --version")   # o desde la terminal de RStudio: git --version


# ============================================================
# 3️⃣ CREAR UN TOKEN PERSONAL DE ACCESO (PAT)
# ============================================================

# Esto abrirá tu navegador para crear un token con permisos "repo"
usethis::create_github_token(scopes = c("repo"))

# 👉 En GitHub, copia el token (algo como: ghp_sD7xKfLZtPq6...)

# ============================================================
# 4️⃣ GUARDAR EL TOKEN EN TU SISTEMA DE R
# ============================================================

# Esto abrirá tu archivo .Renviron
usethis::edit_r_environ()

# 💡 En el archivo que se abre, pega una línea como esta (sin comillas):
# GITHUB_PAT=ghp_tuTokenLargoQueCopiasteDeGitHub
# Guarda y cierra el archivo, luego REINICIA RStudio.

# ============================================================
# 5️⃣ VERIFICAR QUE R RECONOZCA TU TOKEN
# ============================================================
Sys.getenv("GITHUB_PAT")
# ✅ Debe mostrar tu token (o al menos empezar con "ghp_").
# Si devuelve "", vuelve a editar el .Renviron y revisa la sintaxis.


# ============================================================
# 6️⃣ INICIALIZAR GIT EN TU PROYECTO LOCAL
# ============================================================

# 📁 Asegúrate de estar dentro del proyecto que deseas conectar
# Si no tienes un proyecto abierto, crea uno nuevo en RStudio (File > New Project)
usethis::use_git()

# 👉 Esto crea la carpeta .git y un archivo .gitignore
# RStudio puede pedirte reiniciar; hazlo.


# ============================================================
# 7️⃣ CREAR Y VINCULAR REPOSITORIO EN GITHUB
# ============================================================

# Esto creará automáticamente un repositorio en GitHub
# y lo enlazará con tu proyecto local
usethis::use_github()

# 🔸 Se hará un commit inicial y se subirá el código a GitHub.
# 🔸 Se abrirá el repositorio en tu navegador.


# ============================================================
# 8️⃣ VERIFICAR LA CONEXIÓN
# ============================================================

system("git remote -v")

# Debes ver algo como:
# origin  https://github.com/BioRamirez/NombreDeTuRepo.git (fetch)
# origin  https://github.com/BioRamirez/NombreDeTuRepo.git (push)


# ============================================================
# 9️⃣ SUBIR CAMBIOS FUTUROS
# ============================================================
system('git config --global --list')
user.name=BioRamirez
user.email=bioramirezjuan@gmail.com

system('git config --global --list')

# Después de modificar tu proyecto, ejecuta:
system("git add .")                         # Añade todos los archivos
system('git commit -m "Actualización de análisis"')  # Describe el cambio
system("git push")                          # Sube los cambios a GitHub

# O puedes usar el panel Git en RStudio (arriba a la derecha).


# ============================================================
# 🧠 EXTRA: VERIFICAR ESTADO DE GIT EN CUALQUIER MOMENTO
# ============================================================
system("git status")

#----------------------------------------------------Iniciar Python en R studio--------------------------

system("python --version")

#Instalar paquete para ejecutar python en R studio
#install.packages("reticulate")
library(reticulate)

py_config()

#---------------------------------------------------------------------------------------------------------



