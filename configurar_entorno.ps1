# ===========================
# ⚙️ CONFIGURACIÓN INICIAL (SOLO UNA VEZ)
# ===========================

# 1️⃣ Verificar instalación de Git y Python
git --version
python --version

# 2️⃣ Crear entorno virtual (solo la primera vez)
python -m venv .venv

# 3️⃣ Activar entorno virtual
.venv\Scripts\activate

# 4️⃣ Instalar dependencias necesarias para el proyecto
pip install --upgrade pip
pip install pandas matplotlib numpy seaborn jupyter

# 5️⃣ Configurar identidad de Git (solo una vez)
git config --global user.name "BioRamirez"
git config --global user.email "bioramirezjuan@gmail.com"

# 6️⃣ Iniciar Git en el proyecto (solo la primera vez)
git init

# 7️⃣ Conectar con tu repositorio de GitHub (reemplaza el enlace con el tuyo)
git remote add origin https://github.com/BioRamirez/Backet.git

# 8️⃣ Primer commit y push
git add .
git commit -m "Primer commit del proyecto"
git branch -M main
git push -u origin main

Write-Host "✅ Configuración inicial completada con éxito."
