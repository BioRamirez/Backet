@echo off
title ⚙️ Configuración inicial del entorno - BioRamirez
color 0A

echo ===================================================
echo     ⚙️ CONFIGURACIÓN INICIAL DEL ENTORNO PYTHON
echo ===================================================
echo.

:: 1️⃣ Verificar instalación de Git y Python
echo 🔍 Verificando instalación de Git y Python...
git --version
python --version
if errorlevel 1 (
    echo ❌ ERROR: Asegúrate de tener Python y Git instalados y en el PATH.
    pause
    exit /b
)
echo ✅ Git y Python detectados correctamente.
echo.

:: 2️⃣ Crear entorno virtual
echo 🧩 Creando entorno virtual (.venv)...
python -m venv .venv
if not exist .venv (
    echo ❌ ERROR: No se pudo crear el entorno virtual.
    pause
    exit /b
)
echo ✅ Entorno virtual creado correctamente.
echo.

:: 3️⃣ Activar entorno virtual
echo 🚀 Activando entorno virtual...
call .venv\Scripts\activate
if errorlevel 1 (
    echo ❌ ERROR: No se pudo activar el entorno virtual.
    pause
    exit /b
)
echo ✅ Entorno virtual activado.
echo.

:: 4️⃣ Actualizar pip e instalar dependencias
echo 📦 Instalando dependencias...
pip install --upgrade pip
pip install pandas matplotlib numpy seaborn jupyter
echo ✅ Dependencias instaladas correctamente.
echo.

:: 5️⃣ Configurar identidad de Git (solo una vez)
echo 🧾 Configurando identidad de Git...
git config --global user.name "BioRamirez"
git config --global user.email "bioramirezjuan@gmail.com"
echo ✅ Identidad de Git configurada.
echo.

:: 6️⃣ Inicializar repositorio Git si no existe
if not exist ".git" (
    echo 🌀 Inicializando repositorio Git local...
    git init
    echo ✅ Repositorio local creado.
) else (
    echo 🔁 Repositorio Git ya existe.
)
echo.

:: 7️⃣ Conectar con repositorio remoto
git remote remove origin >nul 2>&1
git remote add origin https://github.com/BioRamirez/Backet
echo 🔗 Repositorio remoto vinculado.
git remote -v
echo.

:: 8️⃣ Mensaje final
echo ===================================================
echo ✅ CONFIGURACIÓN INICIAL COMPLETADA CON ÉXITO ✅
echo ===================================================
echo.
pause
