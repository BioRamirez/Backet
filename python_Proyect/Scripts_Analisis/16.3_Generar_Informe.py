import subprocess
from pypdf import PdfReader
import textwrap
import os
import shutil
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import cm

# -----------------------------------------
# LOCALIZAR OLLAMA
# -----------------------------------------
def obtener_ruta_ollama():
    ruta = shutil.which("ollama")
    if ruta:
        return ruta

    ruta_fija = r"C:\Users\Ramirez Juan\AppData\Local\Programs\Ollama\ollama.exe"
    if os.path.exists(ruta_fija):
        return ruta_fija

    raise FileNotFoundError("❌ No se encontró Ollama en el sistema.")


# -----------------------------------------
# CONSULTA A OLLAMA
# -----------------------------------------
def preguntar_ollama(prompt, modelo="llama3.1"):
    ruta = obtener_ruta_ollama()

    proceso = subprocess.run(
        [ruta, "run", modelo, prompt],
        capture_output=True,
        text=True
    )

    if proceso.returncode != 0:
        print("❌ Error con Ollama:")
        print(proceso.stderr)
        return "[Error procesando texto]"

    return proceso.stdout.strip()


# -----------------------------------------
# LEER PDF Y DIVIDIRLO EN BLOQUES
# -----------------------------------------
def leer_pdf_en_bloques(pdf_path, max_chars=3000):
    reader = PdfReader(pdf_path)
    texto = ""

    for page in reader.pages:
        texto += page.extract_text() + "\n"

    bloques = textwrap.wrap(texto, max_chars)
    return bloques


# -----------------------------------------
# PROCESAR PDF Y EXPORTAR A PDF
# -----------------------------------------
def mejorar_pdf(pdf_path, salida_pdf):
    print("📄 Leyendo PDF...")
    bloques = leer_pdf_en_bloques(pdf_path)

    styles = getSampleStyleSheet()
    story = []

    # Título del documento
    story.append(Paragraph("Informe Mejorado por IA", styles["Title"]))
    story.append(Spacer(1, 0.5 * cm))

    for i, bloque in enumerate(bloques, 1):
        print(f"Procesando bloque {i}/{len(bloques)}...")

        prompt = f"""
Eres un experto en redacción técnica ambiental.
Mejora el siguiente texto sin inventar datos, manteniendo el tono técnico.

--- TEXTO ORIGINAL ---
{bloque}

--- INSTRUCCIONES ---
• Claridad y coherencia
• Redacción técnica ambiental
• No inventar información
• No hacer conclusiones nuevas

Texto mejorado:
"""

        mejorado = preguntar_ollama(prompt)

        # Agregar al PDF
        story.append(Paragraph(f"<b>Sección {i}</b>", styles["Heading2"]))
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph(mejorado.replace("\n", "<br/>"), styles["BodyText"]))
        story.append(Spacer(1, 0.5 * cm))

    print("📝 Generando PDF final...")
    doc = SimpleDocTemplate(salida_pdf, pagesize=letter)
    doc.build(story)

    print(f"\n✅ PDF generado con éxito: {salida_pdf}")


# -----------------------------------------
# EJECUCIÓN PRINCIPAL
# -----------------------------------------
if __name__ == "__main__":
    ruta_pdf = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados_Final\Informe_Biodiversidad_Fauna.pdf"
    salida = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados_Final\Informe_Biodiversidad_Fauna_IA.pdf"

    mejorar_pdf(ruta_pdf, salida)












