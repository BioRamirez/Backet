


#pip install pdfminer.six

from docx2pdf import convert
from pdfminer.high_level import extract_text
import os

# Ruta del archivo DOCX
docx_path = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados\4_Informe_Completo_iNEXT.docx"

# Rutas destino
base_path = os.path.splitext(docx_path)[0]
pdf_path = base_path + ".pdf"
txt_path = base_path + ".txt"

# -------------------------------------------
# 1. Eliminar PDF previo si existe
# -------------------------------------------
if os.path.exists(pdf_path):
    os.remove(pdf_path)

# -------------------------------------------
# 2. Convertir DOCX → PDF
# -------------------------------------------
convert(docx_path)

# -------------------------------------------
# 3. Eliminar DOCX después de convertir
# -------------------------------------------
if os.path.exists(docx_path):
    os.remove(docx_path)

print(f"PDF generado:\n{pdf_path}")
print("DOCX eliminado.")

# -------------------------------------------
# 4. Convertir PDF → TXT
# -------------------------------------------
try:
    texto = extract_text(pdf_path)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(texto)

    print(f"TXT generado exitosamente:\n{txt_path}")

except Exception as e:
    print("Error al convertir PDF a TXT:", e)
