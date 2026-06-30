import requests
import pandas as pd
import time

# --- 1. Leer tu lista ---
df = pd.read_csv("tu_lista_aves.csv", dtype=str)

# Supongamos columnas: Genero, Epiteto

# --- 2. Definir funciones para consultar API SiB (Colombia) ---
SIB_BASE = "https://api.catalogo.biodiversidad.co"
API_KEY = "TU_API_KEY"  # si se requiere

def consultar_sib(genus, species):
    params = {"genus": genus, "species": species}
    headers = {"Authorization": f"Bearer {API_KEY}"}  # si aplica
    url = f"{SIB_BASE}/species"
    resp = requests.get(url, params=params, headers=headers)
    if resp.status_code == 200:
        return resp.json()
    else:
        return None

# --- 3. (Opcional) Función para consultar otra base global --- 
# Ej: consultar en BIRDBASE o Avibase — depende si hay API pública.
# Aquí solo un esquema.

def consultar_birdbase(genus, species):
    # Pseudocódigo: reemplazar con endpoint real si existe
    url = f"https://birdbase.org/api/species/{genus}_{species}"
    resp = requests.get(url)
    if resp.status_code == 200:
        return resp.json()
    return None

# --- 4. Iterar sobre lista y guardar resultados ---
out = []
for idx, row in df.iterrows():
    gen = row["Genero"].strip()
    sp = row["Epiteto"].strip()
    rec = {"Genero": gen, "Epiteto": sp}

    sib = consultar_sib(gen, sp)
    if sib:
        # extraer campos: distribución, altitud, estatus, etc.
        rec.update({
            "Distribucion_SiB": sib.get("distribution"),
            "Altitud_min": sib.get("min_elevation"),
            "Altitud_max": sib.get("max_elevation"),
            "Otro_SiB": sib.get("other_field", None)
        })

    bird = consultar_birdbase(gen, sp)
    if bird:
        rec.update({
            "Habitat": bird.get("habitat"),
            "Migracion": bird.get("migration_status"),
            "Estado_IUCN": bird.get("iucn_status"),
            # etc.
        })

    out.append(rec)
    time.sleep(1)  # evitar saturar servidores

df_out = pd.DataFrame(out)
df_out.to_csv("aves_con_datos.csv", index=False)
print("Terminado — resultados en aves_con_datos.csv")
