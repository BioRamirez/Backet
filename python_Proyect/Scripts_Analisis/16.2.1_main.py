import subprocess
import os

def preguntar_ollama(prompt, modelo="llama3.1"):
    # Ruta fija donde está instalado Ollama en Windows
    ruta_ollama = r"C:\Users\Ramirez Juan\AppData\Local\Programs\Ollama\ollama.exe"

    if not os.path.exists(ruta_ollama):
        raise FileNotFoundError("❌ No se encontró ollama.exe en la ruta esperada.")

    comando = [ruta_ollama, "run", modelo, prompt]

    resultado = subprocess.run(comando, capture_output=True, text=True)

    return resultado.stdout.strip()


if __name__ == "__main__":
    print(preguntar_ollama("Hola, ¿estás funcionando?"))
