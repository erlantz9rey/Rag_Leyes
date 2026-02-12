import json
import os
import re
import base64
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb

# --- 1. CONFIGURACIÓN ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_DIR = str(BASE_DIR / "chroma_db_v1")
GOLDEN_SET_PATH = BASE_DIR / "src" / "evaluar" / "golden_set_imagenes.jsonl"

# Claves API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("LLM_API_KEY")

# Inicializar Clientes
client_groq = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel('gemini-1.5-flash')

# Modelos de Recuperación
model_emb = SentenceTransformer("intfloat/multilingual-e5-small")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

# --- 2. FUNCIONES TÉCNICAS ---

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def juez_evaluar_limpio(prompt, rango=(1, 5)):
    """Juez IA blindado para devolver solo un número."""
    try:
        res = client_groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": f"Eres un juez de métricas RAG. Responde ÚNICAMENTE con un número del {rango[0]} al {rango[1]}. No escribas nada más."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        texto = res.choices[0].message.content.strip()
        # Buscamos el primer dígito que coincida con el rango
        match = re.search(r'[1-5]', texto)
        return int(match.group(0)) if match else rango[0]
    except:
        return rango[0]

# --- 3. PROCESAMIENTO RAG HÍBRIDO ---

def ejecutar_rag_hibrido(pregunta, coleccion):
    # Recuperación
    q_emb = model_emb.encode(["query: " + pregunta]).tolist()
    res = coleccion.query(query_embeddings=q_emb, n_results=10)
    
    docs = res['documents'][0]
    metadatos = res['metadatas'][0]
    ids_recuperados = res['ids'][0]
    
    # Reranking para el contexto de texto
    scores = reranker.predict([[pregunta, d] for d in docs])
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[0]
    
    contexto_texto = docs[top_idx]
    
    # Lógica de detección de imagen (buscando en los metadatos recuperados)
    path_img_detectado = None
    for meta in metadatos:
        posible_path = meta.get('imagen_path') or meta.get('source')
        if posible_path and any(posible_path.lower().endswith(ext) for ext in ['.jpg', '.png', '.jpeg']):
            path_img_detectado = posible_path
            break

    respuesta = ""
    es_multimodal = False

    # Si hay imagen, Gemini Vision toma el control
    if path_img_detectado:
        p = Path(path_img_detectado)
        if p.exists():
            es_multimodal = True
            try:
                img_data = {'mime_type': 'image/jpeg', 'data': encode_image(str(p))}
                prompt_v = f"Pregunta: {pregunta}\nAnaliza la imagen adjunta y responde con datos exactos."
                response = model_gemini.generate_content([prompt_v, img_data])
                respuesta = response.text
            except Exception as e:
                respuesta = f"Error Vision: {str(e)}"
    
    # Si no hubo imagen o falló, Groq responde por texto
    if not respuesta:
        prompt_t = f"Contexto: {contexto_texto}\nPregunta: {pregunta}\nResponde de forma concisa."
        res_g = client_groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt_t}]
        )
        respuesta = res_g.choices[0].message.content

    return respuesta, contexto_texto, es_multimodal, ids_recuperados

# --- 4. EJECUCIÓN Y TABLA ---

def main():
    client_db = chromadb.PersistentClient(path=DB_DIR)
    col = client_db.get_collection("coleccion_leyes")
    
    with open(GOLDEN_SET_PATH, 'r', encoding='utf-8') as f:
        casos = [json.loads(line) for line in f]

    reporte = []

    for caso in tqdm(casos, desc="Evaluando sistema"):
        query = caso['query']
        gt = caso.get('ground_truth', "No disponible")
        ids_esperados = caso.get('relevant_ids', [])

        ans, ctx, multi, ids_obt = ejecutar_rag_hibrido(query, col)

        # Cálculo de métricas
        fidelidad = juez_evaluar_limpio(f"Contexto: {ctx}\nRespuesta: {ans}\n¿La respuesta es fiel al contexto y no inventa nada? (1: No, 5: Sí)", (1, 5))
        exactitud = juez_evaluar_limpio(f"Respuesta Ideal: {gt}\nRespuesta IA: {ans}\n¿Qué tan exacta es la respuesta? (1: Nada, 5: Perfecta)", (1, 5))
        
        recall = sum(1 for i in ids_esperados if i in ids_obt) / len(ids_esperados) if ids_esperados else 0

        reporte.append({
            "Pregunta": query[:45] + "...",
            "Fid (1-5)": fidelidad,
            "Acc (1-5)": exactitud,
            "Recall": f"{recall*100:.0f}%",
            "Multi": "SÍ" if multi else "NO"
        })

    # Mostrar resultados
    df = pd.DataFrame(reporte)
    print("\n" + "="*80)
    print("🏆 RESULTADOS FINALES RA6 - EVALUACIÓN MULTIMODAL HÍBRIDA")
    print("="*80)
    print(df.to_markdown(index=False))
    print(f"\n📊 PROMEDIO EXACTITUD: {df['Acc (1-5)'].mean():.2f} / 5")
    print(f"📸 IMÁGENES PROCESADAS POR GEMINI: {df[df['Multi']=='SÍ'].shape[0]}")
    print("="*80)

if __name__ == "__main__":
    main()