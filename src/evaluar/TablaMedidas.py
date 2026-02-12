import os
import json
import chromadb
import time
from sentence_transformers import SentenceTransformer
import pandas as pd
from pathlib import Path

# --- 1. CONFIGURACIÓN DE RUTAS ---
# Ajustamos para que detecte las carpetas desde la ubicación del script
BASE_DIR = Path(__file__).resolve().parent.parent.parent 

# IMPORTANTE: Verifica que esta ruta a tu Golden Set sea la correcta
GOLDEN_SET_PATH = BASE_DIR / "src" / "evaluar" / "golden_set_manual.jsonl"

# Definición de las bases de datos a comparar
configs = [
    {"name": "v1 (Pequeño - 400)", "path": BASE_DIR / "chroma_db_v1"},
    {"name": "v2 (Medio - 800)", "path": BASE_DIR / "chroma_db_v2"},
    {"name": "v3 (Grande - 950)", "path": BASE_DIR / "chroma_db_v3"}
]

# --- 2. CARGA DEL MODELO ---
# Modelo multilingüe optimizado para leyes en español
MODELO_NOMBRE = "intfloat/multilingual-e5-small"

print(f"📦 Cargando modelo: {MODELO_NOMBRE}...")
# Nota: La primera vez descargará ~150MB de HuggingFace
model_emb = SentenceTransformer(MODELO_NOMBRE)

# --- 3. FUNCIÓN DE EVALUACIÓN ---
def evaluar_completo(path_db):
    if not os.path.exists(path_db):
        print(f"⚠️ Salto: No se encontró la carpeta {path_db.name}")
        return None
    
    client = chromadb.PersistentClient(path=str(path_db))
    try:
        # Asegúrate de que el nombre coincida con el usado en la ingesta
        col = client.get_collection("coleccion_leyes")
    except Exception as e:
        print(f"⚠️ Salto: No se pudo conectar a la colección en {path_db.name}")
        return None
    
    if not os.path.exists(GOLDEN_SET_PATH):
        print(f"❌ ERROR: No se encuentra el archivo {GOLDEN_SET_PATH}")
        return None

    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        casos = [json.loads(line) for line in f]
    
    hits = 0
    mrr_sum = 0
    tiempos = []
    
    for caso in casos:
        start_time = time.time()
        
        # Prefijo 'query: ' obligatorio para modelos de la familia E5
        query_con_prefijo = "query: " + caso['query']
        q_emb = model_emb.encode([query_con_prefijo]).tolist()
        
        # Buscamos el Top 5 para el Hit Rate
        res = col.query(query_embeddings=q_emb, n_results=5)
        tiempos.append(time.time() - start_time)
        
        ids_recuperados = res['ids'][0]
        relevant_ids = [str(rid) for rid in caso['relevant_ids']]
        
        acierto_encontrado = False
        for rank, rec_id in enumerate(ids_recuperados, 1):
            # Comparamos si el ID relevante es parte del ID recuperado
            if any(rid in str(rec_id) for rid in relevant_ids):
                if not acierto_encontrado:
                    hits += 1
                    acierto_encontrado = True
                mrr_sum += (1.0 / rank)
                break

    total = len(casos)
    if total == 0: return None

    return {
        "Hit Rate @5": f"{(hits / total) * 100:.1f}%",
        "MRR": round(mrr_sum / total, 3),
        "Latencia Media": f"{(sum(tiempos) / total) * 1000:.2f} ms"
    }

# --- 4. EJECUCIÓN Y TABLA ---
print("\n📊 Iniciando evaluación comparativa (RA2)...")
resultados_finales = []

for conf in configs:
    print(f"🔍 Procesando {conf['name']}...")
    metricas = evaluar_completo(conf["path"])
    
    if metricas:
        res = {"Configuración": conf["name"]}
        res.update(metricas)
        resultados_finales.append(res)

# Renderizado final
if resultados_finales:
    df = pd.DataFrame(resultados_finales)
    print("\n" + "="*75)
    print("🏆 RESULTADOS FINALES: COMPARATIVA DE ESTRATEGIAS DE CHUNKING")
    print("="*75)
    print(df.to_markdown(index=False))
    print("="*75)
    print("CONSEJO: Si el Hit Rate es muy bajo, comprueba que usaste el prefijo 'passage: ' en la ingesta.")
else:
    print("\n❌ No se pudo generar la tabla. Verifica que las rutas chroma_db_vX existan.")