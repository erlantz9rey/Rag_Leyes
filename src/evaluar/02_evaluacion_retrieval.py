"""
================================================================================
02_evaluacion_retrieval.py - ¿EL BUSCADOR ENCUENTRA EL CHUNK CORRECTO?
================================================================================


================================================================================
¿QUÉ HACE ESTE SCRIPT?
================================================================================

    PASO 1: Generar preguntas automáticamente
    
    - Coge chunks al azar de ChromaDB
    - Usa un LLM para INVENTAR una pregunta para cada chunk
    - Ejemplo: 
        Chunk dice: "La insulina se inyecta cada 8h"
        LLM genera: "¿Cada cuánto se inyecta insulina?"
    - Guarda: {pregunta, chunk_id_correcto}
    
    PASO 2: Evaluar el buscador
    
    - Para cada pregunta:
        1. Busca en ChromaDB los 3 chunks más similares
        2. Mira si el chunk_id_correcto está en esos 3
        3. Si está = ACIERTO
        4. Si no está = FALLO
    
    RESULTADO: Métricas
    
    - Hit Rate: "En 8 de 10 búsquedas, encontró el chunk" = 80%

================================================================================
¿QUÉ HIT RATE ES BUENO?
================================================================================

    < 50%   = Problema grave. Revisar chunking o embeddings.
    50-70%  = Aceptable para RAG básico.
    70-85%  = Bueno. Añadir Reranking mejorará más.
    > 85%   = Excelente. Sistema bien afinado.


================================================================================
MÉTRICAS QUE CALCULAMOS
================================================================================

1. HIT RATE @K
   ¿En cuántas búsquedas encontró el chunk correcto? (sí/no)
   
   Fórmula: Aciertos / Total de preguntas
   
   Ejemplo con K=3 y 10 preguntas:
   - 7 veces el chunk correcto estaba en top-3
   - 3 veces NO estaba
   - Hit Rate @3 = 7/10 = 70%

2. MRR @K (Mean Reciprocal Rank)
   No solo si acertó (sí/no), sino QUÉ TAN ARRIBA quedó de los K que devolvió.
   
   Puntuación por posición:
   | Posición | Puntuación | Significado              |
   |----------|------------|--------------------------|
   | 1º       | 1/1 = 1.0  | Perfecto                 |
   | 2º       | 1/2 = 0.5  | Bien, pero no primero    |
   | 3º       | 1/3 = 0.33 | Lo encontró, pero tarde  |
   | No está  | 0          | Falló                    |
   
   Ejemplo con 4 preguntas:
   | Pregunta | ¿Encontró? | Posición | Puntuación |
   |----------|------------|----------|------------|
   | Q1       | Sí         | 1º       | 1.0        |
   | Q2       | Sí         | 3º       | 0.33       |
   | Q3       | No         | -        | 0          |
   | Q4       | Sí         | 2º       | 0.5        |
   
   MRR = (1.0 + 0.33 + 0 + 0.5) / 4 = 0.46

RESUMEN:
   - Hit Rate: "¿Acertó?" (sí/no)
   - MRR: "¿En qué posición acertó?" (premia estar arriba)
   - Ambas usan el mismo K (top_k=3 por defecto)


En ambos casos, cuanto más alto mejor.


Si el documento correcto está en posición 5º pero solo recuperamos 3 (top_k=3):

Hit Rate: 0 (no lo encontró)
MRR: 0 (no cuenta porque está fuera del rango)

================================================================================
USO:
    cd src/evaluacion_rag
    python 02_evaluacion_retrieval.py
================================================================================
"""

import os
import sys
import random
import json
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI  # SDK compatible con Groq y OpenRouter
import logging
from dotenv import load_dotenv
import time

# Añadir la carpeta padre (src/) al path para encontrar utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

# Configuración - Cargar .env desde la raíz del proyecto
ENV_PATH = utils.project_root() / ".env"
load_dotenv(dotenv_path=ENV_PATH)
utils.setup_logging()
logger = logging.getLogger("eval_auto")

# RUTAS
from pathlib import Path

# Base de datos ChromaDB - siempre en la raíz del proyecto
DB_DIR = str(utils.project_root() / "chroma_db_v2")

COLLECTION_NAME = "coleccion_leyes" # nombre de la coleccion creada en el script 12
GOLDEN_SET_FILE = str(utils.project_root() / "src" / "golden_set_automatico.jsonl")

logger.info(f" DBDIR... {DB_DIR}")
logger.info(f" COLLECTION_NAME... {COLLECTION_NAME}")
logger.info(f" GOLDEN_SET_FILE... {GOLDEN_SET_FILE}")    

# ============================================================================
# CONFIGURACION DEL LLM (Unificada)
# ============================================================================
LLM_API_KEY = os.getenv("GROQ_API_KEY")
LLM_BASE_URL = "https://api.groq.com/openai/v1"
MODELO_LLM = "llama-3.1-8b-instant"  

MODELO_EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2"

# ============================================================================
# 1. GENERADOR DE PREGUNTAS SINTÉTICAS
# ============================================================================
def generar_pregunta_para_chunk(texto_chunk, metadata, client_llm):
    """Usa el LLM para inventar una pregunta basada en el texto y su contexto (metadata)."""
    
    # ------------------------------------------------------------------------
    #  SYSTEM PROMPT Y TEMPERATURA
    # ------------------------------------------------------------------------
    # 1. TEMPERATURA (0.7): Para generar datos creativos (inventar preguntas),
    #    necesitamos variedad. 
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------

    contexto_extra = f"Categoría: {metadata.get('category', 'General')}"
    if 'subcategory' in metadata:
        contexto_extra += f", Subcategoría: {metadata['subcategory']}"

    system_content = f"""Eres un profesor experto creando preguntas de examen sobre {contexto_extra}.

INSTRUCCIONES:
1. Lee el texto y formula UNA pregunta clara y específica.
2. La pregunta debe ser NATURAL, como si un estudiante la hiciera.
3. NO uses frases como "según el texto", "del documento", "proporcionado".
4. Devuelve SOLO la pregunta, sin explicaciones.

EJEMPLOS DE PREGUNTAS BUENAS:
- "¿Cuáles son los síntomas de la hipoglucemia?"
- "¿Cada cuánto se debe administrar la insulina?"
- "¿Qué diferencia hay entre diabetes tipo 1 y tipo 2?"

EJEMPLOS DE PREGUNTAS MALAS (NO HACER):
- "¿Qué dice el texto sobre la diabetes?" 
- "Según el documento, ¿cuáles son los síntomas?" """

    user_content = f"""TEXTO:
{texto_chunk[:1500]}

PREGUNTA:"""

    logger.info(f" User Prompt: {user_content}")

    try:
        if not LLM_API_KEY or LLM_API_KEY == "ollama":
            # Si estamos en modo Ollama local (sin key) o ha fallado la key, seguimos.
            # Pero logueamos aviso si es critico.
            pass

        response = client_llm.chat.completions.create(
            model=MODELO_LLM,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            temperature=0.7, # Creatividad alta para variar preguntas
            max_tokens=150
        )

        logger.info(f" Pregunta generada: {response.choices[0].message.content.strip().replace('"', '')}")

        # message.content  - salida del modelo  
        # cogemos la primera respuesta que ha dado el modelo
        
         

        return response.choices[0].message.content.strip().replace('"', '')
    except Exception as e:
        logger.error(f"Error generando pregunta: {e}")
        return None

def crear_golden_set_automatico(collection, client_llm, num_preguntas=5):
    """Crea un dataset de evaluación automáticamente (Formato Estándar)."""
    logger.info(f" Generando Golden Set Automático ({num_preguntas} preguntas)...")
    
    # Obtener todos los IDs de la colección
    all_data = collection.get()
    all_ids = all_data['ids']
    all_docs = all_data['documents']
    all_metas = all_data['metadatas'] # <--- Recuperar metadatos
    
    if len(all_ids) == 0:
        logger.error(" La base de datos está vacía. Ejecuta primero script 12.")
        return []

    # Seleccionar chunks aleatorios de la colección
    # Esos son los índices de los chunks que usará para generar preguntas.
    # Elige N números AL AZAR sin repetir.
    indices = random.sample(range(len(all_ids)), min(num_preguntas, len(all_ids)))
    
    golden_set = []
    
    for i, idx in enumerate(indices):
        chunk_id = all_ids[idx]
        texto = all_docs[idx]
        meta = all_metas[idx] # Metadatos del chunk
        
        # Generar pregunta para el chunk seleccionado
        pregunta = generar_pregunta_para_chunk(texto, meta, client_llm)
        
        if pregunta:
            logger.info(f" [{i+1}/{num_preguntas}] Generada: {pregunta[:60]}...")
            
            # FORMATO ESTÁNDAR DE INDUSTRIA (RAGAS / TRULENS)
            entry = {
                "id": f"q_{i}",
                "query": pregunta,
                "relevant_ids": [chunk_id], # Lista de IDs correctos
                "texto_original": texto[:100], 
                "metadata": {   # <--- NUEVO: Guardamos contexto útil
                    "source": meta.get('source', 'unknown'),
                    "category": meta.get('category', 'General'),
                    "subcategory": meta.get('subcategory', '')
                }
            }
            golden_set.append(entry)
            
            time.sleep(10) # Pausa de 5s para evitar Rate Limit (429) en modelos gratuitos o cambiar de modelo 
            
    # Guardar en archivo
    with open(GOLDEN_SET_FILE, 'w', encoding='utf-8') as f:
        for entry in golden_set:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    logger.info(f" Golden Set guardado en: {GOLDEN_SET_FILE}")
    return golden_set

# ============================================================================
# 2. EVALUADOR (Retrieval Metrics)
# ============================================================================
def evaluar_retrieval(collection, model_emb, golden_set, top_k=3):
    """
    Evalúa la calidad del módulo de recuperación.
    
    Args:
        collection: Colección de ChromaDB con los documentos indexados
        model_emb: Modelo de embeddings (SentenceTransformer)
        golden_set: Lista de casos de prueba {query, relevant_ids}
        top_k (int): Número de documentos a recuperar por consulta.
                     Default=3. Rango típico: 3-10.
                     
                     ¿Por qué 3?
                     - Suficiente para la mayoría de consultas simples
    
    Returns:
        tuple: (hit_rate, mrr) métricas de evaluación
    """

    logger.info(f"\n Iniciando Evaluación (Top-{top_k})...")
    
    aciertos = 0
    mrr_sum = 0
    
    # Recorrer cada pregunta del golden set
    for i, item in enumerate(golden_set):
        pregunta = item['query']             # Estándar: 'query'
        target_ids = item['relevant_ids']    # Estándar: Lista de IDs
        
        # 1. Buscar en BD
        query_emb = utils.generar_embeddings(model_emb, [pregunta])

        # le preguntamso a ChromaDB que nos devuelva los top_k chunks más similares
        results = collection.query(
            query_embeddings=query_emb,
            n_results=top_k
        )
        
        recuperados_ids = results['ids'][0]
        
        # 2. Comprobar si acertamos (Si ALGUNO de los targets está en los recuperados)
        # Intersección de listas > 0

        # Comprueba si la respuesta correcta está dentro de los 3 que trajo el buscador.
        acierto = any(tid in recuperados_ids for tid in target_ids)
        
        if acierto:
            aciertos += 1
            # Para MRR, buscamos la posición del PRIMER acierto.  Si acertamos, miramos en qué puesto quedó.
            # Si quedó el 1º -> Sumamos 1/1 = 1.0 puntos.
            # Si quedó el 2º -> Sumamos 1/2 = 0.5 puntos.
            # Premia acertar Arriba del todo. Acertar el 3º es bueno, pero acertar el 1º es mejor.
            for rank, rid in enumerate(recuperados_ids):
                if rid in target_ids:
                    mrr_sum += 1.0 / (rank + 1)
                    logger.info(f"    Acierto (Pos {rank+1}): {pregunta[:50]}...")
                    break
        else:
            logger.info(f"    Fallo: {pregunta[:50]}...")
            
    # 3. Calcular Métricas Finales
    #	total: Número de preguntas
    #	aciertos: Cuántas veces encontró el chunk correcto
    #	mrr_sum:	Suma de 1/posición de cada acierto
    
    total = len(golden_set)
    hit_rate = aciertos / total if total > 0 else 0
    mrr = mrr_sum / total if total > 0 else 0
    
    return hit_rate, mrr

# ============================================================================
# MAIN
# ============================================================================
def main():
    logger.info("="*80)
    logger.info(" EVALUACIÓN AUTOMÁTICA DE RAG (PDFs)")
    logger.info("="*80)

    # 1. Conectar a BD
    if not os.path.exists(DB_DIR):
        logger.error(f" No existe BD en {DB_DIR}")
        return

    client_db = chromadb.PersistentClient(path=DB_DIR)
    collection = client_db.get_collection(COLLECTION_NAME)
    
    # 2. Conectar a LLM  y Embeddings
    client_llm = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    model_emb = SentenceTransformer(MODELO_EMBEDDINGS)

    # 3. ¿Generar Golden Set Nuevo o Usar Existente?
    golden_set = []
    if os.path.exists(GOLDEN_SET_FILE):
        opcion = input(" Existe un Golden Set anterior. ¿Usarlo (s) o generar nuevo (n)? [s/n]: ")
        if opcion.lower() == 's':
            with open(GOLDEN_SET_FILE, 'r', encoding='utf-8') as f:
                golden_set = [json.loads(line) for line in f]
            logger.info(f"  Leídas {len(golden_set)} preguntas.")
    
    if not golden_set:
        num = int(input(" ¿Cuántas preguntas generar para la prueba? (Rec: 5-10): "))
        golden_set = crear_golden_set_automatico(collection, client_llm, num_preguntas=num)

    # 4. Ejecutar Evaluación
    if golden_set:

        logger.info("\n" + "="*60)
        logger.info("  EVALUANDO RETRIEVAL")
        logger.info("="*60)
        # Hit Rate @ 3  top_k=3  -> devuelve lso 3 doumentos más parecidos
        # Número máximo de documentos (chunks) que el sistema de recuperación devuelve, ordenados por similitud semántica descendente.
        hit_rate, mrr = evaluar_retrieval(collection, model_emb, golden_set, top_k=3)
        
        logger.info("\n" + "="*60)
        logger.info("  RESULTADOS DEL EXAMEN")
        logger.info("="*60)
        logger.info(f"  Preguntas Evaluadas : {len(golden_set)}")
        logger.info(f"  Hit Rate @ 3        : {hit_rate*100:.1f}%  (Aciertos totales)")
        logger.info(f"  MRR Score           : {mrr:.3f}   (Calidad del ranking, 0 a 1)")
        logger.info("="*60)
        
        if hit_rate < 0.5:
            logger.info("  CONSEJO: Tu Hit Rate es bajo. Prueba a mejorar el chunking (Script 12)")
            logger.info("             o usar búsqueda híbrida (Script 09).")
        else:
            logger.info("  RESULTADO: El sistema funciona bien")
        logger.info("="*60)

if __name__ == "__main__":
    main()
