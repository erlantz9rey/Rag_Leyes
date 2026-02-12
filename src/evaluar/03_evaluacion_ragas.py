"""
================================================================================
16_evaluacion_ragas.py - EL JUEZ IA (LLM-as-a-Judge)
================================================================================
Ya tengo el RAG terminado, y sirve para la evaluación final del proyecto

OBJETIVO:
    Evaluar la CALIDAD de las respuestas del RAG usando otro LLM como Juez.
    
    IMPORTANTE: Este script implementa el CONCEPTO de "RAGAS: Automated Evaluation of RAG Systems" de forma manual
    (RAGAS-Inspired o "RAGAS Lite"), pero NO usa la librería oficial 'ragas'.
    
   El objetivo es entender los fundamentos de evaluación de sistemas RAG antes de usar frameworks más complejos.
    
    Para usar RAGAS oficial: pip install ragas (ver documentación al final)


"""

"""

================================================================================
LAS 4 METRICAS QUE EVALUAMOS
================================================================================

A) FIDELIDAD (Faithfulness) - Puntuacion: 0 o 1
   Objetivo (¿Qué se pregunta el Juez?): Medir si lo que escribio la IA esta respaldado por los documentos.
   (NOTA: Aqui NO importa la pregunta del usuario, solo si la IA dice la verdad segun el texto).
   
   Detecta ALUCINACIONES (inventarse datos).
   - Documentos dicen: "La aspirina alivia el dolor"
   - La IA escribe: "La aspirina cura el cancer" -> FIDELIDAD = 0 (Alucinacion!) 
   - La IA escribe: "La aspirina reduce el dolor" -> FIDELIDAD = 1 (Correcto)

B) RELEVANCIA (Answer Relevance) - Puntuacion: 1 a 5
   Objetivo (¿Qué se pregunta el Juez?): Medir si la respuesta de la IA contesta realmente a lo que pregunto el usuario.
   
   Detecta respuestas que se van por las ramas.
   - Usuario pregunta: "Que hora es?"
   - La IA responde: "Hoy hace sol" -> RELEVANCIA = 1 (Irrelevante!)
   - La IA responde: "Son las 3pm" -> RELEVANCIA = 5 (Perfecto)

C) EXACTITUD (Answer Correctness) - Puntuacion: 1 a 5 [NUEVA]
   Objetivo: Medir si la respuesta coincide con la respuesta IDEAL (Ground Truth) del humano.
   
   - Ground Truth: "El ibuprofeno se toma cada 8 horas"
   - IA dice: "Tomar ibuprofeno cada 8h" -> EXACTITUD = 5 (Semanticamente igual)
   - IA dice: "Consulte a su medico" -> EXACTITUD = 1 (No contesta)

D) CONTEXT RECALL - Puntuacion: 0% a 100% [NUEVA]
   Objetivo: Medir si el sistema ENCONTRO el documento correcto (Chunk ID).
   
   - Tu marcaste: "La respuesta esta en chunk_12"
   - El sistema trajo: [chunk_12, chunk_5, chunk_8] -> RECALL = 100% (Lo encontro!)
   - El sistema trajo: [chunk_1, chunk_2, chunk_3] -> RECALL = 0% (Fallo!)

================================================================================
RESUMEN VISUAL
================================================================================

| Metrica        | Que mide?                    | Rango    | Ideal |
|----------------|------------------------------|----------|-------|
| Fidelidad      | ¿Se inventó datos? (Alucina  | 0 o 1    | 1     |
| Relevancia     | Responde a la pregunta?      | 1-5      | 5     |
| Exactitud      | Coincide con Ground Truth?   | 1-5      | 5     |
| Context Recall | Encontro el chunk correcto?  | 0%-100%  | 100%  |

================================================================================
REFERENCIAS
================================================================================
- Paper RAGAS: https://arxiv.org/abs/2309.15217
- Libreria oficial: https://github.com/explodinggradients/ragas
- Documentacion: https://docs.ragas.io/
"""

import json
import os
import sys
import logging
import time # Para posibles reintentos o medición
import re   # <--- Importante para parsear respuestas del LLM Juez
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

import csv  # Guardar reporte
from tqdm import tqdm  # Barra de progreso

# Añadir la carpeta padre (src/) al path para encontrar utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

# Configuración - Cargar .env desde la raíz del proyecto
ENV_PATH = utils.project_root() / ".env"
load_dotenv(dotenv_path=ENV_PATH)
utils.setup_logging()
logger = logging.getLogger("ragas_lite")

# RUTAS
from pathlib import Path

# Base de datos ChromaDB - siempre en la raíz del proyecto
DB_DIR = str(utils.project_root() / "chroma_db")

COLLECTION_NAME = "coleccion_leyes"
GOLDEN_SET_PATH = str(utils.project_root() / "src" / "evaluar" / "golden_set_manual.jsonl")


# ============================================================================
# CONFIGURACION DEL LLM (Unificada)
# ============================================================================
LLM_API_KEY = os.getenv("GROQ_API_KEY") 
LLM_BASE_URL = "https://api.groq.com/openai/v1"
MODELO_LLM = "llama-3.1-8b-instant"

# ============================================================================
MODELO_EMBEDDINGS = os.getenv("MODELO_EMBEDDINGS", "Qwen/Qwen3-Embedding-0.6B")
MODELO_RERANKER = os.getenv("MODELO_RERANKER", "BAAI/bge-reranker-v2-m3")
# ============================================================================
# 1. SETUP (Copiado del Script 14 para tener el RAG funcional)
# ============================================================================
def iniciar_sistema():
    logger.info("  Iniciando Juez RAGAS...")
    
    # 1. DB
    if not os.path.exists(DB_DIR):
        logger.error("No hay base de datos Chroma.")
        return None, None, None, None
    client_db = chromadb.PersistentClient(path=DB_DIR)
    col = client_db.get_collection(COLLECTION_NAME)
    
    # 2. Modelos
    model_emb = SentenceTransformer(MODELO_EMBEDDINGS)
    reranker = CrossEncoder(MODELO_RERANKER)
    
    # 3. Cliente LLM (Unificado)
    client_llm = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    
    return col, model_emb, reranker, client_llm

# ============================================================================
# 2. FUNCIONES RAG (GENERACIÓN)
# ============================================================================
def generar_respuesta_rag(pregunta, col, model_emb, reranker, client_llm):
    """
    Ejecuta el ciclo de vida RAG completo: Retrieve -> Rerank -> Generate
    
    FLUJO:
    1. RETRIEVE: Buscar 10 chunks similares a la pregunta (Bi-Encoder)
    2. RERANK: Reordenar esos 10 con un modelo más preciso (Cross-Encoder)
    3. GENERATE: Usar los top-3 como contexto para el LLM
    
    Args:
        pregunta: La pregunta del usuario
        col: Colección de ChromaDB
        model_emb: Modelo de embeddings (Bi-Encoder)
        reranker: Modelo Cross-Encoder para reordenar
        client_llm: Cliente de API para generación
    
    Returns:
        tuple: (respuesta_texto, lista_de_chunks_usados)
    """
    
    # =========================================================================
    # PASO 1: RETRIEVE (Búsqueda inicial con Bi-Encoder)
    # =========================================================================
    # Buscamos 10 candidatos porque el Bi-Encoder es rápido pero menos preciso
    q_emb = utils.generar_embeddings(model_emb, [pregunta])
    res = col.query(query_embeddings=q_emb, n_results=10)
    docs = res['documents'][0]
    
    if not docs: 
        return "", []

    # =========================================================================
    # PASO 2: RERANK (Reordenar con Cross-Encoder)
    # =========================================================================
    # El Cross-Encoder es más lento pero más preciso
    # Compara directamente pregunta-documento (no usa embeddings)
    pares = [[pregunta, doc] for doc in docs]
    scores = reranker.predict(pares)
    
    # Ordenar por score descendente y quedarnos con los 3 mejores
    top_3 = [doc for score, doc in sorted(zip(scores, docs), reverse=True)[:3]]
    
    # =========================================================================
    # PASO 3: GENERATE (Generar respuesta con LLM)
    # =========================================================================
    contexto_str = "\n---\n".join(top_3)
    
    # Prompt estructurado para el LLM
    prompt = f"""Eres un asistente experto. Responde SOLO con información del contexto.

CONTEXTO:
{contexto_str}

PREGUNTA: {pregunta}

INSTRUCCIONES:
- Si la respuesta está en el contexto, responde de forma clara y directa.
- Si NO encuentras la respuesta en el contexto, di exactamente: "No tengo información sobre eso."
- NO inventes información que no esté en el contexto.

RESPUESTA:"""
    
    resp = client_llm.chat.completions.create(
        model=MODELO_LLM,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1  # Baja temperatura para respuestas consistentes
    )
    return resp.choices[0].message.content, top_3

# ============================================================================
# 3. EL JUEZ (EVALUACIÓN) - LLM-as-a-Judge
# ============================================================================
# 
# CONCEPTO: Usamos un LLM para evaluar las respuestas de otro LLM.
# Es como tener un "profesor" que corrige los "exámenes" del RAG.
#
# ============================================================================

def evaluar_fidelidad(pregunta, respuesta, contexto, client_llm):
    """
    JUEZ DE FIDELIDAD (Faithfulness) - Detector de Alucinaciones
    
    ¿QUÉ MIDE?
    ----------
    Si la respuesta del RAG se INVENTA información que NO está en el contexto.
    
    EJEMPLO:
    - Contexto: "La aspirina alivia el dolor de cabeza"
    - Respuesta: "La aspirina cura el cáncer" → FIDELIDAD = 0 (¡Alucinación!)
    - Respuesta: "La aspirina ayuda con el dolor" → FIDELIDAD = 1 (Correcto)
    
    ¿POR QUÉ ES IMPORTANTE?
    -----------------------
    Un RAG que alucina es PELIGROSO. Puede dar información falsa con
    total confianza. Esta métrica detecta ese problema.
    
    Args:
        pregunta: La pregunta original (no se usa en esta evaluación)
        respuesta: Lo que respondió el RAG  
        contexto: Los chunks que se usaron para responder
        client_llm: Cliente de API para el LLM Juez
    
    Returns:
        int: 0 = Alucinación detectada, 1 = Respuesta fiel al contexto
    """
    
    # =========================================================================
    # PROMPT DEL JUEZ
    # =========================================================================
    # Diseñado para que el LLM sea estricto y solo responda 0 o 1
    
    prompt_juez = f"""Eres un JUEZ DE VERIFICACIÓN DE HECHOS.

Tu tarea es verificar si la RESPUESTA contiene SOLO información del CONTEXTO.

CONTEXTO (fuente de verdad):
{contexto}

RESPUESTA A EVALUAR:
{respuesta}

INSTRUCCIONES:
1. Lee el contexto cuidadosamente.
2. Compara cada afirmación de la respuesta con el contexto.
3. Si la respuesta dice algo que NO está en el contexto → Es una ALUCINACIÓN.

EJEMPLOS:
- Contexto: "El paracetamol reduce la fiebre"
  Respuesta: "El paracetamol baja la temperatura" → 1 (correcto, mismo significado)
  
- Contexto: "El paracetamol reduce la fiebre"
  Respuesta: "El paracetamol cura infecciones" → 0 (alucinación, no dice eso)

VEREDICTO (responde SOLO con el número):
- 0 = Contiene información inventada (alucinación)
- 1 = Todo está respaldado por el contexto"""

    try:
        veredicto = client_llm.chat.completions.create(
            model=MODELO_LLM,
            messages=[{"role": "user", "content": prompt_juez}],
            temperature=0,  # Determinista . las evaluaciones deben ser reproducibles.
            max_tokens=5  # Limitar para evitar explicaciones
        )
        texto = veredicto.choices[0].message.content.strip()

         # Parsing robusto: buscar SOLO el primer dígito 0 o 1
        match = re.search(r'\b([01])\b', texto)
        if match:

            #Extrae el dígito encontrado (como string: "0" o "1")
            return int(match.group(1))

         # Fallback: si encuentra "1" aislado
        if texto == "1":
            return 1
        elif texto == "0":
            return 0
        
          # Si el LLM no coopera, asumir alucinación (conservador)
        logger.warning(f"Respuesta ambigua del juez: '{texto}'. Asumiendo 0.")
        return 0
    except Exception as e:
        logger.error(f"Error en evaluación de fidelidad: {e}")
        return 0





def evaluar_relevancia(pregunta, respuesta, client_llm):
    """
    JUEZ DE RELEVANCIA (Answer Relevance)
    
    ¿QUÉ MIDE?
    ----------
    Si la respuesta CONTESTA REALMENTE a lo que preguntó el usuario.
    Detecta respuestas que "se van por las ramas".
    
    EJEMPLO:
    - Pregunta: "¿Qué hora es?"
    - Respuesta: "Hoy hace sol" → RELEVANCIA = 1 (Irrelevante!)
    - Respuesta: "Son las 3 de la tarde" → RELEVANCIA = 5 (Perfecta)
    
    ¿POR QUÉ ES IMPORTANTE?
    -----------------------
    Un RAG puede ser FIEL al contexto pero NO responder la pregunta.
    Esta métrica detecta ese problema.
    
    Args:
        pregunta: Lo que preguntó el usuario
        respuesta: Lo que respondió el RAG
        client_llm: Cliente de API para el LLM Juez
    
    Returns:
        int: Puntuación del 1 (irrelevante) al 5 (perfecta)
    """
    
    prompt_juez = f"""Eres un JUEZ DE RELEVANCIA.

Tu tarea es evaluar si la RESPUESTA contesta a la PREGUNTA del usuario.

PREGUNTA DEL USUARIO:
{pregunta}

RESPUESTA DEL SISTEMA:
{respuesta}

ESCALA DE EVALUACIÓN:
| Puntuación | Significado                              |
|------------|------------------------------------------|
| 1          | No responde a la pregunta / "No lo sé"   |
| 2          | Vagamente relacionada pero inútil        |
| 3          | Respuesta parcial, falta información     |
| 4          | Buena respuesta, pequeñas mejoras posibles|
| 5          | Perfecta: completa, precisa y directa    |

VEREDICTO (responde SOLO con el número del 1 al 5):"""

    try:
        veredicto = client_llm.chat.completions.create(
            model=MODELO_LLM,
            messages=[{"role": "user", "content": prompt_juez}],
            temperature=0,
            max_tokens=5
        )
        texto = veredicto.choices[0].message.content.strip()

        # Buscar primer dígito VÁLIDO (1-5)
        for char in texto:
            if char.isdigit():
                num = int(char)
                if 1 <= num <= 5:
                    return num
        logger.warning(f"  Respuesta inválida del juez: '{texto}'. Asumiendo 1.")
        return 1
    except Exception as e:
        logger.error(f" Error en evaluacion: {e}")
        return 1


def evaluar_correctness(respuesta_rag, ground_truth, client_llm):
    """
    JUEZ DE EXACTITUD (Answer Correctness)
    
    ¿QUÉ MIDE?
    ----------
    Si la respuesta del RAG COINCIDE con la respuesta IDEAL que escribió un humano.
    
    EJEMPLO:
    - Ground Truth: "El ibuprofeno se toma cada 8 horas"
    - Respuesta: "Tomar ibuprofeno cada 8h" → EXACTITUD = 5 (Semánticamente igual)
    - Respuesta: "Consulte a su médico" → EXACTITUD = 1 (No contesta)
    
    ¿POR QUÉ ES IMPORTANTE?
    -----------------------
    Esta es LA MÉTRICA MÁS IMPORTANTE cuando tienes Ground Truth manual.
    Compara directamente con la "respuesta correcta" del examen.
    
    Args:
        respuesta_rag: Lo que respondió el RAG
        ground_truth: La respuesta ideal que escribió el humano
        client_llm: Cliente de API para el LLM Juez
    
    Returns:
        int: Puntuación del 1 (incorrecta) al 5 (perfecta)
    """
    
    prompt_juez = f"""Eres un JUEZ DE EXACTITUD.

Tu tarea es comparar si la RESPUESTA DE LA IA dice lo mismo que la RESPUESTA IDEAL.

RESPUESTA IDEAL (escrita por un humano):
{ground_truth}

RESPUESTA DE LA IA (a evaluar):
{respuesta_rag}

ESCALA DE EVALUACIÓN:
| Puntuación | Significado                              |
|------------|------------------------------------------|
| 1          | Completamente diferente o incorrecta     |
| 2          | Vagamente relacionada                    |
| 3          | Parcialmente correcta                    |
| 4          | Muy similar, pequeñas diferencias        |
| 5          | Semánticamente idéntica                  |

NOTA: No importa si usan palabras diferentes. Lo importante es si DICEN LO MISMO.

VEREDICTO (responde SOLO con el número del 1 al 5):"""

    try:
        veredicto = client_llm.chat.completions.create(
            model=MODELO_LLM,
            messages=[{"role": "user", "content": prompt_juez}],
            temperature=0,
            max_tokens=5
        )
        texto = veredicto.choices[0].message.content.strip()
        for char in texto:
            if char.isdigit():
                num = int(char)
                if 1 <= num <= 5:
                    return num
        return 1
    except Exception as e:
        logger.error(f"Error en evaluacion correctness: {e}")
        return 1


def evaluar_context_recall(retrieved_ids, relevant_ids):
    """
    CONTEXT RECALL:
    Mide si el sistema encontro los chunks correctos.
    
    Es simplemente: cuantos de los IDs relevantes estan en los recuperados.
    """
    if not relevant_ids:
        return 0.0
    
    encontrados = sum(1 for rid in relevant_ids if rid in retrieved_ids)
    return encontrados / len(relevant_ids)


# ============================================================================
# MAIN
# ============================================================================
def main():
    col, model_emb, reranker, client_llm = iniciar_sistema()
    if not col: return

    # 1. Cargar Golden Set MANUAL
    if not os.path.exists(GOLDEN_SET_PATH):
        logger.error(f"Falta el Golden Set: {GOLDEN_SET_PATH}")
        logger.info("Ejecuta crear_ground_truth.py primero.")
        return
        
    logger.info("Cargando Golden Set Manual...")
    casos_test = []
    with open(GOLDEN_SET_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            casos_test.append(json.loads(line))
    
    logger.info(f"Iniciando evaluacion de {len(casos_test)} casos de prueba.\n")

    # 2. Evaluar cada caso
    # =========================================================================
    # CONTADORES DE MÉTRICAS
    # =========================================================================
    # Cada métrica responde a una pregunta diferente sobre la calidad del RAG:
    
    score_fidelidad = 0    # ¿Se inventó datos? (0=alucinó, 1=fiel)
    score_relevancia = 0   # ¿Responde a la pregunta? (1-5)
    score_correctness = 0  # ¿Coincide con la respuesta ideal? (1-5)
    score_recall = 0       # ¿Encontró el chunk correcto? (0-100%)

    print(f"{'PREGUNTA':<35} | {'FID':<4} | {'REL':<4} | {'CORR':<4} | {'RECALL':<6}")
    print("-" * 75)

    resultados_detallados = []
    
    # Barra de progreso con tqdm - 	Lista de preguntas del Golden Set
    for caso in tqdm(casos_test, desc="Evaluando", unit="caso"):
        pregunta = caso['query']
        ground_truth = caso.get('ground_truth', '')
        relevant_ids = caso.get('relevant_ids', [])
        
        # A) EL SISTEMA RAG RESPONDE
        respuesta_rag, docs_contexto = generar_respuesta_rag(
            pregunta, col, model_emb, reranker, client_llm
        )
        ctx_str = "\n".join(docs_contexto)
        
        # Obtener IDs de documentos recuperados
        q_emb = model_emb.encode([pregunta]).tolist()
        res = col.query(query_embeddings=q_emb, n_results=5)
        retrieved_ids = res['ids'][0] if res['ids'] else []

        # B) EL JUEZ EVALUA
        fid = evaluar_fidelidad(pregunta, respuesta_rag, ctx_str, client_llm)
        rel = evaluar_relevancia(pregunta, respuesta_rag, client_llm)
        corr = evaluar_correctness(respuesta_rag, ground_truth, client_llm) if ground_truth else 0
        recall = evaluar_context_recall(retrieved_ids, relevant_ids)

        score_fidelidad += fid
        score_relevancia += rel
        score_correctness += corr
        score_recall += recall
        
        # Guardar detalle
        resultados_detallados.append({
            "pregunta": pregunta,
            "fidelidad": fid,
            "relevancia": rel,
            "exactitud": corr,
            "recall": recall,
            "respuesta_rag": respuesta_rag,
            "ground_truth": ground_truth
        })

        tqdm.write(f"{pregunta[:33]+'...':<35} | {fid:<4} | {rel:<4} | {corr:<4} | {recall:.2f}")

    # RESULTADOS FINALES
    total = len(casos_test)
    if total == 0: return

    promedio_fid = (score_fidelidad / total) * 100
    promedio_rel = score_relevancia / total
    promedio_corr = score_correctness / total
    promedio_recall = (score_recall / total) * 100

    print("\n" + "="*60)
    print(" REPORTE DE CALIDAD RAGAS (Con Ground Truth Manual)")
    print("="*60)
    print(f" FIDELIDAD (No Alucinacion):    {promedio_fid:.1f}%")
    print(f" RELEVANCIA (Calidad Resp.):    {promedio_rel:.2f} / 5.0")
    print(f" EXACTITUD (vs Ground Truth):   {promedio_corr:.2f} / 5.0")
    print(f" CONTEXT RECALL (Chunks):       {promedio_recall:.1f}%")
    print("="*60)
    
    # GUARDAR CSV
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_filename = f"reporte_ragas_{timestamp}.csv"
    csv_path = str(utils.project_root() / "src" / csv_filename)
    
    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["pregunta", "fidelidad", "relevancia", "exactitud", "recall", "respuesta_rag", "ground_truth"])
        writer.writeheader()
        writer.writerows(resultados_detallados)
        
    print(f" Reporte detallado guardado en: {csv_filename}")

    if promedio_fid < 80:
        print(" ALERTA: El modelo esta alucinando.")
    if promedio_corr < 3.5:
        print(" ALERTA: Las respuestas no coinciden con el Ground Truth.")
    if promedio_recall < 50:
        print(" ALERTA: El retriever no esta encontrando los chunks correctos.")

if __name__ == "__main__":
    main()
