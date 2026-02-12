"""
================================================================================
HERRAMIENTA DE AYUDA - CREACION DE GROUND TRUTH 
================================================================================

FLUJO INVERSO (Más fácil que pensar preguntas de la nada):
1. El script te MUESTRA un chunk de tu base de datos.
2. Tu LEES el texto y piensas: "¿Qué pregunta respondería este texto?"
3. Escribes la pregunta. Añade contexto que está en el chunk pero no copies directamente el texto. Hazlo con tus propias palabras e incluye el contexto.
4. Escribes la respuesta ideal basándote en lo que ves en el chunk.
5. El ID del chunk se guarda automáticamente.


¿Para QUIÉN es mi RAG?

 Público general (pacientes, ciudadanos)
   → Simplifica el lenguaje ponlo natural 

    Profesionales (médicos, ingenieros)
   → Puedes mantener algo de tecnicismo


EJEMPLO DE PREGUNTAS:

NO SE DEBE DE HACER :

"¿Qué es la hipertensión?"

RESPUESTA: (Mantener todo el tecnicismo:)
"La hipertensión arterial esencial se caracteriza por 
una presión arterial sistólica ≥140 mmHg y/o diastólica 
≥90 mmHg, secundaria a un incremento de las resistencias 
vasculares periféricas"


SI SE DEBE DE HACER:

PREGUNTA:
"¿Qué es la hipertensión?"

RESPUESTA: ( Simplificar manteniendo precisión:)
"Es tener la presión arterial alta. Se diagnostica cuando 
la presión está por encima de 140/90. Esto ocurre porque 
los vasos sanguíneos ofrecen más resistencia al paso de 
la sangre"


¿Cuándo SÍ mantener términos técnicos?
- Tu RAG es para PROFESIONALES
- El término NO tiene equivalente simple
- Perderías PRECISIÓN al simplificar


================================================================================
REQUISITOS MINIMOS PARA EL PROYECTO (OBLIGATORIO)
================================================================================

CANTIDAD DE PREGUNTAS:
- MINIMO: 20 preguntas.
- IDEAL: 30-50 preguntas.
- Regla: Minimo 3 preguntas por cada PDF de tu base de datos.

================================================================================
COMO CREAR BUENAS PREGUNTAS
================================================================================

EJEMPLO DE PREGUNTAS:
- Pregunta DIRECTA: "¿Cada cuanto se inyecta la insulina?"
- Pregunta con NEGACION: "¿Se puede inyectar insulina solo una vez al dia?"
- Pregunta REFORMULADA: "¿Cual es la frecuencia de administracion de insulina?"


================================================================================
ERRORES A EVITAR
================================================================================

1. Solo preguntas literales (copy-paste del texto).
   -> Solucion: Reformular con tus palabras.

2. Respuestas muy cortas (3-4 palabras).
   -> Solucion: Escribir frases completas con contexto. Incluir sinonimos o reformulaciones que el RAG podria usar. 
   No copiar y pegar directamente del texto.

3. Solo preguntas faciles.
   -> Solucion: Incluir 20% de preguntas dificiles o trampas. 
   Ejemplo: "¿Se cura la diabetes con insulina?" Ya sabes que la respuesta es NO, pero el RAG debe responderla.



================================================================================
EJEMPLOS
================================================================================
 ID: diabetes_es.pdf_child_17
 Fuente: diabetes_es.pdf
 Categoría: Enfermedades Crónicas
HIPOGLUCEMIA
7
Es la bajada de glucosa en sangre por debajo de los niveles normales.
(Solo en quienes estén en tratamiento con insulina o con antidiabéticos orales).
LOS SÍNTOMAS HABITUALES SON: mareo, debilidad, visión borrosa, sudoración y
alteraciones de la conciencia.
¿QUÉ ES?
• Ante mareo o debilidad si es posible conﬁrmar la hipoglucemia con un glucómetro.

PREGUNTA: ¿Cuáles son los síntomas de la hipoglucemia?

 PASO 2: Escribe la RESPUESTA IDEAL basándote en el texto.
 ( Incluir sinonimos o reformulaciones o resume lo importante del chunk)
 RESPUESTA IDEAL: Los síntomas habituales de la hipoglucemia son mareo, debilidad, visión borrosa, sudoración y 
 alteraciones de la conciencia. Si aparecen estos síntomas, es importante confirmar la bajada de azúcar con un glucómetro
----------------------------------------------------------------------


Los chunks incompletos NO SE USAN para crear preguntas.
EJEMPLO

 ID: agorafobia_es.pdf_child_28
 Fuente: agorafobia_es.pdf
 Categoría: Salud Mental
----------------------------------------------------------------------

ocurrir.
· Comportamientos o conductas que tratan de evitar esas
cosas.
Estos tres puntos están relacionados entre sí; pero la única
forma de empezar a actuar es cambiando las conductas. Se
empieza cambiando la conducta, para luego ir cambiando
sensaciones y pensamientos.
Por eso, es importante la lista que acaba de hacer. Tiene que

================================================================================


"""

import json
import os
import sys
import random
import logging
import chromadb
from dotenv import load_dotenv

# Añadir la carpeta padre (src/) al path para encontrar utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils


ENV_PATH = utils.project_root() / ".env"
load_dotenv(dotenv_path=ENV_PATH)

utils.setup_logging()  

logger = logging.getLogger("crear_ground_truth")

ARCHIVO_SALIDA = "golden_set_manual.jsonl"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
logger.info(f"Directorio Actual: {CURRENT_DIR}")

DB_DIR = str(utils.project_root() / "chroma_db")
logger.info(f"Ruta Base de Datos: {DB_DIR}")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "coleccion_leyes")
logger.info(f"Nombre de la Coleccion: {COLLECTION_NAME}")

def main():
    print("\n" + "="*70)
    print(" CREADOR DE GROUND TRUTH")
    print(" (Lee chunks y crea preguntas a partir de ellos)")
    print("="*70)
    
    # Conectar a ChromaDB
    if not os.path.exists(DB_DIR):
        print(f" No existe la base de datos en {DB_DIR}")
        return
    
    client_db = chromadb.PersistentClient(path=DB_DIR)
    collection = client_db.get_collection(name=COLLECTION_NAME)
    
    # Obtener todos los chunks
    all_data = collection.get(include=["documents", "metadatas"])
    total_chunks = len(all_data['ids'])
    print(f" Conectado. Total de chunks disponibles: {total_chunks}")
    
    # Crear lista de índices para ir mostrando
    indices_disponibles = list(range(total_chunks))
    random.shuffle(indices_disponibles)  # Mezclar para variedad
    
    contador = 0
    idx_actual = 0
    
    while idx_actual < len(indices_disponibles):
        print("\n" + "="*70)
        print(f" CHUNK #{idx_actual + 1} de {total_chunks}")
        print("="*70)
        
        # Obtener chunk actual
        i = indices_disponibles[idx_actual]
        chunk_id = all_data['ids'][i]
        chunk_text = all_data['documents'][i]
        chunk_meta = all_data['metadatas'][i]
        
        # Mostrar información del chunk
        print(f"\n ID: {chunk_id}")
        print(f" Fuente: {chunk_meta.get('source', '?')}")
        print(f" Categoría: {chunk_meta.get('category', '?')}")
        print("-"*70)
        print(f"\n{chunk_text}\n")
        print("-"*70)
        
        # Opciones
        print("\n OPCIONES:")
        print("   [1] Crear pregunta a partir de este chunk")
        print("   [2] Saltar al siguiente chunk")
        print("   [3] Salir y guardar")
        
        opcion = input("\n Elige (1/2/3): ").strip()
        
        if opcion == '3':
            break
        elif opcion == '2':
            idx_actual += 1
            continue
        elif opcion == '1':
            # Crear pregunta
            print("\n" + "-"*40)
            print(" PASO 1: Escribe una PREGUNTA que este texto responda.")
            print(" (Ej: '¿Cuáles son los síntomas de...?')")
            pregunta = input(" PREGUNTA: ").strip()
            
            if not pregunta:
                print(" Pregunta vacía, saltando...")
                idx_actual += 1
                continue
            
            # Respuesta ideal
            print("\n PASO 2: Escribe la RESPUESTA IDEAL basándote en el texto.")
            print(" ( Incluir sinonimos o reformulaciones o resume lo importante del chunk)")
            respuesta = input(" RESPUESTA IDEAL: ").strip()
            
            if not respuesta:
                print(" Respuesta vacía, saltando...")
                idx_actual += 1
                continue
            
            # Guardar
            entrada = {
                "query": pregunta,
                "ground_truth": respuesta,
                "relevant_ids": [chunk_id],
                "metadata": {
                    "source": chunk_meta.get('source', ''),
                    "category": chunk_meta.get('category', '')
                }
            }
            
            with open(ARCHIVO_SALIDA, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entrada, ensure_ascii=False) + "\n")
            
            contador += 1
            print(f"\n Guardado (Total creadas: {contador})")
            idx_actual += 1
        else:
            print(" Opción no válida.")
    
    print("\n" + "="*70)
    print(f" COMPLETADO: {contador} preguntas guardadas en {ARCHIVO_SALIDA}")
    print("="*70)

if __name__ == "__main__":
    main()
