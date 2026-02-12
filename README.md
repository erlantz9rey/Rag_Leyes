##########################################################
## ⚖️ Asistente Legal AI: Multimodal RAG con LangGraph
##########################################################

- Sistema avanzado de asistencia legal capaz de procesar documentos PDF y evidencia visual (imágenes) utilizando una arquitectura de Grafo de Estados y modelos de lenguaje de última generación.

---------------------------------------------------------------------------------------------------------------------

#################################
## Características Principales
#################################

- Procesamiento Multimodal: Análisis de imágenes de evidencia y documentos PDF mediante Gemini 2.0 Flash.

- Arquitectura de Grafo (LangGraph): Implementa un flujo de trabajo inteligente:

- Router: Clasifica la consulta en categorías legales.

- Rewriter: Optimiza la pregunta del usuario para búsqueda vectorial.

- Ranker: Re-rankeo de resultados con Cross-Encoder (BGE-Reranker).

- Ingesta Jerárquica: Sistema de Chunks Padre-Hijo para mantener el contexto global sin perder precisión semántica.

- Interfaz Completa: UI en Streamlit con soporte para Texto-a-Voz (TTS) y Evidencia Visual integrada.

-------------------------------------------------------------------------------------------------------------------

######################
## Stack Tecnológico
######################

- LLMs: Google Gemini (Visión), Groq/Llama 3 (Premium).

- Embeddings: intfloat/multilingual-e5-small.

- Base de Datos Vectorial: ChromaDB (Persistente).

- Orquestación: LangGraph & LangChain.

- Frontend: Streamlit con componentes personalizados de audio y portapapeles.

-------------------------------------------------------------------------------------------------------------------

################################
## 📁 Estructura del Proyecto
################################

ProyectoRag
├── chroma_db/
├── data/
│   ├── Pdfs/               # Documentos legales para indexar
│   └── Img/ 
├──api/
|   └── backend.py          # API FastAPI y lógica del Grafo de Estados
├──app/
|   ├──.streamlit/
|   └── RagStreamlit.py     # Interfaz de usuario
├── src/
│   ├── cargar_pdfs.py      # Pipeline de ingesta y análisis visual con Gemini  
│   └── utils.py 
├── requirements.txt
└── .env                # Variables de entorno

-------------------------------------------------------------------------------------------------------------------

#################################
## Configuración e Instalación
#################################

1. Requisitos Previos
    Python 3.10+
    Claves de API para Google Gemini y Groq.

2. Instalación
    Bash
    git clone https://github.com/tu-usuario/legalmind-rag.git
    cd legalmind-rag
    pip install -r requirements.txt

-------------------------------------------------------------------------------------------------------------------

#################
## Guía de Uso
#################

- Paso 1: Ingesta de Datos
    Ejecuta el script para procesar tus leyes y evidencias. Gemini analizará las imágenes para extraer texto (OCR) y categorizarlas legalmente.
    Bash
    python cargar_pdfs.py

- Paso 2: Iniciar el Backend
    La API gestiona el flujo de razonamiento y la búsqueda en la base de datos.
    Bash
    uvicorn api.backend:app --reload --port 8000

- Paso 3: Lanzar la Interfaz
    Bash
    streamlit run app/streamlit_app.py

-------------------------------------------------------------------------------------------------------------------

################################
## Lógica del Grafo (Backend)
################################

- El sistema no solo busca texto; razona sobre la mejor forma de obtenerlo:

- Clasificación: Detecta si la duda es Penal, Civil, etc.

- Re-Ranking: Tras la búsqueda inicial, el modelo BAAI/bge-reranker-v2-m3 evalúa qué fragmentos son realmente relevantes para la pregunta.

- Inyección Visual: Si se detecta una imagen en la base de datos con alta relevancia, se inyecta como contexto visual para que el LLM la describa en su respuesta.

-------------------------------------------------------------------------------------------------------------------

#################################
## Lógica de la RAG Multimodal
#################################

Pregunta -> Categorizar -> Query Rewriting -> Embeddings (top 10) - > Cross-Encoder (Re-ranker) -> Analizar Gemini -> Respuesta

1-Pregunta: Recepción de inputs vía texto, voz o archivos (PDF/Imagen).

2-Categorización: Clasificación automática mediante LLM para filtrar el dominio legal específico.

3-Query Rewriting: Transformación de la duda del usuario en una consulta técnica optimizada para recuperar leyes exactas.

4-Embeddings (Top 10): Búsqueda semántica en ChromaDB utilizando el modelo multilingual-e5-small.

5-Cross-Encoder (Re-ranker): Re-evaluación de los resultados con un modelo ms-marco-MiniLM para eliminar ruido y asegurar la máxima relevancia.

6-Analizar Gemini: Procesamiento multimodal (Vision) para integrar pruebas visuales o documentos escaneados en la respuesta.

7-Respuesta: Generación final esquemática, fundamentada en las fuentes recuperadas y con soporte de audio (TTS).


##########################################################
##  REPORTE DE CALIDAD RAGAS (Con Ground Truth Manual)
##########################################################

 - FIDELIDAD (No Alucinacion):    57.9%
 - RELEVANCIA (Calidad Resp.):    3.26 / 5.0
 - EXACTITUD (vs Ground Truth):   3.66 / 5.0
 - CONTEXT RECALL (Chunks):       23.7%

=========================================================================
🏆 RESULTADOS FINALES: COMPARATIVA DE ESTRATEGIAS DE CHUNKING
=========================================================================
## Modelo : all-MiniLM-L6-v2
|       Configuración        |  Hit Rate @5  |  MRR  |  Latencia Media  |
|:---------------------------|:--------------|------:|:-----------------|
| v1 (size:400, overlap:50)  | 23.7%         | 0.45  | 25.67 ms
| v3 (size:800, overlap:100) | 2.6%          | 0.007 | 23.77 ms         |
| v3 (size:950, overlap:150) | 2.6%          | 0.007 | 21.57 ms         |
=========================================================================
## Modelo : BAAI/bge-small-en-v1
|       Configuración        |  Hit Rate @5  |  MRR  |  Latencia Media  |
|:---------------------------|:--------------|------:|:-----------------|
| v1 (size:400, overlap:50)  | 5.3%          | 0.014 | 33.70 ms         |
| v2 (size:800, overlap:100) | 0.0%          | 0     | 32.80 ms         |
| v3 (size:950, overlap:150) | 2.6%          | 0.007 | 33.27 ms         |
=========================================================================
## Modelo : intfloat/multilingual-e5-small
|       Configuración        |  Hit Rate @5  |  MRR  |  Latencia Media  |
|:---------------------------|:--------------|------:|:-----------------|
| v1 (size:400, overlap:50)  | 94.7%         | 0.742 | 24.38 ms         |
| v2 (size:800, overlap:100) | 13.2%         | 0.32  | 22.90 ms         |
| v3 (size:950, overlap:150) | 6.2%          | 0.05  | 21.95 ms         |
=========================================================================

#############################
## Notas de Implementación
#############################

Seguridad: El sistema incluye reglas críticas para prevenir comandos de jailbreak y evitar que el modelo revele sus instrucciones internas.