# src/00_utils.py

"""
 CHEAT SHEET: GUÍA DE MÉTRICAS PARA RAG 
============================================================

1.  Precision@K (Calidad vs Ruido)
   - Pregunta: "De los K documentos que recuperé, ¿qué % son útiles?"
   - En RAG: Si es baja, le estamos pasando "basura" extra al LLM. Los modelos potentes lo toleran, los pequeños se confunden.

2.  Recall@K (Cobertura Total)
   - Pregunta: "De TODO lo útil que hay escondido en la base de datos, ¿qué % encontré?"
   - En RAG: ¡CRÍTICA! Si el Recall es bajo, tu chatbot NO tiene la información para responder. Alucinará.

3. MRR (Velocidad de Acierto)
   - Pregunta: "¿En qué posición salió el PRIMER documento útil?"
   - 1.0 = El primer resultado era útil.
   - 0.5 = El útil estaba en posición 2.
   - En RAG: Vital. Queremos el contexto correcto lo más arriba posible para no desperdiciar tokens.

4.  nDCG@K (Calidad del Orden)
   - Pregunta: "¿Están los resultados ordenados perfectamente por relevancia?"
   - A diferencia de Precision/Recall, aquí el ORDEN importa.
   - Premia poner lo "muy relevante" antes que lo "poco relevante".
   - Es la métrica favorita de los buscadores profesionales (Google, Bing).

5.  MAP (Consistencia)
   - Pregunta: "¿Qué tan limpio es mi ranking general?"
   - Mide el área bajo la curva de precisión-recall. Es estricta y robusta.

- RESUMEN PARA TU PROYECTO FINAL:
- ¿Tu LLM dice "no lo sé"? -> Tu RECALL es bajo (no encuentra la info).
- ¿Tu LLM responde mal confundido por otro tema? -> Tu MRR es bajo (la info correcta quedó abajo).

Las IMPRESCINDIBLES para RAG:  
- Recall@K (con K alto, ej. 10 o 20): "¿Está la respuesta en algún lugar de los chunks que recuperé?. ¿Encontramos al menos un documento útil". Si recuperas 5 chunks y la respuesta está en el 6, el LLM alucinará. Esta es la métrica de "seguridad".
- MRR (Mean Reciprocal Rank): "¿El chunk más importante está el primero? ¿Estaba arriba del todo?". Los LLMs son vagos "Lost in the Middle phenomenon". Si la info clave está el primero, el LLM responde mejor.
"""

import logging
import re
import unicodedata
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter  # Para contar líneas repetidas en PDFs
import numpy as np
import json  # Para manejo de JSON
from pathlib import Path  # Para manejo de rutas de archivos
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF


def setup_logging(level: int = logging.INFO) -> None:
    """Configura logging simple para la consola."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )



setup_logging()
logger = logging.getLogger("Utils")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def data_dir() -> Path:
    return project_root() / "data"


def generar_embeddings(modelo, textos):
    """
    Genera embeddings normalizados para una lista de textos.
    
   
    Args:
        modelo: Modelo SentenceTransformer ya cargado
        textos (list[str]): Lista de textos a convertir en embeddings
        
    Returns:
        list[list[float]]: Lista de vectores normalizados
        
    Nota: 
        normalize_embeddings=True hace que todos los vectores tengan longitud 1,
        lo que mejora la comparación por similitud coseno.
        
    Ejemplo:
        from sentence_transformers import SentenceTransformer
        modelo = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        embeddings = generar_embeddings(modelo, ["Hola mundo", "Adiós"])
    """
    return modelo.encode(textos, normalize_embeddings=True).tolist()




def limpiar_texto_basico(texto: str, 
                         minusculas: bool = True,
                         quitar_numeros: bool = False,
                         quitar_simbolos: bool = True) -> str:
    """
    Limpieza MÍNIMA de texto.
    BM25 → limpiar_texto_basico
    CRÍTICO para BM25 y TF-IDF: trabajan con tokens exactos.
    IMPORTANTE: El preprocesado MEJORA significativamente TF-IDF/BM25
    porque estas técnicas léxicas son MUY sensibles a variaciones.
    
    Args:
        texto: Texto a limpiar
        minusculas: Convertir a minúsculas (recomendado: True)
        quitar_numeros: Eliminar números (False para docs técnicos)
        quitar_simbolos: Eliminar símbolos especiales (recomendado: True)
    
    Returns:
        Texto limpio
    
    Ejemplo:
        >>> limpiar_texto_basico("¡Python 3.12 es GENIAL! ")
        'python 3 12 es genial'
    """
    if minusculas:
        texto = texto.lower()
    
    # Eliminar números, no siempre lo queremos dependiendo del dominio
    if quitar_numeros:  
        # r'\d+' El patrón a buscar (números)
        texto = re.sub(r'\d+', ' ', texto)
    
    # Eliminar símbolos como puntuación, comas, emojis, etc.
    if quitar_simbolos:
        # "Dame todo lo que NO sea letra/número/espacio" 
        texto = re.sub(r'[^a-záéíóúñü\s0-9]', ' ', texto)
    
    # Normalizar espacios en blanco (múltiples espacios → 1 espacio)
    # \s Cualquier espacio en blanco (espacio, tab, salto de línea)
    # strip()  - quitar espacios al principio y al final
    texto = re.sub(r'\s+', ' ', texto).strip()
    
  
    # Eliminar acentos para unificar palabras (camion = camión)
    # Esto mejora la búsqueda porque usuarios pueden escribir con o sin tildes.
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto)
                        if unicodedata.category(c) != 'Mn')

    return texto


import re


def hacer_chunking(texto: str, chunk_size=200, overlap=50) -> list:
    """
    Usa LangChain para dividir el texto en chunks.

    args:
        texto: str
        chunk_size: int
        overlap: int
    returns:
        list de chunks
    """

    logger.info(f" Configurando Splitter: Size={chunk_size}, Overlap={overlap}")
    
    # 1. Crear el objeto splitter 

    # Usamos RecursiveCharacterTextSplitter. Es "Recursivo" porque prueba la estrategia 1, si falla prueba la 2, si falla la 3... 
    # Intenta siempre ser lo más semántico posible (respetar la estructura del texto) antes de rendirse y cortar a machete.
    # Tiene una lista de prioridades para el corte.
    # Prioridad 1 (\n\n): Cortes limpios (Párrafos). Mira si puede cortar justo donde acaba un párrafo.  
    #   1- No importa si mide menos de 200 caracteres. 
    # Prioridad 2 (\n): orta al final de una línea. Si el párrafo mide más de 200 caracteres, busca un salto de línea simple (\n) para cortar.
    # Prioridad 3 ( ): Cortes por Espacio en blanco - Si no puede cortar por párrafos ni por líneas, cortará entre palabras. La clave es que NO corta la palabra
    # Prioridad 4 (""): Letra a Letra - Si es una palabra muy larga y no hay espacios , corta por lo que pueda. 

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,      # Tamaño objetivo
        chunk_overlap=overlap,      # Solapamiento para contexto
        length_function=len,        # Cómo medimos (por caracteres)
        separators=["\n\n", "\n", " ", ""] # Prioridad de corte : Párrafos > Líneas > Frases 
    )

    # 2. Ejecutar el corte
    chunks = text_splitter.split_text(texto)
    
    return chunks



def limpiar_para_embeddings_pdf(texto: str, umbral_repeticion: int = 3) -> str:
    """
    Limpieza SUAVE para texto extraído de PDF, pensada para embeddings.
    Objetivo: quitar ruido de extracción SIN perder información técnica.

    Limpieza suave para PDFs (quitar ruido sin destruir semántica, los embeddings (Qwen) entienden tildes, mayúsculas y contexto)
    No queremos destruir esa información. Solo queremos quitar el "ruido de PDF" (guiones partidos, espacios raros).
    
    Args:
        texto: Texto extraído del PDF
        umbral_repeticion: Si una línea aparece más de X veces, se considera encabezado/pie y se elimina
    """
    if not texto:
        return ""

    # 1) Normalizar saltos de línea y tabs
    #Convierte todos los retornos de carro (\r) en saltos de línea (\n) y los tabs (\t) en espacios.
    t = texto.replace("\r", "\n").replace("\t", " ")

    # 2) Unir palabras cortadas por guion al final de línea: "trans-\nformers" -> "transformers"
    #    (esto es MUY típico en PDFs que las palabras largas se dividan al final de línea.)
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)

    # 3) Convertir saltos de línea “sueltos” en espacios (sin cargarse párrafos):
    #    - Si hay dos saltos de línea seguidos, los deja (para separar párrafos).
    #    - si hay salto simple lo cambiamos por espacio
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)

    # 4) Quitar “basura” invisible / espacios raros
    #    \u00A0 = non-breaking space típico en PDFs
    #  Sustituye caracteres como el espacio no separable (\u00A0) por espacios normales.
    t = t.replace("\u00A0", " ")

    # 5) Compactar espacios
    # Reemplaza múltiples espacios seguidos por uno solo y elimina espacios al principio y final
    t = re.sub(r"[ ]{2,}", " ", t).strip()

    # 6) NUEVO: Eliminar líneas repetidas (encabezados/pies de página)
    #    Si una línea corta aparece muchas veces, probablemente es un encabezado
    #    Ejemplo: "LA DEPRESIÓN - Información para pacientes" en cada página
    lineas = t.split('\n')
    
    # Contar ocurrencias de cada línea (solo líneas cortas, < 100 chars)
    contador = Counter(linea.strip() for linea in lineas if len(linea.strip()) < 100)
    
    # Identificar líneas que se repiten demasiado (probables encabezados)
    lineas_repetidas = {linea for linea, count in contador.items() 
                        if count > umbral_repeticion and len(linea) > 5}
    
    # Filtrar esas líneas
    if lineas_repetidas:
        logger.debug(f"Eliminando {len(lineas_repetidas)} líneas repetidas (encabezados/pies)")
        lineas_limpias = [linea for linea in lineas if linea.strip() not in lineas_repetidas]
        t = '\n'.join(lineas_limpias)

    return t


# ============================================================================
# LIMPIEZA AVANZADA PARA PDFs CON IMÁGENES
# ============================================================================
# ¿POR QUÉ EXISTE ESTA FUNCIÓN?
# Cuando un PDF tiene IMÁGENES, gráficos o iconos, los extractores de texto
# (PyMuPDF, pdfplumber, pymupdf4llm) a veces "ven" caracteres invisibles o
# símbolos especiales que formaban parte de la imagen o de fuentes embebidas.
#
# EJEMPLOS DE CARACTERES PROBLEMÁTICOS:
#   - \uf0b7 → Bullet point de fuente Wingdings
#   - \uf075 → Icono de fuente FontAwesome
#   - \ue000-\uf8ff → Todo el rango "Private Use Area" de Unicode
#   - \x00-\x08 → Caracteres de control (invisibles)
#
# IMPACTO EN RAG:
# Estos caracteres "basura" contaminan los embeddings porque el modelo
# los tokeniza aunque no aporten significado semántico.
# ============================================================================

def limpiar_caracteres_imagen(texto: str, verbose: bool = False) -> str:
    """
    Limpieza AGRESIVA de caracteres problemáticos típicos de PDFs con imágenes.
    
    Esta función complementa a limpiar_para_embeddings_pdf() y está diseñada
    específicamente para PDFs que contienen imágenes, gráficos, iconos o
    fuentes especiales embebidas.
    
    CARACTERES QUE ELIMINA:
    ========================
    1. Private Use Area (PUA): \uE000-\uF8FF
       → Caracteres "inventados" por fuentes como Wingdings, Symbol, FontAwesome
       
    2. Caracteres de control: \x00-\x08, \x0B-\x0C, \x0E-\x1F
       → Invisibles pero ocupan espacio en tokens
       
    3. Símbolos de reemplazo: \uFFFD (�)
       → Aparecen cuando el PDF tiene encoding roto
       
    4. Bullets y símbolos comunes mal extraídos:
       → \uf0b7, \uf0a7, \uf075, etc.
       
    5. Separadores decorativos:
       → Líneas de guiones, asteriscos, etc.
    
    Args:
        texto: Texto extraído del PDF (puede ser de PyMuPDF o pymupdf4llm)
        verbose: Si True, loguea estadísticas de limpieza
        
    Returns:
        Texto limpio sin caracteres problemáticos
        
    Ejemplo de uso:
        >>> texto_sucio = "La ansiedad \\uf0b7 es un trastorno \\ue001 común"
        >>> texto_limpio = limpiar_caracteres_imagen(texto_sucio)
        >>> print(texto_limpio)
        "La ansiedad es un trastorno común"
    """
    if not texto:
        return ""
    
    original_len = len(texto)
    t = texto
    
    # =========================================================================
    # PASO 1: Eliminar Private Use Area (PUA) completo
    # =========================================================================
    # Unicode reserva el rango E000-F8FF para fuentes personalizadas.
    # Estos caracteres NO tienen significado universal y ensucian embeddings.
    t = re.sub(r'[\uE000-\uF8FF]', '', t)
    
    # =========================================================================
    # PASO 2: Eliminar caracteres de control (excepto \t, \n, \r)
    # =========================================================================
    # Los caracteres 0x00-0x08, 0x0B-0x0C, 0x0E-0x1F son "invisibles"
    # pero pueden aparecer en PDFs con fuentes corruptas.
    t = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', t)
    
    # =========================================================================
    # PASO 3: Eliminar carácter de reemplazo Unicode (�)
    # =========================================================================
    # Aparece cuando hay errores de codificación. Indica "basura".
    t = t.replace('\uFFFD', '')
    
    # =========================================================================
    # PASO 4: Limpiar símbolos específicos comunes en PDFs médicos/técnicos
    # =========================================================================
    # Estos son los más frecuentes que he visto en PDFs reales:
    simbolos_comunes = [
        '\uf0b7',  # Bullet Wingdings
        '\uf0a7',  # Bullet alternativo
        '\uf075',  # Icono FontAwesome
        '\uf0fc',  # Check mark Wingdings
        '\uf0fe',  # Cuadrado Wingdings
        '\uf0d8',  # Flecha arriba
        '\uf0e0',  # Sobre/email
        '\u2022',  # Bullet estándar (• se mantiene o reemplaza)
    ]
    for simbolo in simbolos_comunes:
        t = t.replace(simbolo, '')
    
    # =========================================================================
    # PASO 5: Normalizar bullets a formato consistente
    # =========================================================================
    # Convertir varios tipos de bullets a guión simple para consistencia
    bullets_variados = ['●', '○', '◆', '◇', '▪', '▫', '►', '▸', '‣', '⁃']
    for bullet in bullets_variados:
        t = t.replace(bullet, '- ')
    
    # =========================================================================
    # PASO 6: Eliminar líneas decorativas
    # =========================================================================
    # Patrones típicos de separadores visuales en PDFs
    t = re.sub(r'[-=_]{5,}', '', t)  # Líneas de guiones, iguales, guiones bajos
    t = re.sub(r'[*]{3,}', '', t)    # Líneas de asteriscos
    t = re.sub(r'[─━]{3,}', '', t)   # Líneas Unicode box-drawing
    
    # =========================================================================
    # PASO 7: Limpiar referencias a imágenes de pymupdf4llm
    # =========================================================================
    # pymupdf4llm genera líneas como: ![image](image_path.png)
    # Estas no aportan nada al RAG textual
    t = re.sub(r'!\[.*?\]\(.*?\)', '[IMAGEN]', t)  # Reemplaza por marcador
    
    # =========================================================================
    # PASO 8: Compactar espacios resultantes
    # =========================================================================
    t = re.sub(r'[ ]{2,}', ' ', t)
    t = re.sub(r'\n{3,}', '\n\n', t)  # Máximo 2 saltos de línea seguidos
    t = t.strip()
    
    # Logging opcional
    if verbose:
        chars_eliminados = original_len - len(t)
        pct = (chars_eliminados / original_len * 100) if original_len > 0 else 0
        logger.info(f"Limpieza de caracteres: eliminados {chars_eliminados} chars ({pct:.1f}%)")
    
    return t


def leer_pdf_markdown(ruta_pdf: str, limpiar: bool = True) -> str:
    """
    Lee un PDF usando pymupdf4llm (mejor para PDFs con imágenes/tablas).
    
    DIFERENCIA CON leer_pdf():
    ==========================
    - leer_pdf() usa PyMuPDF básico → Rápido pero pierde estructura
    - leer_pdf_markdown() usa pymupdf4llm → Mantiene estructura en Markdown
    
    CUÁNDO USAR CADA UNO:
    =====================
    - PDFs de texto plano (artículos, libros) → leer_pdf() es suficiente
    - PDFs con tablas, imágenes, columnas → leer_pdf_markdown() es mejor
    
    Args:
        ruta_pdf: Ruta al archivo PDF
        limpiar: Si True, aplica limpiar_caracteres_imagen() automáticamente
        
    Returns:
        Texto en formato Markdown (limpio si limpiar=True)
        
    Ejemplo:
        >>> texto = leer_pdf_markdown("informe_medico.pdf")
        >>> chunks = hacer_chunking(texto)
        
    Requiere: pip install pymupdf4llm
    """
    import os
    
    if not os.path.exists(ruta_pdf):
        logger.error(f"No encuentro el archivo: {ruta_pdf}")
        return ""
    
    try:
        import pymupdf4llm
    except ImportError:
        logger.error("pymupdf4llm no está instalado. Ejecuta: pip install pymupdf4llm")
        logger.info("Usando leer_pdf() como fallback...")
        return leer_pdf(ruta_pdf)
    
    logger.info(f"Leyendo PDF con pymupdf4llm: {os.path.basename(ruta_pdf)}...")
    
    try:
        # pymupdf4llm.to_markdown() extrae texto manteniendo estructura
        texto_markdown = pymupdf4llm.to_markdown(ruta_pdf)
        
        logger.info(f"Extraídos {len(texto_markdown)} caracteres en formato Markdown")
        
        if limpiar:
            # Primero limpieza de caracteres de imagen
            texto_markdown = limpiar_caracteres_imagen(texto_markdown, verbose=True)
            # Luego limpieza general de PDFs
            texto_markdown = limpiar_para_embeddings_pdf(texto_markdown)
        
        return texto_markdown
        
    except Exception as e:
        logger.error(f"Error extrayendo PDF con pymupdf4llm: {e}")
        logger.info("Intentando con PyMuPDF básico como fallback...")
        return leer_pdf(ruta_pdf)


def limpiar_para_embeddings_completo(texto: str, 
                                      umbral_repeticion: int = 3,
                                      es_pdf_con_imagenes: bool = False) -> str:
    """
    Pipeline COMPLETO de limpieza para embeddings.
    
    Combina todas las funciones de limpieza en el orden óptimo.
    
    Args:
        texto: Texto extraído del PDF
        umbral_repeticion: Umbral para detectar headers/footers repetidos
        es_pdf_con_imagenes: Si True, aplica limpieza agresiva de caracteres
        
    Returns:
        Texto completamente limpio y listo para chunking
        
    Ejemplo:
        >>> texto_crudo = leer_pdf("documento.pdf")
        >>> texto_limpio = limpiar_para_embeddings_completo(texto_crudo, es_pdf_con_imagenes=True)
        >>> chunks = hacer_chunking(texto_limpio)
    """
    if not texto:
        return ""
    
    # Paso 1: Si tiene imágenes, limpiar caracteres problemáticos primero
    if es_pdf_con_imagenes:
        texto = limpiar_caracteres_imagen(texto, verbose=True)
    
    # Paso 2: Limpieza general de PDFs
    texto = limpiar_para_embeddings_pdf(texto, umbral_repeticion)
    
    return texto


# ============================================================================
# MÉTRICAS
# ============================================================================

def precision_at_k(resultados: List[str], relevantes: Set[str], k: int) -> float:
    """Precision@K: % de top K que son relevantes
        Calcula qué % de los primeros K resultados son relevantes
        ¿Los que devolví son buenos?
        
        Args:
            resultados:  Lista de IDs devueltos por el sistema
                   Ejemplo: ["doc_A", "doc_B", "doc_C", "doc_D"]
        
            relevantes: Conjunto de IDs que son útiles (ground truth)
                   Ejemplo: {"doc_A", "doc_C", "doc_E"}
        
            k: Número de resultados a evaluar (top K)
                    Ejemplo: k=3 → solo miro los 3 primeros
        Returns:
             float: Proporción de relevantes en top K (entre 0.0 y 1.0)
    """

    # PASO 1: Quedarnos solo con los primeros K resultados devueltos
    top_k = resultados[:k]

     # Paso 2: Si no hay resultados, devolver 0 
    if not top_k:
        return 0.0
    
    logger.info(f" Precision: Top K: {top_k}")
     # Paso 3: Contar cuántos de los resultados en top_k son relevantes
    # relevantes_encontrados = TP (True Positives)
    relevantes_encontrados = 0
    for documento in top_k:
        if documento in relevantes: 
            relevantes_encontrados += 1

    #total_documentos = TP + FP (Todo lo que el modelo predijo como "bueno")
    total_documentos = len(top_k)
    logger.info(f" Precision: relevantes: {relevantes}")
    logger.info(f" Precision: k: {k}")
    logger.info(f" Precision: Total de documentos en top_k: {total_documentos}")
    logger.info(f" Precision: Relevantes encontrados: {relevantes_encontrados}")
    logger.info(f" Precision: Relevantes totales: {len(relevantes)}")

    # Paso 4: Calcular el porcentaje
    # precision = TP / (TP + FP)
    precision = relevantes_encontrados / total_documentos
    logger.info(f" Precision relevantes_encontrados / total_documentos: {precision}")
    return precision 


def recall_at_k(resultados: List[str], relevantes: Set[str], k: int) -> float:
    """Recall@K: % de relevantes encontrados en top K
       Calcula qué porcentaje de documentos relevantes fueron encontrados.
        ¿Encontré todos los buenos?

        Args:
            resultados: Lista de IDs devueltos por el sistema
                     Ejemplo: ["doc_A", "doc_B", "doc_C", "doc_D"]
            relevantes: Set de documentos que SÍ son relevantes para la pregunta
                     Ejemplo: {"doc_A", "doc_C", "doc_E"}
            k: Número de resultados a evaluar (top K)
                     Ejemplo: k=3 → solo miro los 3 primeros
        Returns:
             float: Proporción de relevantes encontrados en top K (entre 0.0 y 1.0)
    """

    # Paso 1: Si no hay documentos relevantes, no podemos medir
    if not relevantes:
        return 0.0
    
    # Paso 2: Quedarnos solo con los primeros K resultados devueltos
    top_k = set(resultados[:k])


    # Paso 3: Contar cuántos de los RELEVANTES están en top_k
    relevantes_encontrados = 0

   
    # Nota: Iteramos sobre relevantes, no sobre top_k

    for documento_relevante in relevantes:
        # ¿Este documento relevante está entre los K primeros?
        if documento_relevante in top_k:
            relevantes_encontrados += 1

    

    # Paso 4: Calcular el porcentaje
    # CONCEPTUALMENTE:
    #   relevantes_encontrados = TP
    #   total_relevantes = TP + FN (Todos los relevantes que existen, los encontrados + los perdidos)
    #   Recall = TP / (TP + FN)
    total_relevantes = len(relevantes)
    recall = relevantes_encontrados / total_relevantes
    return recall





 

def reciprocal_rank(resultados: List[str], relevantes: Set[str]) -> float:
    """
    ¿En qué posición aparece el PRIMER resultado útil? Es una métrica que premia encontrar algo útil PRONTO. 
    Cuanto MÁS ARRIBA esté el primer útil, MEJOR Puntuación.

    RR: 1 / posición del primer relevante
    Si encuentro útil en posición N → RR = 1/N
    Args:
        resultados: Lista ORDENADA de documentos devueltos por el sistema
                    Ejemplo: ["doc_A", "doc_B", "doc_C", "doc_D"]
        
        relevantes: Conjunto de documentos que son útiles
                    Ejemplo: {"doc_C", "doc_D", "doc_E"}
    Returns:
        float: Puntuación RR entre 0.0 y 1.0
               - 1.0 = El primer resultado es útil (perfecto)
               - 0.0 = Ningún resultado es útil (pésimo)
    """

    logger.info(f"Buscando primer documento útil...")
    logger.info(f"  Relevantes: {relevantes}")

    

    #  # Revisar cada resultado en orden. SIN start=1 (empieza en 0) , sino da error de división por cero
    for posicion, documento_id in enumerate(resultados, start=1):

        logger.info(f"  Posición {posicion}: {documento_id}")

         # Verificar si es útil
        es_util = documento_id in relevantes

        logger.info(f"  Posición {posicion}: {documento_id} → {'ÚTIL' if es_util else 'Basura'}")

        # Si encontramos el primer útil, calcular y retornar
        if es_util:
            reciprocal = 1.0 / posicion
           
            logger.info(f" Primer útil encontrado en posición {posicion}")
            logger.info(f"   Reciprocal Rank = 1/{posicion} = {reciprocal:.4f} = {reciprocal:.2%}")
            return reciprocal
        
    logger.info(" No se encontraron documentos útiles.")
    return 0.0
    





def average_precision(resultados: List[str], relevantes: Set[str]) -> float:
    """
    AP (Average Precision): Mide calidad del ranking
    AP es el promedio de nuestra 'puntería' (Precisión) cada vez que acertamos. 

    Premia:
    - Poner útiles arriba
    - Agrupar útiles juntos
    
    Castiga:
    - Intercalar basura entre útiles
    - Poner útiles abajo
    
    Args:
        resultados: Lista ORDENADA de IDs devueltos
        relevantes: Conjunto de IDs útiles
    
    Returns:
        float: Average Precision entre 0.0 y 1.0
    """
    
    # Paso 1: Si no hay relevantes, AP = 0
    if len(relevantes) == 0:
        return 0.0
    
    # Paso 2: Calcular precision en cada posición donde hay un relevante
    suma_precisiones = 0.0

    
    relevantes_encontrados = 0
    
    for posicion, documento_id in enumerate(resultados, start=1):
        # ¿Este documento es relevante?
        if documento_id in relevantes:
            relevantes_encontrados += 1
            
            # Calcular Precision@posicion
            precision_en_k = relevantes_encontrados / posicion
            
            # Sumar (solo cuenta cuando encontramos un relevante)
            suma_precisiones += precision_en_k
    
    # Paso 3: Promediar por el total de relevantes
    # (no por el total de documentos revisados)
    total_relevantes = len(relevantes)
    average_precision = suma_precisiones / total_relevantes
    
    return average_precision


def mean_average_precision(
    resultados_dict: Dict[str, List[str]], 
    relevantes_dict: Dict[str, Set[str]]
) -> float:
    """
    MAP (Mean Average Precision): Promedio de AP sobre múltiples queries
    
    Args:
        resultados_dict: {query_id: [doc_ids ordenados]}
        relevantes_dict: {query_id: {doc_ids relevantes}}
    
    Returns:
        float: MAP entre 0.0 y 1.0
    """
    
    if len(resultados_dict) == 0:
        return 0.0
    
    # Calcular AP para cada query
    aps = []
    for query_id in resultados_dict:
        resultados = resultados_dict[query_id]
        relevantes = relevantes_dict[query_id]
        
        ap = average_precision(resultados, relevantes)
        aps.append(ap)
    
    # Promediar todos los AP
    map_score = sum(aps) / len(aps)
    
    return map_score


def cargar_golden_set_jsonl(path: str) -> Optional[Dict]:
    """
    Carga dataset externo en formato JSONL.
    
    Formato esperado:
    {"id": "q1", "query": "...", "relevant_ids": ["doc_0"], "where_filter": {...}}
    
    Returns:
        Dict con ground truth o None si no existe el archivo
    """
    ground_truth = {}
    
    if not Path(path).exists():
        logger.warning(f" No se encontró {path}, usando dataset demo")
        return None
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:   # lee linea por linea
            item = json.loads(line.strip()) # Convertir JSON a Python , se quita el espacio en blanco
            ground_truth[item['id']] = { # Guarda en una lista de diccionarios por ID
                'query': item['query'],
                'relevantes': item['relevant_ids'],
                'where_filter': item.get('where_filter', None)
            }
    
    logger.info(f" Dataset cargado: {len(ground_truth)} queries desde {path}")
    return ground_truth




# ============================================================================
# FUNCIÓN PARA LEER PDF (PyMuPDF)
# ============================================================================
import fitz  # PyMuPDF - pip install pymupdf
import os

def leer_pdf(ruta_pdf: str) -> str:
    """
    Lee todo el texto de un archivo PDF.
    
    Args:
        ruta_pdf: Ruta absoluta o relativa al archivo PDF.
        
    Returns:
        String con todo el texto del PDF concatenado.
    """
    logger = logging.getLogger("utils")
    
    if not os.path.exists(ruta_pdf):
        logger.error(f" No encuentro el archivo: {ruta_pdf}")
        return ""

    logger.info(f" Leyendo PDF: {os.path.basename(ruta_pdf)}...")
    #doc es un objeto iterable- La len(doc) es el numero de paginas. Itera por cada pagina
    doc = fitz.open(ruta_pdf)
    texto_completo = ""
    
    for pagina in doc:
        texto_completo += pagina.get_text() + "\n"  # pagina.get_text() extrae el texto de la pagina + \n salto de linea
        
    logger.info(f"    Leídas {len(doc)} páginas ({len(texto_completo)} caracteres).")
    return texto_completo


# ============================================================================
# FUNCIÓN PARA LEER PDF CON TABLAS (pdfplumber)
# ============================================================================

# ¿Cuándo usar cada librería?
# 
# 1. PyMuPDF (fitz) - leer_pdf():
#    - Muy rápido
#    - Ideal para PDFs de texto plano (artículos, libros, manuales)
#    - No extrae tablas como estructuras, solo como texto
#    - Ya está incluido arriba
#
# 2. pdfplumber - leer_pdf_con_tablas():
#    - Más lento pero extrae TABLAS como listas de listas
#    - Ideal para PDFs con datos tabulares:
#      * Informes de laboratorio (glucosa, colesterol, etc.)
#      * Facturas y presupuestos
#      * Fichas técnicas con especificaciones
#
# IMPORTANTE: pdfplumber NO viene instalado por defecto.
# Instalación: pip install pdfplumber
#
# ¿Por qué importa en RAG?
# Si tu PDF tiene una tabla con "Glucosa: 120 mg/dL" y usas PyMuPDF,
# puede que extraiga "Glucosa 120 mg dL" todo seguido sin estructura.
# Con pdfplumber obtienes [["Glucosa", "120", "mg/dL"]] como lista,
# lo que permite generar texto más limpio para los embeddings.
# ============================================================================

def leer_pdf_con_tablas(ruta_pdf: str, extraer_texto: bool = True, extraer_tablas: bool = True) -> dict:
    """
    Lee un PDF extrayendo texto Y tablas por separado usando pdfplumber.
    
    Esta función es ideal para PDFs con datos tabulares como:
    - Informes médicos de laboratorio
    - Facturas y presupuestos
    - Fichas técnicas
    
    Args:
        ruta_pdf: Ruta al archivo PDF.
        extraer_texto: Si True, extrae el texto plano.
        extraer_tablas: Si True, extrae las tablas como listas.
        
    Returns:
        dict con claves:
            - 'texto': String con todo el texto del PDF
            - 'tablas': Lista de tablas, cada tabla es lista de filas
            - 'tablas_como_texto': Tablas convertidas a texto legible
            - 'num_paginas': Número de páginas del PDF
            - 'num_tablas': Número de tablas encontradas
            
    Ejemplo de uso:
        >>> resultado = leer_pdf_con_tablas("informe_lab.pdf")
        >>> print(resultado['texto'][:500])  # Ver primeros 500 chars
        >>> print(f"Encontradas {resultado['num_tablas']} tablas")
        >>> for tabla in resultado['tablas']:
        ...     print(tabla)  # Cada tabla es [[fila1], [fila2], ...]
            
    Nota: Requiere instalar pdfplumber: pip install pdfplumber
    """
    logger = logging.getLogger("utils")
    
    # Verificar que pdfplumber está instalado
    try:
        import pdfplumber
    except ImportError:
        logger.error("pdfplumber no está instalado. Ejecuta: pip install pdfplumber")
        return {
            'texto': '',
            'tablas': [],
            'tablas_como_texto': '',
            'num_paginas': 0,
            'num_tablas': 0,
            'error': 'pdfplumber no instalado'
        }
    
    # Verificar que el archivo existe
    if not os.path.exists(ruta_pdf):
        logger.error(f"No encuentro el archivo: {ruta_pdf}")
        return {
            'texto': '',
            'tablas': [],
            'tablas_como_texto': '',
            'num_paginas': 0,
            'num_tablas': 0,
            'error': 'Archivo no encontrado'
        }
    
    logger.info(f"Leyendo PDF con pdfplumber: {os.path.basename(ruta_pdf)}...")
    
    texto_completo = ""
    todas_las_tablas = []
    tablas_texto = ""
    num_paginas = 0
    
    try:
        with pdfplumber.open(ruta_pdf) as pdf:
            num_paginas = len(pdf.pages)
            
            for i, pagina in enumerate(pdf.pages, start=1):
                # Extraer texto plano
                if extraer_texto:
                    texto_pagina = pagina.extract_text()
                    if texto_pagina:
                        texto_completo += texto_pagina + "\n"
                
                # Extraer tablas
                if extraer_tablas:
                    tablas_pagina = pagina.extract_tables()
                    if tablas_pagina:
                        for tabla in tablas_pagina:
                            # Limpiar celdas None
                            tabla_limpia = []
                            for fila in tabla:
                                fila_limpia = [celda if celda else "" for celda in fila]
                                tabla_limpia.append(fila_limpia)
                            
                            todas_las_tablas.append(tabla_limpia)
                            
                            # Convertir tabla a texto legible
                            tablas_texto += f"\n--- Tabla (Página {i}) ---\n"
                            for fila in tabla_limpia:
                                tablas_texto += " | ".join(fila) + "\n"
                            tablas_texto += "---\n"
        
        logger.info(f"Leídas {num_paginas} páginas, encontradas {len(todas_las_tablas)} tablas.")
        
        return {
            'texto': texto_completo.strip(),
            'tablas': todas_las_tablas,
            'tablas_como_texto': tablas_texto.strip(),
            'num_paginas': num_paginas,
            'num_tablas': len(todas_las_tablas)
        }
        
    except Exception as e:
        logger.error(f"Error al procesar PDF: {e}")
        return {
            'texto': '',
            'tablas': [],
            'tablas_como_texto': '',
            'num_paginas': 0,
            'num_tablas': 0,
            'error': str(e)
        }


def combinar_texto_y_tablas(resultado_pdf: dict) -> str:
    """
    Combina texto y tablas de un PDF en un solo string para embeddings.
    
    Esta función toma el resultado de leer_pdf_con_tablas() y genera
    un texto unificado óptimo para generar embeddings.
    
    Args:
        resultado_pdf: Diccionario devuelto por leer_pdf_con_tablas()
        
    Returns:
        String combinado listo para chunking y embeddings
        
    Ejemplo:
        >>> resultado = leer_pdf_con_tablas("informe.pdf")
        >>> texto_para_rag = combinar_texto_y_tablas(resultado)
        >>> chunks = hacer_chunking(texto_para_rag)
    """
    texto = resultado_pdf.get('texto', '')
    tablas_texto = resultado_pdf.get('tablas_como_texto', '')
    
    if tablas_texto:
        return f"{texto}\n\n=== DATOS TABULARES ===\n{tablas_texto}"
    
    return texto
