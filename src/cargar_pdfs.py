import os
import glob
import sys
import logging
import json
import chromadb
import google.generativeai as genai
from pathlib import Path
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import utils 
from PIL import Image

# Configuración de rutas
ROOT_DIR = Path(__file__).resolve().parents[1] 
sys.path.append(str(ROOT_DIR / "src"))

# Cargar .env
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

utils.setup_logging()
logger = logging.getLogger("ingesta_gemini")

# Variables Globales desde .env
TAMANO_PADRE = 2000
TAMANO_HIJO = int(os.getenv("CHUNK_SIZE", 400))
SOLAPE_HIJO = int(os.getenv("CHUNK_OVERLAY", 150))

DB_DIR = str(ROOT_DIR / os.getenv("DB_PATH", "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "coleccion_leyes") 
MODELO_EMBEDDINGS = "intfloat/multilingual-e5-small"
model_emb = SentenceTransformer(MODELO_EMBEDDINGS)

# Configuración de Gemini
GEMINI_API_KEY = os.getenv("LLM_API_KEY")
MODELO_GEMINI = os.getenv("MODELO_LLM", "gemini-2.5-flash") 
genai.configure(api_key=GEMINI_API_KEY)
model_vision = genai.GenerativeModel(MODELO_GEMINI)

DIRECTORIO_IMAGENES = ROOT_DIR / "data" / "Img"
ruta_json = utils.data_dir() / "metadatos_pdfs.json"

# --- FUNCIONES DE APOYO ---

def clasificar_con_gemini(nombre_archivo, texto_o_imagen, es_imagen=False):
    """Usa Gemini para clasificar tanto texto como imágenes."""
    CATEGORIAS_VALIDAS = [
        "Derecho Civil", "Derecho Penal", "Derecho Administrativo",
        "Derechos Humanos", "Tráfico y Seguridad Vial", "Procedimiento Penal"
    ]

    prompt = f"""Eres un experto clasificador y analista de documentos legales.
        Analiza el contenido y responde en JSON.

        Si el contenido es una IMAGEN:
        1. Extrae TODO el texto que veas (OCR). 
        2. Si es una tabla, transcribe los datos clave (ej: "Mujer 50-70kg, 2 vasos vino: 0.50-0.69").
        3. Si es un documento guarda todas la condiciones importates y demas puntos importantes.
        4. Asigna una categoría legal de esta lista: {', '.join(CATEGORIAS_VALIDAS)}.

        Responde estrictamente en este formato JSON:
        {{"descripcion": "Aquí incluye todo el texto extraído y los datos de la tabla", "categoria": "Nombre de la Categoría"}}

        Nombre del archivo: {nombre_archivo}"""

    try:
        if es_imagen:
            # Entrada multimodal: texto + imagen
            response = model_vision.generate_content([prompt, texto_o_imagen])
        else:
            # Entrada de texto
            response = model_vision.generate_content(f"{prompt}\n\nTexto: {texto_o_imagen[:2000]}")

        # Limpieza de la respuesta JSON
        limpio = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(limpio)
        
        # Validar que la categoría sea permitida
        if data["categoria"] not in CATEGORIAS_VALIDAS:
            data["categoria"] = "General"
            
        return data
    except Exception as e:
        logger.warning(f"Error clasificando con Gemini {nombre_archivo}: {e}")
        return {"descripcion": "No disponible", "categoria": "General"}

def ingestar_imagenes_locales(directorio_img, collection, model_emb):    
    extensiones = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    archivos_img = []
    for ext in extensiones:
        archivos_img.extend(glob.glob(os.path.join(directorio_img, ext)))

    if not archivos_img:
        logger.info("No se encontraron imágenes locales.")
        return []

    resumen_guardado = []
    logger.info(f"Analizando {len(archivos_img)} imágenes directamente con Gemini Vision...")

    for idx, ruta_completa in enumerate(archivos_img):
        try:
            nombre_img = os.path.basename(ruta_completa)
            ruta_absoluta = os.path.abspath(ruta_completa).replace("\\", "/")
            
            # Cargar imagen para Gemini
            img_pil = Image.open(ruta_completa)
            
            # Clasificación y descripción en un solo paso
            resultado = clasificar_con_gemini(nombre_img, img_pil, es_imagen=True)
            
            descripcion_ia = resultado["descripcion"]
            categoria_ia = resultado["categoria"]

            logger.info(f"📸 Gemini analizó: {nombre_img} -> {categoria_ia}")

            doc_text = f"Imagen: {descripcion_ia}. Archivo: {nombre_img}"
            id_img = f"img_{idx}_{nombre_img}"
            
            # Guardar en Chroma
            collection.add(
                documents=[doc_text],
                embeddings=[model_emb.encode(f"passage: {doc_text}").tolist()],
                metadatas=[{
                    "source": nombre_img,
                    "category": categoria_ia,
                    "tipo": "imagen",
                    "imagen_path": ruta_absoluta
                }],
                ids=[id_img]
            )
            
            resumen_guardado.append({
                "archivo": nombre_img,
                "categoria": categoria_ia,
                "descripcion": descripcion_ia,
                "ruta": ruta_absoluta
            })

        except Exception as e:
            logger.error(f"Error procesando imagen {ruta_completa}: {e}")

    return resumen_guardado

def crear_chunks_jerarquicos(texto_completo):
    chunks_procesados = []
    chunks_padre = utils.hacer_chunking(texto_completo, chunk_size=TAMANO_PADRE, overlap=200)
    
    for i, texto_padre in enumerate(chunks_padre):
        chunks_hijo = utils.hacer_chunking(texto_padre, chunk_size=TAMANO_HIJO, overlap=SOLAPE_HIJO)
        for texto_hijo in chunks_hijo:
            chunks_procesados.append({
                "texto_vectorizable": texto_hijo,
                "texto_completo_padre": texto_padre,
                "padre_id": i
            })
    return chunks_procesados

# --- MÉTODO PRINCIPAL ---

def main():
    logger.info(f"Iniciando Ingesta Híbrida. Colección: {COLLECTION_NAME}")
    
    # 0. Cargar el JSON de metadatos de referencia
    try:
        with open(ruta_json, 'r', encoding='utf-8') as f:
            metadatos_referencia = json.load(f)
        logger.info(f"✅ Cargados metadatos de referencia desde {ruta_json}")
    except Exception as e:
        logger.error(f"❌ No se pudo cargar el archivo de metadatos JSON: {e}")
        metadatos_referencia = []

    DATA_DIR = utils.project_root() / "data" / "Pdfs"
    archivos_pdf = glob.glob(os.path.join(DATA_DIR, "*.pdf"))

    client = chromadb.PersistentClient(path=DB_DIR)
    try: 
        client.delete_collection(COLLECTION_NAME)
    except: 
        pass
    
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    model_emb = SentenceTransformer(MODELO_EMBEDDINGS)

    # 1. PROCESAR IMÁGENES (Multimodal con Gemini - Se mantiene igual)
    imagenes_resumen = ingestar_imagenes_locales(DIRECTORIO_IMAGENES, collection, model_emb)

    if imagenes_resumen:
        print("\n" + "🖼️  REPORTE DE VISIÓN GEMINI AI")
        print("-" * 120)
        print(f"{'Archivo':<25} | {'Categoría IA':<25} | {'Descripción'}")
        print("-" * 120)
        for img in imagenes_resumen:
            print(f"{img['archivo'][:23]:<25} | {img['categoria']:<25} | {img['descripcion'][:45]}")
        print("-" * 120 + "\n")

    # 2. PROCESAR PDFs (Lógica Híbrida: JSON > Gemini)
    total_chunks = 0
    resumen_pdfs = []

    for archivo in archivos_pdf:
        nombre_archivo = os.path.basename(archivo)
        texto = utils.leer_pdf_markdown(archivo)
        
        # BUSCAR CATEGORÍA EN EL JSON (Prioridad 1)
        # Filtramos el JSON buscando coincidencia en el nombre del archivo
        match = next((m for m in metadatos_referencia if m.get("archivo", "").endswith(nombre_archivo)), None)
        
        if match:
            categoria = match["categoria"]
            logger.info(f"🔗 Vinculado: {nombre_archivo} -> {categoria} (desde JSON)")
        else:
            # FALLBACK A GEMINI (Solo si no está en el JSON para ahorrar cuota)
            logger.warning(f"⚠️ {nombre_archivo} no está en el JSON. Clasificando con Gemini...")
            resultado_pdf = clasificar_con_gemini(nombre_archivo, texto, es_imagen=False)
            categoria = resultado_pdf["categoria"]
        
        print(f"📄 PDF: {nombre_archivo} -> Categoría: {categoria}")
        resumen_pdfs.append((nombre_archivo, categoria))

        # Chunking e Ingesta de los fragmentos (Hijos)
        items = crear_chunks_jerarquicos(texto)
        textos_hijos = [it["texto_vectorizable"] for it in items]
        
        # Inyectamos la categoría correcta en cada chunk
        metadatas = [{
            "source": nombre_archivo,
            "category": categoria,
            "type": "child",
            "parent_id": it["padre_id"],
            "contexto_expandido": it["texto_completo_padre"]
        } for it in items]
        
        ids = [f"{nombre_archivo}_child_{idx}" for idx in range(len(textos_hijos))]
        textos_con_prefijo = [f"passage: {t}" for t in textos_hijos]
        embeddings = model_emb.encode(textos_con_prefijo).tolist()

        collection.add(
            documents=textos_hijos, 
            embeddings=embeddings,   
            metadatas=metadatas, 
            ids=ids
        )
        total_chunks += len(ids)

    print("\n" + "="*85)
    print("📊 RESUMEN FINAL DE INGESTA (HÍBRIDA)")
    print("="*85)
    for nombre, cat in resumen_pdfs:
        print(f"{nombre[:45]:<45} | {cat}")
    print("="*85)
    logger.info(f"✅ FINALIZADO. Total vectores en Chroma: {total_chunks}")

if __name__ == "__main__":
    main()