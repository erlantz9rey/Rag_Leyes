
import streamlit as st    
import requests           
import io
from audio_recorder_streamlit import audio_recorder 
from openai import OpenAI
import whisper
import tempfile
import os
import speech_recognition as sr
import time
import base64
from fpdf import FPDF
from st_copy_to_clipboard import st_copy_to_clipboard
import PyPDF2
import edge_tts
import asyncio
# ============================================================================
# CONFIGURACION E ICONOS
# ============================================================================
# Direccion donde esta el backend (API)
API_URL = "http://localhost:8000"
base_path = os.path.dirname(__file__)
# --- CONFIGURACIÓN DE ICONOS ---
ICONO_USUARIO = os.path.join(base_path, "IconoUsuario.png")
ICONO_ASISTENTE = os.path.join(base_path, "IconoRag.png")


# Configuracion de la pagina web
st.set_page_config(
    page_title="IA Asistente Legal ",  
    page_icon="⚖️",                     
    layout="centered"                  
)

r = sr.Recognizer()

def boton_copiar(texto, key):
    st_copy_to_clipboard(texto, key=key)

# Función para convertir imagen a Base64

def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')


with st.sidebar:
    st.title("Panel de Control")
    st.markdown("---")
    try:
        res = requests.get(f"{API_URL}/health", timeout=2)
        if res.status_code == 200:
            st.success("Conectado a la API")
        else:
            st.error("Error en API")
    except:
        st.error("API No detectada")

    # SUBIDA DE IMÁGENES
    imagen_subida = st.file_uploader("Sube una foto de una imagen o documento", type=["jpg", "jpeg", "png","pdf"])


def texto_a_voz(texto, key):
    # Limpiamos el texto de caracteres que rompan el JavaScript
    texto_escapado = texto.replace('"', '\\"').replace("'", "\\'").replace("\n", " ")
    
    html_code = f"""
    <div id="container_{key}">
        <button id="btn_{key}" style="
            display: inline-flex; align-items: center; justify-content: center;
            background-color: #ffffff; color: #31333F;
            border: 1px solid rgba(49, 51, 63, 0.2); border-radius: 0.5rem;
            padding: 0.25rem; font-size: 1rem; cursor: pointer;
            width: 45px; height: 45px; transition: all 0.2s;">
            🔊
        </button>
    </div>
    <script>
        (function() {{
            const btn = document.getElementById('btn_{key}');
            let msg = null;

            function resetBtn() {{
                btn.innerHTML = "🔊";
                btn.style.color = "#31333F";
                btn.style.borderColor = "rgba(49, 51, 63, 0.2)";
            }}

            btn.addEventListener('click', function() {{
                // Si ya está hablando, lo cancelamos y reseteamos el botón
                if (window.speechSynthesis.speaking) {{
                    window.speechSynthesis.cancel();
                    resetBtn();
                }} else {{
                    // Creamos la instancia de voz justo al hacer click
                    msg = new SpeechSynthesisUtterance("{texto_escapado}");
                    msg.lang = 'es-ES';
                    
                    msg.onend = function() {{
                        resetBtn();
                    }};

                    // Cambiamos aspecto a "Stop"
                    btn.innerHTML = "⏹";
                    btn.style.color = "#ff4b4b";
                    btn.style.borderColor = "#ff4b4b";
                    
                    window.speechSynthesis.speak(msg);
                }}
            }});
        }})();
    </script>
    """
    st.components.v1.html(html_code, height=60)


def enviarPregunta(pregunta, imagen_data=None, modalidad="Rapido"):
    # Creamos el diccionario del mensaje
    nuevo_mensaje = {"role": "user", "content": pregunta}
   
    # Si hay imagen, la guardamos solo en este mensaje
    if imagen_data is not None:
        nuevo_mensaje["image"] = imagen_data

    st.session_state.mensajes.append(nuevo_mensaje)

    # Renderizar el mensaje inmediatamente para el usuario
    with st.chat_message("user", avatar=ICONO_USUARIO):
        st.markdown(pregunta)
        if imagen_data:
           st.image(imagen_data, width=300)


    # 2. Llamar a la API
    with st.chat_message("assistant", avatar=ICONO_ASISTENTE):
        placeholder = st.empty()
        with st.status("Pensando la respuesta ...", expanded=True) as status:
            #placeholder.markdown("💬 El asistente está escribiendo")
            try:
                # Preparar imagen si existe
                image_b64 = None
                if imagen_subida:
                    # Resetear el puntero del archivo por si se leyó antes
                    imagen_subida.seek(0)
                    image_b64 = encode_image(imagen_subida)
                # Historial limpio para la API
                historial_api = [{"role": m["role"], "content": m["content"]} for m in st.session_state.mensajes]
                # LLAMADA AL BACKEND (Timeout de 300s para Ollama)
                if imagen_data and hasattr(imagen_data, 'type') and imagen_data.type != "application/pdf":
                    imagen_data.seek(0)
                    image_b64 = encode_image(imagen_data)
                # LLAMADA AL BACKEND
                resp = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "pregunta": pregunta,
                        "image_base64": image_b64,
                        "historial": historial_api,
                        "modalidad": modalidad
                    },
                    timeout=300
                )

                if resp.status_code == 200:
                    data = resp.json()
                    texto_completo = data.get("respuesta", "")
                    fuentes = data.get("fuentes", [])
                    tiempo = data.get("tiempo_segundos", 0)

                    # 1. FUNCIÓN DE STREAMING (Tu lógica actual)
                    def stream_data():
                        for word in texto_completo.split(" "):
                            yield word + " "
                            time.sleep(0.02)

                    # 2. MOSTRAR EL TEXTO CON EFECTO (Escribe en pantalla)
                    texto_limpio = texto_completo.split("IMAGEN_PATH:")[0]
                    respuesta_final = st.write_stream(stream_data)

                    # 3. NUEVA LÓGICA: MOSTRAR IMÁGENES DE LA BASE DE DATOS
                    # Extraemos las rutas de las imágenes que vienen en las fuentes del backend
                    imagenes_encontradas = [f for f in fuentes if f.get("tipo") == "imagen"]
                    if imagenes_encontradas:
                        st.markdown("### 📄 Evidencia Visual:")
                        cols = st.columns(len(imagenes_encontradas))
                        for idx, img_info in enumerate(imagenes_encontradas):
                            with cols[idx]:
                                # 1. Intentamos usar la URL directa del backend
                                url = img_info.get("url")
                                
                                # 2. Si no hay URL, la construimos (Plan B)
                                if not url:
                                    url = f"{API_URL}/imagenes/{img_info.get('archivo')}"
                                
                                # Mostramos la imagen usando la URL
                                st.image(url, caption=img_info.get("archivo"), use_container_width=True)


                    # 4. BOTONES DE ACCIÓN (Copiar y Voz)
                    col_c1, col_c2 = st.columns([0.2, 0.8])
                    with col_c1:
                        boton_copiar(texto_completo, key="temp_copy_flow")
                    with col_c2:
                        texto_a_voz(texto_completo, key=f"voice_stream_{int(time.time())}")
                    
                    status.update(label="✅ Consulta completada", state="complete", expanded=True)
                    # Fuentes (tu lógica original)
                    if fuentes:
                        with st.expander("Fuentes consultadas"):
                            for src in fuentes:
                                archivo = src.get('archivo', '?')
                                chunk_id = src.get('chunk_id', '')
                                score = src.get('score', None)
                                texto_chunk = src.get('texto', '')
                                relevante = src.get('relevante', None)
                               
                                if relevante is True:
                                    info = f" **{archivo}**"
                                elif relevante is False:
                                    info = f" **{archivo}** _(descartado)_"
                                else:
                                    info = f" **{archivo}**"
                             
                                if chunk_id: info += f" (chunk: {chunk_id})"
                                if score is not None: info += f" | Score: {score:.3f}"
                                st.markdown(info)
                                if texto_chunk: st.caption(f"_{texto_chunk}_")
                    # Debug Self-RAG 
                    debug_info = data.get("debug_info", {})
                    if debug_info:
                        with st.expander(" Debug Self-RAG"):
                            graders = debug_info.get("graders", {})
                            if graders:
                                st.markdown(f"**Docs recuperados:** {graders.get('docs_recuperados', '?')}")
                                st.markdown(f"**Docs aprobados:** {graders.get('docs_aprobados', '?')}")
                                if graders.get('intentos_totales') != "?":
                                    st.markdown(f"**Intentos de generación:** {graders.get('intentos_totales')}")
                                exito = graders.get("exito", None)
                                if exito is True: st.success(" Respuesta verificada")
                                elif exito is False: st.warning(" No se pudo verificar la respuesta")
                    # Guardar en el historial para persistencia
                    st.session_state.mensajes.append({
                        "role": "assistant",
                        "content": texto_completo,
                        "fuentes": fuentes,
                        # Guardamos la lista de rutas directamente
                        # Verifica que en el append del historial estés usando la clave correcta:
                        "imagenes_recuperadas": [f["imagen_path"] for f in fuentes if f.get("tipo") == "imagen" and f.get("imagen_path")] 
                    })
                else:
                    placeholder.error(f"Error {resp.status_code}: {resp.text}")
            except Exception as e:
                placeholder.error(f"Error de conexión: {e}")
                st.info("¿Has encendido la API?")

# ============================================================================
# CHAT PRINCIPAL
# ============================================================================

BIENVENIDA = """⚖️ **¡Bienvenido a tu Asistente Legal !**
Estoy aquí para ayudarte a consultar leyes, reglamentos y jurisprudencia de forma rápida.
**Nota importante:** Soy una inteligencia artificial, no soy un abogado.
La información proporcionada es con fines informativos y no constituye un asesoramiento legal vinculante.
 Para casos críticos, consulta siempre con un profesional colegiado."""

if "mensajes" not in st.session_state:
    st.session_state.mensajes = [{"role": "assistant", "content": BIENVENIDA}]

st.title("IA Asistente Legal ⚖️")
st.caption("Pregunta lo que necesites sobre Temas Legales.")

# --- PASO 1: PINTAR EL HISTORIAL CON ICONOS PERSONALIZADOS ---
for i, msg in enumerate(st.session_state.mensajes):
    icono_actual = ICONO_ASISTENTE if msg["role"] == "assistant" else ICONO_USUARIO
    
    with st.chat_message(msg["role"], avatar=icono_actual):
        if msg["content"] == BIENVENIDA:
            st.markdown(msg["content"])
        
        elif msg["role"] == "assistant":
            # TODO esto queda dentro de la respuesta del asistente
            with st.expander("✅ Respuesta del Asistente", expanded=True):
                st.markdown(msg["content"])
                
                # Imágenes recuperadas (dentro de la respuesta)
                if "imagenes_recuperadas" in msg:
                    for img_p in msg["imagenes_recuperadas"]:
                        if img_p and os.path.exists(img_p):
                            st.image(img_p, width=400)

                # Botones de utilidad (dentro de la respuesta)
                c1, c2 = st.columns([0.15, 0.85])
                with c1:
                    boton_copiar(msg["content"], key=f"copy_hist_{i}")
                with c2:
                    texto_a_voz(msg["content"], key=f"voice_hist_{i}")

                # --- FUENTES DENTRO DE LA RESPUESTA ---
                if "fuentes" in msg and msg["fuentes"]:
                    st.divider() # Una línea sutil para separar texto de fuentes
                    with st.expander("📚 Fuentes consultadas"):
                        for src in msg["fuentes"]:
                            archivo = src.get('archivo', 'Documento desconocido')
                            chunk_id = src.get('chunk_id', '')
                            texto_chunk = src.get('texto', '')
                            
                            st.write(f"📄 **{archivo}** {f'(ID: {chunk_id})' if chunk_id else ''}")
                            if texto_chunk:
                                st.caption(f"Fragmento: {texto_chunk[:200]}...") 
                            st.divider()

        else: # Mensaje del Usuario
            st.markdown(msg["content"])
            if "image" in msg and msg["image"] is not None:
                st.image(msg["image"], width=300)

# --- PASO 2: CAPTURAR LA PREGUNTA ---
# Inicializamos el rastreador de imágenes procesadas si no existe
if "imagenes_procesadas" not in st.session_state:
    st.session_state.imagenes_procesadas = set()

def extraer_texto_pdf(archivo_pdf):
    texto_extraido = ""
    try:
        reader = PyPDF2.PdfReader(archivo_pdf)
        for page in reader.pages:
            texto_extraido += page.extract_text() + "\n"
        return texto_extraido
    except Exception as e:
        return f"Error al leer PDF: {e}"

# --- MODIFICACIÓN EN LA CAPTURA DE LA PREGUNTA ---
if pregunta := st.chat_input("Escribe tu cuestion legal..."):
    imagen_a_enviar = None

    if imagen_subida and imagen_subida.name not in st.session_state.imagenes_procesadas:
        if imagen_subida.type == "application/pdf":
            texto_pdf = extraer_texto_pdf(imagen_subida)
            # Unimos el texto del PDF con la pregunta en un solo string
            pregunta = f"DOCUMENTO PDF:\n{texto_pdf}\n\nPREGUNTA: {pregunta}"
            # IMPORTANTE: Forzamos que no se envíe nada como imagen
            imagen_a_enviar = None
        else:
            imagen_a_enviar = imagen_subida  
        st.session_state.imagenes_procesadas.add(imagen_subida.name)
    enviarPregunta(pregunta, imagen_data=imagen_a_enviar, modalidad=st.session_state.opcion)

def generar_pdf(mensajes):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Título del documento
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Historial de Consulta Legal", ln=True, align='C')
    pdf.ln(10) # Salto de línea

    for msg in mensajes:
        role = "Asistente" if msg["role"] == "assistant" else "Usuario"
        content = msg["content"]

        # Configurar color y estilo según el rol
        if role == "Usuario":
            pdf.set_font("Arial", 'B', 12)
            pdf.set_text_color(0, 51, 102)
        else:
            pdf.set_font("Arial", 'B', 12)
            pdf.set_text_color(34, 139, 34)

        pdf.cell(0, 10, txt=f"{role}:", ln=True)

        # Contenido del mensaje (Normal)
        pdf.set_font("Arial", '', 11)
        pdf.set_text_color(0, 0, 0) # Negro
        clean_text = content.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 7, txt=clean_text)
        pdf.ln(5)
    return pdf.output(dest='S').encode('latin-1')

# --- BARRA LATERAL ---
with st.sidebar:

    with st.expander(f"{st.session_state.get('opcion', 'Rapido')}"):
        opcion = st.radio(
            "Elige la modalidad:",
            ["Rapido", "Premium"],
            key="opcion",
            label_visibility="collapsed"
        )
    col_voz, col_img, col_vacia = st.columns([0.2, 0.2, 0.6])
    with col_voz:
        # Tu componente de audio existente
        audio_bytes = audio_recorder(
            text="",
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
            icon_name="microphone",
            icon_size="2x",
        )

    st.markdown("---")
    if st.session_state.mensajes:
        pdf_bytes = generar_pdf(st.session_state.mensajes)
        st.download_button(
            label="📥 Descargar Historial (PDF)",
            data=pdf_bytes,
            file_name="historial_legal.pdf",
            mime="application/pdf",
    )

    if st.button("Borrar Historial"):
        st.session_state.mensajes = [{"role": "assistant", "content": BIENVENIDA}]
        st.rerun()

# --- LÓGICA DE AUDIO ---
if audio_bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    with sr.AudioFile(tmp_path) as source:
        audio = r.record(source)
    try:
        pregunta_voz = r.recognize_google(audio, language="es-ES")
        enviarPregunta(pregunta_voz, modalidad=st.session_state.opcion)
    except sr.UnknownValueError:
        st.error("No se pudo entender el audio")
    except sr.RequestError as e:
        st.error(f"Error del servicio: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)






