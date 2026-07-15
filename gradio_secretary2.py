import gradio as gr
import time
import os
import threading
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
from ollama import Client
from PyPDF2 import PdfReader
import yt_dlp
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import numpy as np
from PIL import Image
import io
import pyttsx3

# --- Optional imports with fallbacks ---
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# --- Configuration ---
client = Client(host='http://localhost:11434')
speaker_on = True

# --- RAG Setup ---
rag_model = None
rag_collection = None
rag_persist_dir = "./chroma_db"

# --- Memory ---
chat_history = []
MEMORY_LIMIT = 10

# --- Helper Functions ---
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def speak(text):
    """Speak text in a background thread to avoid blocking the main program."""
    if not speaker_on:
        return

    def _speak():
        try:
            engine = pyttsx3.init()
            try:
                voice_id = r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
                engine.setProperty('voice', voice_id)
            except:
                pass
            engine.setProperty('rate', 170)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"⚠️ TTS Error: {e}")
        finally:
            try:
                engine.stop()
            except:
                pass

    thread = threading.Thread(target=_speak)
    thread.daemon = True
    thread.start()

# --- RAG Functions ---
def load_rag_model():
    global rag_model
    if rag_model is None:
        rag_model = SentenceTransformer('all-MiniLM-L6-v2')
    return rag_model

def get_rag_collection():
    global rag_collection
    if rag_collection is None:
        try:
            client_db = chromadb.PersistentClient(path=rag_persist_dir)
            rag_collection = client_db.get_collection("documents")
        except:
            rag_collection = None
    return rag_collection

def index_pdf(file):
    global rag_collection, rag_model
    if file is None:
        return "No file uploaded."

    try:
        reader = PdfReader(file.name)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text()

        chunk_size = 500
        chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

        model = load_rag_model()
        embeddings = model.encode(chunks)

        client_db = chromadb.PersistentClient(path=rag_persist_dir)
        try:
            collection = client_db.get_collection("documents")
            next_id = collection.count()
            ids = [str(next_id + i) for i in range(len(chunks))]
            collection.add(
                documents=chunks,
                embeddings=embeddings.tolist(),
                ids=ids
            )
            rag_collection = collection
            return f"✅ Added {len(chunks)} chunks to existing index."
        except:
            collection = client_db.create_collection("documents")
            collection.add(
                documents=chunks,
                embeddings=embeddings.tolist(),
                ids=[str(i) for i in range(len(chunks))]
            )
            rag_collection = collection
            return f"✅ Created new index with {len(chunks)} chunks."
    except Exception as e:
        return f"❌ Error: {e}"

def ask_document(query):
    global rag_collection, rag_model
    if rag_collection is None:
        rag_collection = get_rag_collection()
        if rag_collection is None:
            return "⚠️ No document indexed. Please upload a PDF first."

    try:
        model = load_rag_model()
        query_embedding = model.encode([query])
        results = rag_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=3
        )
        context = "\n\n".join(results['documents'][0])
        prompt = f"Answer the following question based only on the provided context.\n\nContext:\n{context}\n\nQuestion: {query}"
        response = client.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
        answer = response['message']['content']
        if speaker_on:
            speak(answer)
        return answer
    except Exception as e:
        return f"❌ Error: {e}"

def clear_rag_index():
    global rag_collection
    try:
        client_db = chromadb.PersistentClient(path=rag_persist_dir)
        client_db.delete_collection("documents")
        rag_collection = None
        return "🗑️ RAG index cleared."
    except:
        return "⚠️ No index to clear."

# --- Web Summary ---
def summarize_web(url):
    if not url.startswith('http'):
        return "⚠️ Invalid URL. Must start with http or https."

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return f"❌ Site blocked. Status: {response.status_code}"

        soup = BeautifulSoup(response.content, 'html.parser')
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        clean_text = text[:3000]

        prompt = f"Summarize the following webpage content in 3-5 sentences:\n\n{clean_text}"
        response = client.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
        answer = response['message']['content']
        if speaker_on:
            speak(answer)
        return answer
    except Exception as e:
        return f"❌ Error: {e}"

# --- YouTube Audio Summary ---
def summarize_youtube_audio(url):
    if not WHISPER_AVAILABLE:
        return "❌ Whisper not installed. Please install: pip install openai-whisper"

    if not url.strip():
        return "⚠️ No URL entered."

    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': 'audio.%(ext)s',
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        model = whisper.load_model("base")
        result = model.transcribe("audio.mp3")
        transcript = result["text"]

        prompt = f"Summarize this YouTube video transcript in a few sentences:\n\n{transcript}"
        response = client.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
        answer = response['message']['content']
        if speaker_on:
            speak(answer)
        return answer
    except Exception as e:
        return f"❌ Error: {e}"

# --- Image Analysis (File, URL, Webcam) ---
def analyze_image_file(file):
    if file is None:
        return "No image uploaded."

    try:
        with open(file.name, 'rb') as img_file:
            response = client.generate(
                model='llava',
                prompt="Describe this image in detail.",
                images=[img_file.read()]
            )
            answer = response['response']
            if speaker_on:
                speak(answer)
            return answer
    except Exception as e:
        return f"❌ Error: {e}"

def analyze_image_url(url):
    if not url.startswith('http'):
        return "⚠️ Invalid URL. Must start with http or https."

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10, stream=True)
        if response.status_code != 200:
            return f"❌ Failed to download image. Status: {response.status_code}"

        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image'):
            return f"⚠️ The URL does not point to an image. Content-Type: {content_type}"

        img_bytes = response.content
        response = client.generate(
            model='llava',
            prompt="Describe this image in detail.",
            images=[img_bytes]
        )
        answer = response['response']
        if speaker_on:
            speak(answer)
        return answer
    except Exception as e:
        return f"❌ Error: {e}"

def analyze_image_webcam(image):
    if image is None:
        return "No image captured."

    try:
        pil_image = Image.fromarray(image.astype('uint8'))
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()

        response = client.generate(
            model='llava',
            prompt="Describe this image in detail.",
            images=[img_bytes]
        )
        answer = response['response']
        if speaker_on:
            speak(answer)
        return answer
    except Exception as e:
        return f"❌ Error: {e}"

# --- Chat Function ---
def chat(message, history, sound_on):
    global chat_history
    if not message:
        return ""

    if not any(msg.get('role') == 'system' for msg in chat_history):
        chat_history.append({
            'role': 'system',
            'content': "You are a helpful, professional secretary. Respond clearly and directly. Do not use excessive enthusiasm or casual language."
        })

    chat_history.append({'role': 'user', 'content': message})

    if message.lower().startswith("rag:"):
        query = message[4:].strip()
        answer = ask_document(query)
    else:
        try:
            response = client.chat(
                model='llama3',
                messages=chat_history[-MEMORY_LIMIT:]
            )
            answer = response['message']['content']
        except Exception as e:
            answer = f"❌ Error: {e}"

    chat_history.append({'role': 'assistant', 'content': answer})

    if sound_on:
        speak(answer)

    return answer

# --- Gradio UI ---
def respond(message, history, sound_on):
    if not message:
        return "", history

    response = chat(message, history, sound_on)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})

    return "", history

# --- Log Export ---
def export_log():
    log_file = "secretary_permanent_log.txt"
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return content, f"secretary_log_{timestamp}.txt"
    return "No log file found.", None

# --- Voice Input HTML ---
voice_html = """
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
    <button id="start-record" style="padding: 10px 20px; background: #00FF41; color: #000; border: none; border-radius: 5px; cursor: pointer;">
        🎤 Start Recording
    </button>
    <button id="stop-record" style="padding: 10px 20px; background: #FF4444; color: #fff; border: none; border-radius: 5px; cursor: pointer; display: none;">
        ⏹️ Stop
    </button>
    <span id="recording-status" style="color: #00FF41; font-weight: bold;">Idle</span>
</div>
<script>
    const startBtn = document.getElementById('start-record');
    const stopBtn = document.getElementById('stop-record');
    const status = document.getElementById('recording-status');
    let recognition = null;

    if ('webkitSpeechRecognition' in window) {
        recognition = new webkitSpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = true;
        recognition.lang = 'en-US';

        recognition.onstart = function() {
            status.textContent = '🎤 Recording...';
            status.style.color = '#FF4444';
            startBtn.style.display = 'none';
            stopBtn.style.display = 'inline-block';
        };

        recognition.onresult = function(event) {
            let transcript = '';
            for (let i = event.resultIndex; i < event.results.length; i++) {
                if (event.results[i].isFinal) {
                    transcript += event.results[i][0].transcript;
                }
            }
            if (transcript) {
                const textbox = document.querySelector('#msg_textbox textarea');
                if (textbox) {
                    textbox.value = transcript;
                    textbox.dispatchEvent(new Event('input', { bubbles: true }));
                }
            }
        };

        recognition.onend = function() {
            status.textContent = '✅ Done';
            status.style.color = '#00FF41';
            startBtn.style.display = 'inline-block';
            stopBtn.style.display = 'none';
        };

        recognition.onerror = function(event) {
            status.textContent = '❌ Error: ' + event.error;
            status.style.color = '#FF4444';
            startBtn.style.display = 'inline-block';
            stopBtn.style.display = 'none';
        };
    } else {
        status.textContent = '❌ Speech recognition not supported in this browser.';
    }

    startBtn.onclick = function() {
        if (recognition) {
            try {
                recognition.start();
            } catch(e) {
                status.textContent = '❌ Error: ' + e.message;
            }
        }
    };

    stopBtn.onclick = function() {
        if (recognition) {
            recognition.stop();
        }
    };
</script>
"""

# --- Build the Gradio Interface ---
with gr.Blocks(title="Secretary 2026", theme=gr.themes.Soft(), css="""
    .gradio-container { max-width: 900px !important; margin: auto !important; }
    .webcam-container { display: flex; flex-direction: column; align-items: center; }
    .webcam-container .gr-image { max-width: 400px !important; margin: auto !important; }
    .webcam-btn { margin-top: 10px !important; }
""") as demo:

    # Top row with title and sound toggle
    with gr.Row():
        gr.Markdown("# 📄 Secretary 2026")
        sound_toggle = gr.Checkbox(label="🔊 Sound", value=True)

    # Voice input section
    with gr.Row():
        gr.HTML(voice_html)

    with gr.Tabs():
        # Tab 1: Chat
        with gr.TabItem("💬 Chat"):
            chatbot = gr.Chatbot(height=400)
            with gr.Row():
                msg = gr.Textbox(label="Ask your secretary...", placeholder="Type or speak your message...", elem_id="msg_textbox", scale=4)
                send_btn = gr.Button("📤 Send", scale=1)
            clear = gr.Button("🗑️ Clear Chat")

            send_btn.click(respond, [msg, chatbot, sound_toggle], [msg, chatbot])
            msg.submit(respond, [msg, chatbot, sound_toggle], [msg, chatbot])
            clear.click(lambda: [], None, [chatbot])

        # Tab 2: RAG
        with gr.TabItem("📄 RAG (PDF Q&A)"):
            gr.Markdown("Upload a PDF and ask questions about it.")
            with gr.Row():
                pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"])
                index_btn = gr.Button("📥 Index PDF")
                index_output = gr.Textbox(label="Index Status", interactive=False)
            with gr.Row():
                rag_query = gr.Textbox(label="Ask a question about the PDF")
                rag_btn = gr.Button("🔍 Ask")
                rag_output = gr.Textbox(label="Answer", interactive=False)

            index_btn.click(index_pdf, [pdf_file], [index_output])
            rag_btn.click(ask_document, [rag_query], [rag_output])

        # Tab 3: Web Summary
        with gr.TabItem("🌐 Web Summary"):
            gr.Markdown("Enter a URL to summarize.")
            with gr.Row():
                web_url = gr.Textbox(label="URL", placeholder="https://example.com")
                web_btn = gr.Button("📄 Summarize")
            web_output = gr.Textbox(label="Summary", interactive=False)

            web_btn.click(summarize_web, [web_url], [web_output])

        # Tab 4: YouTube Audio
        with gr.TabItem("🎥 YouTube Audio"):
            gr.Markdown("Enter a YouTube URL to transcribe and summarize.")
            with gr.Row():
                yt_url = gr.Textbox(label="YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
                yt_btn = gr.Button("📝 Summarize")
            yt_output = gr.Textbox(label="Summary", interactive=False)

            yt_btn.click(summarize_youtube_audio, [yt_url], [yt_output])

        # Tab 5: Image Analysis (File, URL, Webcam) - Improved Layout
        with gr.TabItem("🖼️ Image Analysis"):
            gr.Markdown("Analyze an image from a file, a URL, or your webcam.")
            with gr.Tabs():
                with gr.TabItem("📂 Upload File"):
                    with gr.Row():
                        img_file = gr.File(label="Upload Image", file_types=[".jpg", ".jpeg", ".png", ".bmp"])
                        file_btn = gr.Button("🔍 Analyze")
                    file_output = gr.Textbox(label="Description", interactive=False)
                    file_btn.click(analyze_image_file, [img_file], [file_output])

                with gr.TabItem("🔗 Image URL"):
                    with gr.Row():
                        img_url = gr.Textbox(label="Image URL", placeholder="https://example.com/image.jpg")
                        url_btn = gr.Button("🔍 Analyze")
                    url_output = gr.Textbox(label="Description", interactive=False)
                    url_btn.click(analyze_image_url, [img_url], [url_output])

                with gr.TabItem("📸 Webcam"):
                    gr.Markdown("Click 'Start Webcam' below, then 'Capture & Analyze'.")
                    with gr.Row():
                        webcam_input = gr.Image(sources=["webcam"], label="Webcam Feed", height=300, width=400)
                    with gr.Row():
                        webcam_btn = gr.Button("📸 Capture & Analyze", variant="primary")
                    webcam_output = gr.Textbox(label="Description", interactive=False)
                    webcam_btn.click(analyze_image_webcam, [webcam_input], [webcam_output])

        # Tab 6: Logs
        with gr.TabItem("📋 Logs"):
            gr.Markdown("Download the conversation log.")
            with gr.Row():
                log_output = gr.Textbox(label="Log Content", lines=20, interactive=False, scale=4)
                download_btn = gr.Button("📥 Download Log", scale=1)
            download_btn.click(export_log, outputs=[log_output, gr.File()])

        # Tab 7: Settings
        with gr.TabItem("⚙️ Settings"):
            gr.Markdown("### Memory & Configuration")
            gr.Markdown(f"**Memory Limit:** {MEMORY_LIMIT} messages")
            clear_rag_btn = gr.Button("🗑️ Clear RAG Index")
            clear_rag_output = gr.Textbox(label="Status", interactive=False)
            clear_rag_btn.click(clear_rag_index, [], [clear_rag_output])

# --- Run ---
if __name__ == "__main__":
    demo.launch(share=True)
