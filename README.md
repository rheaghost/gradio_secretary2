
python -m venv secretary_web
pip install -r requirements.txt

On Windows (Command Prompt):

bash
secretary_web\Scripts\activate.bat
On Windows (PowerShell):

bash
secretary_web\Scripts\Activate.ps1
On macOS/Linux:

bash
source secretary_web/bin/activate
Command	Purpose
python -m venv secretary_web	Creates the virtual environment
secretary_web\Scripts\activate	Activates it (Windows)
pip install -r requirements.txt	Installs all dependencies


📄 requirements.txt
Create a file named requirements.txt in your project directory with this content:

txt
# Core Web Framework
gradio>=4.0.0

# AI & LLM
ollama>=0.1.0
sentence-transformers>=2.2.0
chromadb>=0.4.0
PyPDF2>=3.0.0

# Web & Networking
requests>=2.31.0
beautifulsoup4>=4.12.0
yt-dlp>=2023.0.0
whisper>=1.1.0

# Image Processing
Pillow>=10.0.0
opencv-python>=4.8.0
numpy>=1.24.0

# Voice (TTS)
pyttsx3>=2.90

# Utilities
python-dotenv>=1.0.0
Install the Requirements
After activating your virtual environment, run:

bash
pip install -r requirements.txt
📋 Summary
Command	Purpose
python -m venv secretary_web	Creates the virtual environment
secretary_web\Scripts\activate	Activates it (Windows)
pip install -r requirements.txt	Installs all dependencies
You are now ready to run your Gradio web secretary. 🟢🐍📦🌐💻💚


requirements.txt:
# Core Web Framework
gradio>=4.0.0

# AI & LLM
ollama>=0.1.0
sentence-transformers>=2.2.0
chromadb>=0.4.0
PyPDF2>=3.0.0

# Web & Networking
requests>=2.31.0
beautifulsoup4>=4.12.0
yt-dlp>=2023.0.0
whisper>=1.1.0

# Image Processing
Pillow>=10.0.0
opencv-python>=4.8.0
numpy>=1.24.0

# Voice (TTS)
pyttsx3>=2.90

# Utilities
python-dotenv>=1.0.0
