#  LocalAI-Assistant

**Private, Offline AI Chatbot powered by Ollama + Streamlit + Whisper + LLaVA + RAG**

![badge](https://img.shields.io/badge/Offline-AI-blue) ![badge](https://img.shields.io/badge/Privacy-100%25-green) ![badge](https://img.shields.io/badge/Framework-Streamlit-orange)

---

##  Overview

**LocalAI-Assistant** is a fully private, offline AI-powered assistant that runs directly on your local machine. It integrates multiple advanced AI models like **Llama, Mistral, DeepSeek, Phi, TinyLlama, and LLaVA**, providing a seamless and privacy-focused experience without any cloud dependency.

###  Key Features

-  **Chat with LLMs (Offline)** — Converse with AI models locally
-  **PDF Summarization & Document Q&A** — Upload documents and interact with them using AI
-  **Voice Input & Output** — Convert speech to text (Whisper) and text to speech (pyttsx3) — fully offline
-  **Image Analysis with LLaVA** — Understand and analyze image content through AI-powered vision models
-  **Chat with Documents via RAG** — Retrieval-Augmented Generation for querying custom knowledge bases
-  **Multi-Chat Memory Management** — Auto-save chats with options to rename, delete, and restore from a recycle bin
-  **100% Local & Private** — No internet required, no data leaves your machine

---

##  Tech Stack

| Component              | Description                                                                  |
| ---------------------- | ---------------------------------------------------------------------------- |
| **Ollama**             | Run LLMs locally (Supports Llama3, Mistral, DeepSeek, Phi, TinyLlama, LLaVA) |
| **Streamlit**          | Web-based user interface for easy interaction                                |
| **LangChain**          | Enables document-based RAG (Retrieval-Augmented Generation)                  |
| **Whisper (Offline)**  | Speech-to-Text model for voice input                                         |
| **pyttsx3 (Offline)**  | Text-to-Speech for voice responses                                           |
| **LLaVA**              | Vision-Language model for AI-powered image analysis                          |
| **Local JSON Storage** | Chat history, knowledge base, and recycle bin management                     |

---


---

##  Installation

### 1️⃣ Install Ollama

- Download and install Ollama from 👉 https://ollama.com/download

- Run Ollama in the background:
```bash
ollama serve
```
### 2️⃣ Clone the Repository:
```bash
git clone https://github.com/your-username/LocalAI-Assistant.git
cd LocalAI-Assistant
```


### 3️⃣ Install Python Dependencies
Create a virtual environment (recommended):

```bash
python -m venv offenv
# Activate it:
offenv\Scripts\activate  # On Windows
source offenv/bin/activate  # On Mac/Linux
```

```bash
pip install -r requirements.txt
```

### 4️⃣ Pull Models via Ollama:
```bash
ollama pull llama3
ollama pull mistral
ollama pull deepseek-coder
ollama pull phi3
ollama pull tinyllama
ollama pull llava
```

### ▶️ How to Run
Run the app:
```bash
streamlit run app.py
```
Open in browser:
```bash
http://localhost:8501
```
