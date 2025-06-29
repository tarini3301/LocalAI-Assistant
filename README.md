#  LocalAI-Assistant

**Private, Offline AI Chatbot powered by Ollama + Streamlit + Whisper + LLaVA + RAG**

![badge](https://img.shields.io/badge/Offline-AI-blue) ![badge](https://img.shields.io/badge/Privacy-100%25-green) ![badge](https://img.shields.io/badge/Framework-Streamlit-orange)

---

##  Overview

**LocalAI-Assistant** is a fully private, offline AI-powered assistant that runs on your local machine. It supports multiple AI models like **Llama, Mistral, DeepSeek, Phi, TinyLlama**, and offers features such as:

-  **Chat with LLMs (Offline)**
-  **PDF Summarization**
-  **Voice Input & Output (Speech-to-Text & Text-to-Speech)**
-  **Image Analysis with LLaVA**
-  **Chat with Documents using RAG (Retrieval-Augmented Generation)**
-  **Multi-Chat Memory (Auto Save) with Rename, Delete, Recycle Bin with Restore**
-  **No Internet Required. 100% Local & Private**

---

##  Tech Stack

-  **Ollama** — Run LLMs like Llama3, Mistral, DeepSeek, Phi, TinyLlama, LLaVA
-  **Streamlit** — For the web-based UI
-  **LangChain** — For Document RAG (Chat with Docs)
-  **Whisper** — Speech to Text (Offline)
-  **pyttsx3** — Text to Speech (Offline)
-  **LLaVA** — Vision-based LLM for Image Analysis
-  **JSON-based Local Storage** — For chat memory persistence

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
python -m venv venv
# Activate it:
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Mac/Linux
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
