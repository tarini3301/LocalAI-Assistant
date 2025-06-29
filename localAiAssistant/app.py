import streamlit as st
from ollama_api import chat_with_model
from pdf_utils import extract_text_from_pdf
from voice_utils import transcribe_audio, speak
from image_utils import analyze_image_with_llava
from vector_store_utils import create_chunks, create_vector_store, query_document
import tempfile

import json
import os


import streamlit as st
from streamlit.runtime.scriptrunner import RerunException
import streamlit.runtime.scriptrunner.script_runner as script_runner

def rerun():
    raise RerunException("rerun")
#script_runner.RerunException()

# Then call rerun() where you want to restart the app

CHAT_FILE = "chats.json"
RECYCLE_BIN_FILE = "recycle_bin.json"

def load_chats():
    if os.path.exists(CHAT_FILE):
        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        # Return default if no file exists
        return {"Chat 1": []}

def save_chats(chats):
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(chats, f, indent=2)

if "all_chats" not in st.session_state:
    st.session_state.all_chats = load_chats()

if "current_chat" not in st.session_state:
    # Select the first chat by default
    st.session_state.current_chat = list(st.session_state.all_chats.keys())[0]

def update_and_save_chats():
    save_chats(st.session_state.all_chats)


st.set_page_config(page_title="Local AI Assistant", layout="wide")
st.title("🤖 Local AI Assistant (100% Offline)")

# 🔥 Initialize multi-chat memory
if 'all_chats' not in st.session_state:
    st.session_state.all_chats = {}  # Stores all chats as {chat_name: [history]}
if 'current_chat' not in st.session_state:
    st.session_state.current_chat = "Chat 1"
    st.session_state.all_chats["Chat 1"] = []


def load_recycle_bin():
    if os.path.exists(RECYCLE_BIN_FILE):
        try:
            with open(RECYCLE_BIN_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return {}  # empty file → return empty dict
                return json.loads(content)
        except (json.JSONDecodeError, ValueError):
            # File corrupted or invalid JSON
            return {}
    else:
        return {}


def save_recycle_bin():
    with open(RECYCLE_BIN_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.recycle_bin, f, indent=2)

if "recycle_bin" not in st.session_state:
    st.session_state.recycle_bin = load_recycle_bin()



mode = st.sidebar.radio(
    "Choose Mode", ["Chat", "PDF Summarization", "Voice Input", "Chat with Docs", "Image Analysis"]
)

# 🔥 Available models
available_models = ["llama3", "mistral", "phi3", "tinyllama", "deepseek-coder", "llava"]

# 🔥 Generate flag for Enter key handling
if 'generate' not in st.session_state:
    st.session_state.generate = False

def handle_generate():
    st.session_state.generate = True

# 🔥 Function to pick model based on prompt
def get_default_model(prompt):
    prompt_lower = prompt.lower()

    if any(word in prompt_lower for word in ["code", "python", "function", "program", "algorithm"]):
        return "deepseek-coder"
    elif any(word in prompt_lower for word in ["summarize", "summary", "document", "pdf"]):
        return "mistral"
    elif any(word in prompt_lower for word in ["image", "picture", "photo"]):
        return "llava"
    elif len(prompt.split()) <= 3:
        return "phi3"
    else:
        return "llama3"


# 🔥 Global input box to select model
# 🔥 Input box with Enter key trigger
user_input = st.text_input(
    "You:",
    key="main_user_input",
    on_change=handle_generate,
    placeholder="Type your message and press Enter or click Send..."
)


# 🔥 Dynamic Model Detection
if user_input:
    dynamic_default_model = get_default_model(user_input)
else:
    dynamic_default_model = "llama3"

# 🔥 Sidebar for model selection (❌ no markdown shown)
selected_model = st.sidebar.selectbox(
    "Select Model (or use suggested one)",
    available_models,
    index=available_models.index(dynamic_default_model)
)


# Session chat memory
with st.sidebar:
    st.subheader("💬 Chats")

    # Show existing chats in dropdown
    chat_list = list(st.session_state.all_chats.keys())
    if chat_list:
        selected_chat = st.selectbox("Select Chat", chat_list, index=chat_list.index(st.session_state.current_chat))
        st.session_state.current_chat = selected_chat
    else:
        st.session_state.current_chat = "Chat 1"
        st.session_state.all_chats["Chat 1"] = []

    # Button to start a new chat
    if st.button("➕ Start New Chat"):
        new_chat_name = f"Chat {len(st.session_state.all_chats) + 1}"
        st.session_state.all_chats[new_chat_name] = []
        st.session_state.current_chat = new_chat_name
        update_and_save_chats() 

    # Button to clear current chat
    # Confirmation flag
    if "confirm_clear" not in st.session_state:
        st.session_state.confirm_clear = False

    # Clear chat with confirmation
    if st.button("🗑️ Clear Current Chat", key="clear_button"):
        st.session_state.confirm_clear = True

    if st.session_state.confirm_clear:
        st.warning("⚠️ Are you sure you want to clear the current chat? This action cannot be undone.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes, Clear", key="confirm_clear_button"):
                st.session_state.all_chats[st.session_state.current_chat] = []
                update_and_save_chats() 
                st.success("Chat cleared.")
                st.session_state.confirm_clear = False  # Reset confirmation
        with col2:
            if st.button("❌ Cancel", key="cancel_clear_button"):
                st.info("Clear chat cancelled.")
                st.session_state.confirm_clear = False  # Reset confirmation

    # 🔥 Rename Chat Block
    st.subheader("✏️ Rename Chat")
    new_name = st.text_input("Enter new chat name", value=st.session_state.current_chat)
    if st.button("Rename"):
        if new_name and new_name != st.session_state.current_chat:
            if new_name not in st.session_state.all_chats:
                st.session_state.all_chats[new_name] = st.session_state.all_chats.pop(st.session_state.current_chat)
                st.session_state.current_chat = new_name
                update_and_save_chats() 
                st.success(f"Renamed to {new_name}")
            else:
                st.warning("A chat with this name already exists.")



    # Delete chat confirmation flag
    if "confirm_delete" not in st.session_state:
        st.session_state.confirm_delete = False

    # Button to trigger delete confirmation
    if st.button("Delete", key="confirm_delete_button"):
        if len(st.session_state.all_chats) > 1:
            deleted_chat = st.session_state.current_chat

            # Move to recycle bin instead of permanent deletion
            st.session_state.recycle_bin[deleted_chat] = st.session_state.all_chats.pop(deleted_chat)

            # Switch to another chat automatically
            st.session_state.current_chat = list(st.session_state.all_chats.keys())[0]

            update_and_save_chats()
            save_recycle_bin()

            st.success(f"Moved '{deleted_chat}' to Recycle Bin")
        else:
            st.warning("Cannot delete the only remaining chat.")

        st.session_state.confirm_delete = False


with st.sidebar.expander("🗑️ Recycle Bin", expanded=False):
    if st.session_state.recycle_bin:
        for chat_name in list(st.session_state.recycle_bin.keys()):
            st.markdown(f"**{chat_name}**")

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button(f"♻️ Restore", key=f"restore_{chat_name}"):
                    st.session_state.all_chats[chat_name] = st.session_state.recycle_bin.pop(chat_name)
                    st.session_state.current_chat = chat_name
                    update_and_save_chats()
                    save_recycle_bin()
                    st.rerun()

            with col2:
                if f"confirm_delete_{chat_name}" not in st.session_state:
                    st.session_state[f"confirm_delete_{chat_name}"] = False

                if not st.session_state[f"confirm_delete_{chat_name}"]:
                    if st.button(f"🗑️ Delete Permanently", key=f"delperm_{chat_name}"):
                        st.session_state[f"confirm_delete_{chat_name}"] = True
                else:
                    st.warning(f"⚠️ Are you sure you want to delete **'{chat_name}'** permanently?")

                    c1, c2 = st.columns([1, 1])
                    with c1:
                        if st.button("✅ Yes, Delete", key=f"yes_del_{chat_name}"):
                            st.session_state.recycle_bin.pop(chat_name)
                            save_recycle_bin()
                            st.success(f"'{chat_name}' deleted permanently.")
                            st.session_state[f"confirm_delete_{chat_name}"] = False
                            st.rerun()

                    with c2:
                        if st.button("❌ Cancel", key=f"cancel_del_{chat_name}"):
                            st.info(f"Cancelled deleting '{chat_name}'.")
                            st.session_state[f"confirm_delete_{chat_name}"] = False
    else:
        st.write("Recycle Bin is empty.")




# 🔥 Current chat history pointer
current_chat_history = st.session_state.all_chats[st.session_state.current_chat]


# Chat Mode
if mode == "Chat":
    send_pressed = st.button("Send")

    if (send_pressed or st.session_state.generate) and user_input:
        # Build full conversation history for context
        conversation = "\n".join(current_chat_history)
        prompt = f"{conversation}\nUser: {user_input}\nAssistant:"

        # Get response from model
        response = chat_with_model(prompt, model=selected_model)

        # Append user and assistant messages to history
        current_chat_history.append(f"User: {user_input}")
        current_chat_history.append(f"Assistant: {response}")
        

        # Display response
        st.success(response)

        update_and_save_chats()

        # Clear input after sending
        st.session_state.user_input = "" 

        st.session_state.generate = False

    


if st.sidebar.checkbox("Show Chat History", True):
    if current_chat_history:
        with st.expander("📝 Chat History", expanded=True):
            for line in current_chat_history:
                if line.startswith("User:"):
                    st.markdown(f"**🧑 {line[5:]}**")
                else:
                    st.markdown(f"**🤖 {line[10:]}**")

# PDF Summarization
elif mode == "PDF Summarization":
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file)
        st.subheader("Extracted Text")
        st.write(text)

        question = st.text_input("Ask a question about the PDF:")
        if st.button("Get Answer"):
            prompt = f"Answer the following question based on this document:\n\n{text}\n\nQuestion: {question}"
            response = chat_with_model(prompt)
            st.success(response)

# Voice Input
elif mode == "Voice Input":
    audio_file = st.file_uploader("Upload audio file", type=["mp3", "wav", "m4a"])
    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(audio_file.read())
            temp_file_path = temp_file.name

        st.info("Transcribing audio...")
        transcribed_text = transcribe_audio(temp_file_path)
        st.subheader("Transcribed Text")
        st.write(transcribed_text)

        if st.button("Send to LLM"):
            response = chat_with_model(transcribed_text)
            st.subheader("LLM Response")
            st.success(response)

        if st.button("🔊 Speak Response"):
            speak(response)

# Chat with Docs (RAG)
elif mode == "Chat with Docs":
    uploaded_file = st.file_uploader("Upload PDF for RAG", type="pdf")
    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file)
        chunks = create_chunks(text)
        vector_store = create_vector_store(chunks)

        question = st.text_input("Ask a question about the document:")
        if st.button("Get Answer"):
            response = query_document(vector_store, question)
            st.success(response)

# Image Analysis
elif mode == "Image Analysis":
    image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if image:
        prompt = st.text_input("Ask about this image:")
        if st.button("Analyze"):
            result = analyze_image_with_llava(image, prompt)
            st.subheader("Response")
            st.success(result)
