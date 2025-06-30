import streamlit as st
from ollama_api import chat_with_model
from pdf_utils import extract_text_from_pdf
from voice_utils import transcribe_audio, speak
from image_utils import analyze_image_with_llava
from vector_store_utils import create_chunks, store_data, search_data, build_knowledge_base, query_knowledge_base
from vector_store_utils import create_chunks, create_vector_store, query_knowledge_base

db = create_vector_store()

# Now you can use db to add/query manually if needed

import tempfile

import json
import os


import streamlit as st
from streamlit.runtime.scriptrunner import RerunException
import streamlit.runtime.scriptrunner.script_runner as script_runner

from vector_store_utils import store_data, search_data


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
    "Choose Mode", ["Chat", "PDF Summarization", "Voice Input", "Chat with Docs", "Image Analysis", "Knowledge Base"]
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


# PDF Summarization
elif mode == "PDF Summarization":

    import fitz  # PyMuPDF

    def extract_text_and_links_from_pdf(uploaded_file):
        text = ""
        links = []

        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page_num, page in enumerate(doc, start=1):
                text += page.get_text()

                for link in page.get_links():
                    if link["uri"]:
                        links.append({"page": page_num, "url": link["uri"]})

        return text, links


    uploaded_file = st.file_uploader("Upload PDF", type="pdf", key="pdf_uploader")

    if uploaded_file:
        # ✅ Extract text and links
        text, links = extract_text_and_links_from_pdf(uploaded_file)

        # ✅ Display extracted text
        st.subheader("📑 Extracted Text")
        with st.expander("Show Extracted Text"):
            st.write(text)

        # ✅ Display extracted links
        st.subheader("🔗 Extracted Links")
        if links:
            for link in links:
                st.markdown(f"- [Page {link['page']}]({link['url']}) → {link['url']}")
        else:
            st.info("No links found in the document.")

        # ✅ Question input with Enter and Send support
        st.subheader("💬 Ask a question or type 'summarize'")

        st.text_input(
            "Your question (or type 'summarize'):",
            key="pdf_question",
            on_change=lambda: handle_pdf_query(st.session_state.pdf_question, text, selected_model)
        )

        # ✅ Query handler function
        def handle_pdf_query(question, text, selected_model):
            if not question.strip():
                st.warning("❗ Please enter a question or 'summarize'.")
                return

            if question.lower().strip() == "summarize":
                prompt = f"Summarize the following document:\n\n{text}\n\nSummary:"
            else:
                prompt = f"Answer the following question based on this document:\n\n{text}\n\nQuestion: {question}\nAnswer:"

            response = chat_with_model(prompt, model=selected_model)

            st.subheader("💡 Response")
            st.success(response)

            # ✅ Clear input after sending
            st.session_state.pdf_question = ""

        # ✅ Optional Send button
        if st.button("Get Answer / Summarize"):
            handle_pdf_query(st.session_state.pdf_question, text, selected_model)

# Voice Input with Chat on Audio Content
elif mode == "Voice Input":

    import tempfile
    from st_audiorec import st_audiorec


    st.subheader("🎙️ Record or Upload Audio")

    # 🎧 Option 1: Upload Audio
    audio_file = st.file_uploader(
        "Upload audio file (mp3, wav, m4a)", type=["mp3", "wav", "m4a"], key="voice_uploader"
    )

    # 🎙️ Option 2: Record Audio in Browser
    wav_audio_data = st_audiorec()

    # ------------------------------
    # 🔥 Process Audio (if uploaded or recorded)
    if audio_file or wav_audio_data:

        if audio_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(audio_file.read())
                temp_file_path = temp_file.name

        elif wav_audio_data:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(wav_audio_data)
                temp_file_path = temp_file.name

        st.info("🎧 Transcribing audio...")
        transcribed_text = transcribe_audio(temp_file_path)

        st.subheader("📝 Transcribed Text")
        with st.expander("Show Transcribed Text"):
            st.write(transcribed_text)

        # Initialize response state
        if "voice_response" not in st.session_state:
            st.session_state.voice_response = ""

        # 🔘 Send full transcript to LLM
        if st.button("Send to LLM"):
            response = chat_with_model(transcribed_text, model=selected_model)
            st.subheader("🤖 LLM Response")
            st.success(response)
            st.session_state.voice_response = response

        # 🔊 Speak response
        if st.button("🔊 Speak Response"):
            if st.session_state.voice_response:
                speak(st.session_state.voice_response)
            else:
                st.warning("⚠️ No response yet. Please click 'Send to LLM' first.")

        # ------------------------------
        # 🎯 Chat Box for Audio Transcript
        st.subheader("💬 Chat with Audio Content")

        if "audio_chat_history" not in st.session_state:
            st.session_state.audio_chat_history = []

        def handle_audio_chat():
            if st.session_state.audio_user_input.strip() == "":
                st.warning("Please enter a question.")
                return

            conversation = "\n".join(st.session_state.audio_chat_history)
            prompt = f"""You are an assistant. Use the following audio transcript as context:

Transcript:
{transcribed_text}

Chat History:
{conversation}

User: {st.session_state.audio_user_input}
Assistant:"""

            response = chat_with_model(prompt, model=selected_model)

            st.session_state.audio_chat_history.append(f"User: {st.session_state.audio_user_input}")
            st.session_state.audio_chat_history.append(f"Assistant: {response}")

            st.session_state.audio_user_input = ""
            st.success(response)

        # Text input with Enter support
        st.text_input(
            "Type your question about the audio...",
            key="audio_user_input",
            on_change=handle_audio_chat,
        )

        # Optional Send button
        if st.button("Send"):
            handle_audio_chat()

        # Show chat history
        if st.session_state.audio_chat_history:
            with st.expander("📝 Chat History"):
                for msg in st.session_state.audio_chat_history:
                    st.markdown(msg)

        if st.button("Clear Chat History"):
            st.session_state.audio_chat_history = []

    else:
        st.warning("🎙️ Please upload an audio file or use the recorder to start.")




# 🔍 Chat with Docs (RAG)
elif mode == "Chat with Docs":
    st.subheader("📄 Chat with Documents (RAG)")

    uploaded_file = st.file_uploader("📤 Upload PDF for RAG", type="pdf")

    if uploaded_file:
        # ✅ Extract text from PDF
        text = extract_text_from_pdf(uploaded_file)

        # ✅ Split into chunks
        chunks = create_chunks(text)

        # ✅ Store chunks into vector database
        for idx, chunk in enumerate(chunks):
            store_data(doc_id=f"{uploaded_file.name}_chunk_{idx}", text=chunk)

        st.success("✅ Document chunks stored in database for retrieval.")

        # ✅ Input question
        question = st.text_input("💬 Ask a question about the document:")

        if st.button("🔍 Get Answer"):
            # 🔍 Search in vector database
            results = search_data(db, question, top_k=3)

            if results["documents"]:
                combined_context = " ".join(results["documents"][0])

                prompt = f"""You are an assistant. Use the following context to answer the question.

Context:
{combined_context}

Question: {question}
Answer:"""

                # 🔥 Query the LLM
                response = chat_with_model(prompt, model=selected_model)

                st.subheader("💡 Response")
                st.success(response)
            else:
                st.info("❌ No relevant information found in the document.")



# Image analysis
elif mode == "Image Analysis":
    st.subheader("🖼️ Image Analysis & Chat with Memory")

    image = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

    if image:
        # ✅ Show the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # ✅ Initialize chat history if not exists
        if "image_chat_history" not in st.session_state:
            st.session_state.image_chat_history = []

        # ✅ Input prompt for image chat
        prompt = st.text_input("💬 Ask something about this image:")

        if st.button("🔍 Analyze"):
            with st.status("🤖 Generating response...", expanded=True) as status:
                progress = st.progress(0)

                for percent_complete in range(100):
                    progress.progress(
                        (percent_complete + 1) / 100,
                        text=f"Analyzing... {percent_complete + 1}%"
                    )
                    import time
                    time.sleep(0.01)

                try:
                    # 🔥 Analyze with image and chat history
                    response = analyze_image_with_llava(
                        image, prompt, history=st.session_state.image_chat_history
                    )

                    # 🔥 Append to chat history
                    st.session_state.image_chat_history.append(("user", prompt))
                    st.session_state.image_chat_history.append(("assistant", response))

                    # 🔥 Store the response into vector DB for future searches
                    store_data(doc_id=image.name + "_" + str(len(st.session_state.image_chat_history)), text=response)

                    progress.empty()
                    status.update(label="✅ Response ready!", state="complete")
                    st.subheader("Response")
                    st.success(response)
                    st.info("✅ Description saved for searching.")

                except Exception as e:
                    progress.empty()
                    status.update(label="❌ Failed", state="error")
                    st.error(f"❌ Error: {e}")

        # ✅ Display chat history
        if st.session_state.image_chat_history:
            with st.expander("📝 Chat History with Image"):
                for role, content in st.session_state.image_chat_history:
                    if role == "user":
                        st.markdown(f"**🧑‍💻 You:** {content}")
                    else:
                        st.markdown(f"**🤖 Assistant:** {content}")

        # ✅ Clear chat history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.image_chat_history = []

        st.markdown("---")

        # 🔍 Vector Search on Stored Image Descriptions
        st.subheader("🔎 Search Across Stored Image Descriptions")

        search_query = st.text_input("💬 Ask about any stored images:")

        if st.button("🔍 Search Images"):
            results = search_data(db, search_query, top_k=3)
            if results["documents"]:
                st.subheader("🔍 Search Results")
                for doc in results["documents"][0]:
                    st.markdown(f"🖼️ {doc}")
            else:
                st.info("No similar images found.")

    else:
        st.info("📤 Please upload an image to start.")



# 🔍 Knowledge Base Mode
elif mode == "Knowledge Base":
    st.subheader("📚 Local Knowledge Base")

    import os
    import shutil
    from vector_store_utils import create_chunks, store_data, search_data
    from pdf_utils import extract_text_from_pdf

    # Create necessary folders
    os.makedirs("knowledge_base", exist_ok=True)
    os.makedirs("recycle_bin_kb", exist_ok=True)

    # -------------------------------
    # 📤 Upload Files
    st.subheader("📤 Upload Files to Knowledge Base")

    uploaded_file = st.file_uploader(
        "Upload a file (.pdf, .txt, .md)", type=["pdf", "txt", "md"]
    )

    if uploaded_file:
        save_path = os.path.join("knowledge_base", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"✅ '{uploaded_file.name}' uploaded successfully!")

    # -------------------------------
    # 📂 Display Knowledge Base Files
    st.subheader("📂 Knowledge Base Files")

    knowledge_files = os.listdir("knowledge_base")

    if knowledge_files:
        st.markdown(f"**Files in Knowledge Base:** {', '.join(knowledge_files)}")

        file_to_delete = st.selectbox("🗑️ Select file to move to Recycle Bin", ["None"] + knowledge_files)

        if file_to_delete != "None":
            if st.button("🗑️ Move to Recycle Bin"):
                source = os.path.join("knowledge_base", file_to_delete)
                destination = os.path.join("recycle_bin_kb", file_to_delete)
                shutil.move(source, destination)
                st.success(f"✅ '{file_to_delete}' moved to Recycle Bin.")
                knowledge_files = os.listdir("knowledge_base")  # Refresh
    else:
        st.info("No files in Knowledge Base. Upload to begin.")

    # -------------------------------
    # 🔄 Load and Index Files into Vector Store
    if st.button("🔄 Load Knowledge Base"):
        with st.spinner("🔄 Loading and indexing files..."):
            for file in knowledge_files:
                filepath = os.path.join("knowledge_base", file)

                if file.endswith((".txt", ".md")):
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                        chunks = create_chunks(content)
                        for idx, chunk in enumerate(chunks):
                            store_data(db, doc_id=f"{file}_chunk_{idx}", text=chunk)

                elif file.endswith(".pdf"):
                    with open(filepath, "rb") as f:
                        text = extract_text_from_pdf(f)
                        chunks = create_chunks(text)
                        for idx, chunk in enumerate(chunks):
                            store_data(db, doc_id=f"{file}_chunk_{idx}", text=chunk)

        st.success("✅ Knowledge Base Loaded Successfully!")

    # -------------------------------
    # 🔎 Query Knowledge Base
    st.subheader("🔎 Query the Knowledge Base")

    query = st.text_input("💬 Ask something about the Knowledge Base:")

    if st.button("🔍 Search and Answer"):
        if not query.strip():
            st.warning("❗ Please enter a query.")
        else:
            with st.spinner("🔍 Searching..."):
                results = search_data(db, query, top_k=3)

                if results and results["documents"]:
                    combined_context = "\n\n".join(results["documents"][0])

                    prompt = f"""You are an assistant. Use the following context to answer the question.

Context:
{combined_context}

Question: {query}

Answer:"""

                    response = chat_with_model(prompt, model=selected_model)

                    st.subheader("💡 Response")
                    st.success(response)
                else:
                    st.info("❌ No relevant information found.")

    # -------------------------------
    # 🗑️ Recycle Bin Management
    st.subheader("🗑️ Knowledge Base Recycle Bin")

    recycle_files = os.listdir("recycle_bin_kb")

    if recycle_files:
        st.markdown(f"**Files in Recycle Bin:** {', '.join(recycle_files)}")

        # --------- Restore
        file_to_restore = st.selectbox("♻️ Select file to restore", ["None"] + recycle_files)

        if file_to_restore != "None":
            if st.button("♻️ Restore File"):
                src = os.path.join("recycle_bin_kb", file_to_restore)
                dest = os.path.join("knowledge_base", file_to_restore)
                shutil.move(src, dest)
                st.success(f"✅ '{file_to_restore}' restored to Knowledge Base.")

        # --------- Permanent Delete with Confirmation
        file_to_perma_delete = st.selectbox("🚫 Select file to permanently delete", ["None"] + recycle_files)

        if file_to_perma_delete != "None":
            confirm_key = f"confirm_delete_{file_to_perma_delete}"
            if confirm_key not in st.session_state:
                st.session_state[confirm_key] = False

            if not st.session_state[confirm_key]:
                if st.button(f"🚫 Delete '{file_to_perma_delete}' Permanently"):
                    st.session_state[confirm_key] = True
                    st.warning(f"⚠️ Are you sure you want to permanently delete '{file_to_perma_delete}'?")
            else:
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("✅ Yes, Delete"):
                        os.remove(os.path.join("recycle_bin_kb", file_to_perma_delete))
                        st.success(f"❌ '{file_to_perma_delete}' permanently deleted.")
                        st.session_state[confirm_key] = False

                with col2:
                    if st.button("❌ Cancel"):
                        st.info("Deletion cancelled.")
                        st.session_state[confirm_key] = False
    else:
        st.info("Recycle Bin is empty.")
