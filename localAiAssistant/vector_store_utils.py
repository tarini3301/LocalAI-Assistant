from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import os
from pdf_utils import extract_text_from_pdf
from ollama_api import chat_with_model

# ✅ Initialize ChromaDB Client globally
client = PersistentClient(path="./chroma_db")
embedding_func = embedding_functions.DefaultEmbeddingFunction()


# ✅ Create or connect to collection dynamically
def create_vector_store(collection_name="local_knowledge"):
    """
    Returns a ChromaDB collection instance.
    """
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_func
    )


# ✅ Split text into chunks
def create_chunks(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


# ✅ Store data in ChromaDB
def store_data(collection, doc_id, text):
    collection.add(
        ids=[doc_id],
        documents=[text],
        metadatas=[{"source": doc_id}]
    )


# ✅ Search in ChromaDB
def search_data(collection, query, top_k=3):
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    return results


# ✅ Build Knowledge Base from a folder (PDFs, text files)
def build_knowledge_base(collection, folder_path):
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)

        if file.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif file.endswith((".txt", ".md")):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            continue  # Skip unsupported files

        chunks = create_chunks(text)

        for idx, chunk in enumerate(chunks):
            doc_id = f"{file}_chunk_{idx}"
            store_data(collection, doc_id, chunk)


# ✅ Query the Knowledge Base with LLM assistance
def query_knowledge_base(collection, question):
    results = search_data(collection, question, top_k=3)

    if results and results["documents"]:
        combined_context = "\n\n".join(results["documents"][0])

        prompt = f"""You are an assistant. Use the following context to answer the question.

Context:
{combined_context}

Question: {question}

Answer:"""

        response = chat_with_model(prompt)
        return response
    else:
        return "❌ No relevant information found."
