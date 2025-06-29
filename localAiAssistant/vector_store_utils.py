import langchain_community
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os
from pdf_utils import extract_text_from_pdf
from ollama_api import chat_with_model

# Split text into chunks
def create_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

# Create vector DB for a single document
def create_vector_store(chunks):
    embeddings = OllamaEmbeddings(model="llama3")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

# Build vector DB for multiple files in a folder
def build_file_database(folder_path):
    vector_store = None
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf(file_path)
            chunks = create_chunks(text)
            embeddings = OllamaEmbeddings(model="llama3")
            if vector_store is None:
                vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            else:
                vector_store.add_texts(chunks)
    return vector_store

# Query the document/vector store
def query_document(vector_store, question):
    docs = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Answer based on the following context:\n\n{context}\n\nQuestion: {question}"
    return chat_with_model(prompt)
