import streamlit as st
import fitz # PyMuPDF
import tempfile
import time
import requests
import numpy as np


OLLAMA_EMBEDDING_URL = "http://10.10.70.57:11434/api/embeddings"
EMBEDDING_MODEL = "inke/Qwen3-Embedding-0.6B:latest"

def get_embedding(text):
    response = requests.post(OLLAMA_EMBEDDING_URL, json={
        "model": EMBEDDING_MODEL,
        "prompt": text
    })
    response.raise_for_status()
    return response.json()["embedding"]

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def embedding_based_retrieval(chunks, query, top_k=3):
    query_embedding = get_embedding(query)
    chunk_embeddings = [get_embedding(chunk) for chunk in chunks]

    similarities = [cosine_similarity(query_embedding, emb) for emb in chunk_embeddings]
    ranked_chunks = sorted(zip(chunks, similarities), key=lambda x: x[1], reverse=True)

    return [chunk for chunk, score in ranked_chunks[:top_k]]


#from langchain_ollama import ChatOllama
from langchain_community.chat_models import ChatOllama

# Simulated placeholders for future integrations
INTEGRATIONS = {
    "OneDrive Ingestion": "Placeholder for OneDrive ingestion pipeline",
    "GPU Server": {
        "Embedding Model (Qwen)": "Placeholder for Qwen embedding model",
        "LLM (DeepSeek)": "Placeholder for DeepSeek LLM"
    },
    "Vector Base (Milvus)": "Placeholder for Milvus vector database",
    "Vector Admin UI (Attu)": "Placeholder for Attu admin interface",
    "Authorized User Desktop": "Placeholder for user authentication and desktop access"
}

#Simulated chunking and retrieval
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def simulate_chunking(text, chunk_size=300):
    words = text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks


# def simulate_retrieval(chunks, query):
    #return [chunk for chunk in chunks if query.lower() in chunk.lower()]


def simulate_retrieval(chunks, query):
    query_keywords = set(query.lower().split())
    retrieved = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        if query_keywords & chunk_words:  # if there's any overlap
            retrieved.append(chunk)
    return retrieved


# def real_llm_response(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"Answer the following query based on the context: \n\nContext: \n{context}\n\nQuery: {query}"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choice'][0]['message']['content']



def real_llm_response(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"Answer the following query based on the context:\n\n{context}\n\nQuery: {query}"
    llm = ChatOllama(
        model="deepseek-r1:1.5b",  # ✅ This model is confirmed to be available
        base_url="http://10.10.70.57:11434"
    )
    response = llm.invoke(prompt)
    return response.content


def simulate_llm_response(query, context):
    return f" Simulated response to '{query}' based on retrieved context."

# Streamlit UI
st.title("RAG Query UI & Orchestrator (Simulated)")
st.markdown("This app simulates a basic RAG pipeline with placeholders for future integration.")

chunks = []

# File upload or sample PDF
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
use_sample = st.checkbox("Use sample PDF (Introduction to Renewable Energy)")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name
    text = extract_text_from_pdf(pdf_path)
    st.write("Extracted Text Preview:", text[:1000])
    chunks = simulate_chunking(text)

elif use_sample:
    sample_text = """
    Introduction to Renewable Energy

    Renewable energy is derived from natural sources that are replenished constantly. Examples include solar, wind, hydro, geothermal, and biomass. These sources are considered sustainable and environmentally friendly compared to fossil fuels.

    Solar energy harnesses sunlight using photovoltaic cells. Wind energy uses turbines to convert wind into electricity. Hydropower utilizes flowing water to generate power. Geothermal taps into the Earth's heat, and biomass uses organic materials.

    The transition to renewable energy is crucial for reducing greenhouse gas emissions and combating climate change. Governments and industries are investing in clean energy technologies to build a sustainable future.
    """
    text = sample_text
    chunks = simulate_chunking(text)

# Query input
query = st.text_input("Enter your query:")
use_real_llm = st.checkbox("Use real LLM (Ollama)", value=True)

use_embedding_retrieval = st.checkbox("Use embedding-based retrieval", value=True)


if query:
    if use_embedding_retrieval:
        try:
            retrieved_chunks = embedding_based_retrieval(chunks, query, top_k=3)
        except Exception as e:
            st.error(f"Embedding retrieval failed: {e}")
            retrieved_chunks = simulate_retrieval(chunks, query)
    else:
        retrieved_chunks = simulate_retrieval(chunks, query)
    #response = simulate_llm_response(query, retrieved_chunks)
    #Choose between real or simulated LLM
    if use_real_llm:
        try:
            response = real_llm_response(query, retrieved_chunks)
        except Exception as e:
            st.error(f"LLM invocation failed: {e}")
            response = simulate_llm_response(query, retrieved_chunks)
    else:
        response = real_llm_response(query, retrieved_chunks)

    st.subheader("📄 Retrieved Context")
    for i, chunk in enumerate(retrieved_chunks):
        st.text_area(f"Chunk {i+1}", chunk, height=100)

    st.subheader("🧠 LLM Response")
    st.write(response)

# Display integration placeholders
st.sidebar.title("🔧 Future Integrated Integrations")
for key, value in INTEGRATIONS.items():
    if isinstance(value, dict):
        st.sidebar.subheader(key)
        for subkey, subvalue in value.items():
            st.sidebar.text(f"{subkey}: {subvalue}")
    else:
        st.sidebar.text(f"{key}: {value}")




