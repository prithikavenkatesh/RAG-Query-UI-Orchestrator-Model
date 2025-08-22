
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import requests
import ast

# Connect to Milvus
connections.connect(alias="default", host="10.10.70.57", port="19530")

# Drop existing collection if it exists
collection_name = "pdf_chunks"
if utility.has_collection(collection_name):
    Collection(collection_name).drop()
    print(f"🗑️ Dropped existing collection '{collection_name}'.")

# Define new schema with 1024-d embeddings
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000)
]
schema = CollectionSchema(fields, description="Chunked PDF embeddings")

# Create new collection
collection = Collection(name=collection_name, schema=schema)
print(f"✅ Recreated collection '{collection_name}' with 1024-d embeddings.")

# Sample chunk texts
chunk_texts = [
    "Renewable energy sources include solar, wind, and hydro.",
    "Geothermal energy taps into the Earth's heat.",
    "Biomass energy uses organic materials for power."
]

# Ollama embedding API
OLLAMA_EMBEDDING_URL = "http://10.10.70.57:11434/api/embeddings"
EMBEDDING_MODEL = "inke/Qwen3-Embedding-0.6B:latest"

def get_embedding(text):
    response = requests.post(OLLAMA_EMBEDDING_URL, json={
        "model": EMBEDDING_MODEL,
        "prompt": text
    })
    response.raise_for_status()
    embedding = response.json()["embedding"]

    # If it's a string, convert to list of floats
    if isinstance(embedding, str):
        embedding = ast.literal_eval(embedding)
    embedding = [float(x) for x in embedding]

    # Debug print
    print(f"🔢 Embedding type: {type(embedding)}")
    print(f"📊 First 5 values: {embedding[:5]}")

    return embedding

# Generate embeddings
chunk_embeddings = [get_embedding(text) for text in chunk_texts]

# Insert into Milvus (embedding, text)
entities = [
    chunk_embeddings,
    chunk_texts
]
collection.insert(entities)
print("✅ Inserted chunk embeddings into Milvus.")


collection.create_index(
    field_name="embedding",
    index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}}
)


# Load collection into memory before searching
collection.load()
# Query embedding
query_text = "What are examples of renewable energy?"
query_embedding = get_embedding(query_text)

# Search Milvus
search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
results = collection.search(
    data=[query_embedding],
    anns_field="embedding",
    param=search_params,
    limit=2,
    output_fields=["text"]
)

# Display results
print("🔍 Top matching chunks:")
for result in results[0]:
    print(f"- {result.entity.get('text')} (score: {result.distance:.4f})")
