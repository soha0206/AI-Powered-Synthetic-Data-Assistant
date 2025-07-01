from transformers import pipeline
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer

def query_dataset(query, filename, cleaned_folder='cleaned'):
    vector_store_path = os.path.join(cleaned_folder, f"{filename}_vector_store.pkl")
    if not os.path.exists(vector_store_path):
        raise FileNotFoundError(f"Vector store not found at {vector_store_path}. Please process the dataset first.")
    
    with open(vector_store_path, 'rb') as f:
        vector_store = pickle.load(f)
    chunks, index = vector_store['chunks'], vector_store['index']
    
    generator = pipeline('text-generation', model='distilgpt2')
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = embedder.encode([query])
    k = 3
    distances, indices = index.search(query_embedding, k)
    context = [chunks[i] for i in indices[0]]
    context_text = " ".join(context)

    prompt = f"Answer the query '{query}' based on: {context_text}"
    answer = generator(prompt, max_length=150, num_return_sequences=1)[0]['generated_text']
    return answer.strip(), context