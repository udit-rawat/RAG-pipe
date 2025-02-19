import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")


def load_faiss_index():
    return faiss.read_index("faiss_index.idx")


def load_text_chunks():
    with open("text_chunks.pkl", "rb") as f:
        return pickle.load(f)


def retrieve_relevant_chunks(query, top_k=3):
    index = load_faiss_index()
    text_chunks = load_text_chunks()

    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    return [text_chunks[i] for i in indices[0]]
