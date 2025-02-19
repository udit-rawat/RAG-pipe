import faiss
import numpy as np
import pickle
import logging
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

logging.info("Loading FAISS HNSW index and stored text chunks...")

# Load the FAISS HNSW index
index = faiss.read_index("faiss_index_hnsw.idx")

# Load stored text chunks and metadata
with open("text_chunks.pkl", "rb") as f:
    text_chunks = pickle.load(f)

with open("metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

with open("references.pkl", "rb") as f:
    references = pickle.load(f)

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")
logging.info("Loaded embedding model.")


def retrieve_relevant_chunks(query, top_k=5, min_sim_threshold=0.4):
    """
    Retrieves the top-k relevant chunks using FAISS HNSW index.
    """
    query_embedding = np.array([embedder.encode(query)], dtype=np.float32)

    # Perform nearest neighbor search
    distances, indices = index.search(query_embedding, top_k)

    results = []
    highest_sim = 0.0

    for i, idx in enumerate(indices[0]):
        if idx == -1:  # Handle case where no valid index is found
            continue

        chunk_text = text_chunks[idx]
        meta = metadata[idx]

        # Convert FAISS L2 distance to cosine similarity
        similarity = 1 / (1 + distances[0][i])
        highest_sim = max(highest_sim, similarity)

        # Retrieve citations
        cited_refs = [f"[{c}]: {references.get(c, 'No citation found')}" for c in meta.get(
            "citations", [])]

        # Format retrieved chunk
        formatted_chunk = f"{chunk_text}\n Source: Page {meta['page']} of attention_is_all_you_need.pdf"
        if cited_refs:
            formatted_chunk += "\n Citations:\n" + "\n".join(cited_refs)

        results.append((formatted_chunk, similarity))

    # If retrieved chunks have low similarity, expand the search range
    if highest_sim < min_sim_threshold and top_k < 10:
        logging.warning(
            "Low similarity scores detected. Expanding retrieval range...")
        return retrieve_relevant_chunks(query, top_k=top_k+2)

    return sorted(results, key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    user_query = input("Enter query: ")
    retrieved_results = retrieve_relevant_chunks(user_query)

    logging.info(f"Retrieved {len(retrieved_results)} relevant chunks.\n")
    for res, sim_score in retrieved_results:
        print(f"{res}\n Similarity: {sim_score:.4f}\n")
