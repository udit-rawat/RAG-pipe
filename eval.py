import logging
from retrieval import retrieve_relevant_chunks
from sentence_transformers import SentenceTransformer, util
import numpy as np


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


model = SentenceTransformer("all-MiniLM-L6-v2")


def clean_text(text):
    """Removes citations and metadata to focus on actual content."""
    return text.split("Source:")[0].split("Citations:")[0].strip()


def evaluate_retrieval(query, ground_truth):
    """
    Evaluates retrieval using:
    1. Exact match (Does any passage contain the ground truth?)
    2. Semantic similarity score (cosine similarity)
    3. Mean Reciprocal Rank (MRR)
    """
    logging.info(f"Evaluating retrieval for query: {query}")

    retrieved_data = retrieve_relevant_chunks(query)

    if not retrieved_data:
        logging.warning("No relevant passages retrieved.")
        return {"similarity_score": 0.0, "exact_match": False, "MRR": 0.0}

    cleaned_retrieved_texts = [clean_text(p[0]) for p in retrieved_data]

    exact_match = any(ground_truth.lower() in passage.lower()
                      for passage in cleaned_retrieved_texts)

    retrieved_embedding = model.encode(
        cleaned_retrieved_texts, convert_to_tensor=True)
    ground_truth_embedding = model.encode(ground_truth, convert_to_tensor=True)

    similarity_scores = util.pytorch_cos_sim(
        ground_truth_embedding, retrieved_embedding).cpu().numpy()[0]
    avg_similarity = np.mean(similarity_scores)

    ranks = np.argsort(-similarity_scores)
    mrr = 1.0 / (ranks[0] + 1)

    logging.info(
        f"Exact Match: {exact_match}, Similarity Score: {avg_similarity:.4f}, MRR: {mrr:.4f}")

    return {"similarity_score": avg_similarity, "exact_match": exact_match, "MRR": mrr}


if __name__ == "__main__":
    evaluation_set = [
        ("What are the key concepts in the Attention Is All You Need paper?",
         "The paper introduces self-attention, transformer architecture, and attention mechanisms."),
        ("What is self-attention in Transformers?",
         "Self-attention allows a model to weigh different words in a sequence based on their relevance to a given word."),
        ("How does multi-head attention work?",
         "Multi-head attention enables the model to focus on different parts of the input sequence by using multiple attention heads."),
        ("Why are positional encodings used in Transformers?",
         "Since Transformers lack recurrence, positional encodings provide information about word order in a sentence."),
        ("What are the advantages of the Transformer model over RNNs?",
         "Transformers enable parallel processing, reduce vanishing gradient issues, and improve long-range dependency learning.")
    ]

    for query, ground_truth in evaluation_set:
        score = evaluate_retrieval(query, ground_truth)
        print(f"Query: {query}\nScore: {score}\n")
