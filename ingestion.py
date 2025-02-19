import os
import re
import pickle
import faiss
import numpy as np
import requests
import pymupdf
import logging
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Download the research paper if it doesn't exist
URL = "https://arxiv.org/pdf/1706.03762.pdf"
PDF_FILE = "attention_is_all_you_need.pdf"

if not os.path.exists(PDF_FILE):
    logging.info("Downloading research paper...")
    response = requests.get(URL)
    with open(PDF_FILE, "wb") as f:
        f.write(response.content)

logging.info("PDF ready for processing.")

# Function to extract text and metadata


def extract_text_with_metadata(pdf_path):
    """Extracts text, page numbers, and inline citations from a PDF."""
    doc = pymupdf.open(pdf_path)
    text_chunks = []
    metadata = []
    references = {}
    bibliography_started = False

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        if "References" in text or "REFERENCES" in text:
            bibliography_started = True

        if bibliography_started:
            matches = re.findall(r"\[(\d+)\]\s(.+)", text)
            for match in matches:
                ref_id, ref_text = match
                references[int(ref_id)] = ref_text
        else:
            citations = re.findall(r"\[(\d+)\]", text)
            text_chunks.append(text)
            metadata.append(
                {"page": page_num + 1, "citations": list(set(map(int, citations)))})

    doc.close()
    return text_chunks, metadata, references


logging.info("Extracting text and metadata...")
chunks, metadata, references = extract_text_with_metadata(PDF_FILE)

# Generate embeddings
logging.info("Generating embeddings...")
embeddings = np.array([embedder.encode(chunk) for chunk in chunks])

# Define FAISS index with HNSW
dimension = embeddings.shape[1]
M = 32  # Number of neighbors for HNSW graph (tuneable parameter)
index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_L2)  # Use L2 distance

# Add embeddings to the FAISS index
index.add(embeddings)

# Save the index and metadata
faiss.write_index(index, "faiss_index_hnsw.idx")

with open("text_chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

with open("metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

with open("references.pkl", "wb") as f:
    pickle.dump(references, f)

logging.info("FAISS HNSW index, text chunks, and metadata saved successfully.")
