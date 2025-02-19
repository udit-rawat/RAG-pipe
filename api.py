from fastapi import FastAPI
from pydantic import BaseModel
import ollama
import logging
from retrieval import retrieve_relevant_chunks

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def query_model(request: QueryRequest):
    user_query = request.query
    logging.info(f"Received query: {user_query}")

    # Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(user_query)

    if not relevant_chunks:
        logging.warning("No relevant chunks found for the query.")
        return {
            "query": user_query,
            "answer": "I'm unable to find relevant information in the retrieved documents.",
            "retrieved_context": []
        }

    # Extract text from retrieved tuples (assuming each chunk is a tuple)
    formatted_chunks = [chunk[0] if isinstance(
        chunk, tuple) else chunk for chunk in relevant_chunks]

    context = "\n\n".join(formatted_chunks)

    # Construct the system prompt
    prompt = (
        f"You are an AI assistant trained to answer questions based on retrieved documents.\n\n"
        f"Here is the relevant context:\n{context}\n\n"
        f"Answer the following question using ONLY the given information:\n\n"
        f"Query: {user_query}"
    )

    # Query Ollama API
    try:
        response = ollama.chat(
            model="llama3.2:latest",
            messages=[{"role": "user", "content": prompt}]
        )["message"]["content"]
    except Exception as e:
        logging.error(f"Error while querying Ollama: {e}")
        return {
            "query": user_query,
            "answer": "There was an error generating a response. Please try again.",
            "retrieved_context": formatted_chunks
        }

    logging.info("Response generated successfully.")

    return {
        "query": user_query,
        "answer": response,
        "retrieved_context": formatted_chunks
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
