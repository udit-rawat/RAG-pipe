import ollama
import logging
from retrieval import retrieve_relevant_chunks


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


PROMPT_TEMPLATE = """
You are an AI assistant answering based on retrieved academic papers. Ensure your response includes in-text citations in the format ( Source: Page X).

### Context (with sources):
{context}

### User Query:
{query}

### Response:
Provide a detailed response with in-text citations in the following format:  
Example: "Self-attention allows models to focus on different words based on context ( Source: Page X)."
"""


def generate_response(query):
    """
    Generates a response by retrieving relevant documents and prompting the model.
    """
    logging.info(f"Generating response for query: {query}")

    retrieved_data = retrieve_relevant_chunks(query)

    if not retrieved_data:
        logging.warning(
            "No relevant context found. Returning default response.")
        return "I couldn't find relevant information for your query."

    logging.info(f"Retrieved Data: {retrieved_data}")

    formatted_context = []
    sources_list = set()
    for chunk in retrieved_data:
        if isinstance(chunk, tuple) and len(chunk) == 2:
            text, source = chunk
            formatted_context.append(f"{text} ( Source: {source})")
            sources_list.add(source)
        else:
            formatted_context.append(chunk)  # No source available

    formatted_context_text = "\n\n".join(formatted_context)

    prompt = PROMPT_TEMPLATE.format(
        context=formatted_context_text, query=query)

    try:
        response = ollama.chat(
            model="llama3.2:latest",
            messages=[{"role": "user", "content": prompt}]
        )
        result = response["message"]["content"]
        logging.info(f"Generated Response: {result}")

        # Append sources only if available
        if sources_list:
            sources_text = "\n".join(
                [f"Source: {source}" for source in sources_list])
            result += f"\n\nSources used:\n{sources_text}"

        return result

    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return "An error occurred while generating the response. Please try again."


if __name__ == "__main__":
    user_query = "Explain self-attention in transformers."
    response = generate_response(user_query)
    print(response)
