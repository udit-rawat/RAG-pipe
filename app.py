import streamlit as st
import ollama
from retrieval import retrieve_relevant_chunks

st.title("Q&A Retrieval System")

user_input = st.text_input("Enter your query:")

if st.button("Search"):
    if user_input:
        retrieved_data = retrieve_relevant_chunks(user_input)

        if not retrieved_data:
            st.write("No relevant context found.")
        else:

            if isinstance(retrieved_data[0], tuple) and len(retrieved_data[0]) == 2:
                formatted_context = "\n".join(
                    [f"{text} (Source: {source})" for text,
                     source in retrieved_data]
                )
                sources = {text: source for text, source in retrieved_data}
            else:
                formatted_context = "\n".join(retrieved_data)
                sources = {}

            response = ollama.chat(
                model="llama3.2:latest",
                messages=[
                    {"role": "user", "content": f"Answer based on:\n{formatted_context}\n\nQuery: {user_input}"}
                ]
            )["message"]["content"]

            for text, source in sources.items():
                if text in response and source not in response:
                    response += f"\n(Source: {source})"

            st.write("### Answer:")
            st.write(response)

            st.write("### Retrieved Context:")
            for chunk in retrieved_data:
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    text, source = chunk
                    st.write(f"- {text} (Source: {source})")
                else:
                    st.write(f"- {chunk}")
