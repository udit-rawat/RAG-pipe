import requests

API_URL = "http://127.0.0.1:8000/query"


test_question = "What is self-attention in Transformers?"


response = requests.post(API_URL, json={"query": test_question})

# Print response
print("\n Test Query:", test_question)
print(" API Response:", response.json())
