
from langchain_ollama import ChatOllama
import time

# Initialize the Ollama model
#llm = ChatOllama(model="gemma:2b")


llm = ChatOllama(
    model="deepseek-r1:1.5b",  # or any model available on your server
    base_url="http://10.10.70.57:11434"
)


# List of prompts to test
prompts = [
    "What is sustainable packaging?",
    "Explain circular economy.",
    "How does solar energy work?"
]

# Loop through each prompt and measure response time
for prompt in prompts:
    start = time.time()
    response = llm.invoke(prompt)
    print(f"\nPrompt: {prompt}\nResponse: {response.content}\nTime taken: {time.time() - start:.2f}s")
