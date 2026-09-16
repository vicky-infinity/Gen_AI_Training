from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv
import time

load_dotenv()

api_key = os.getenv("OLLAMA_API_KEY")

llm = ChatOllama(
    model="gpt-oss:120b-cloud",
    api_key=api_key,
    temperature=0.2
)

def main():
    while True:
        
        user_input = input("Enter your query: ")
        start_time = time.time()
        if user_input.lower() == "bye":
            break
        response = llm.invoke(user_input)
        print(f"Response: {response.content}")
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()