# Insurance chatbot example 
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOllama(
    model="gemma4:31b-cloud",
    base_url="https://ollama.com",
    client_kwargs={
        "headers": {
            "Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"
        }
    }
)


print("#"*100)
while True:
    
    user_input = input("User Message: ")
    
    endst = ["bye","good bye", "goodbye", "end", "Stop"]

    if user_input.lower() in endst:
        
        print("Thanks for using our chatbot bye!")
        
        break

    model_prompt = f"Answer the users questions in short and simple words: {user_input}"
    response = llm.invoke(model_prompt)
    print(f"AI Message: {response.content}")


# Available models to use in cloude version #
# model="gemma4:31b-cloud"
# model="gpt-oss:120b-cloud"
# model="gpt-oss:20b-cloud"
# model="nemotron-3-nano:30b-cloud"
# model="nemotron-3-super-cloud"
# model="nemotron-3-ultra-cloud"

