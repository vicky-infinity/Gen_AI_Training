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

while True:
    print("#"*30)
    user_input = input("User Message: ")
    
    endst = ["bye","good bye", "goodbye", "end", "Stop"]

    if user_input.lower() in endst:
        
        print("Thanks for using our chatbot bye!")
        
        break

    model_prompt = f"""
    You are a helpfull insurance domain assistant 
    Understand the users query and answer like a experienced person of insurance domain 
    do not answer the query if it is not related to the insurance domin denay to answer politely 
    and asnwer in plain text no md or html 
    {user_input}"""
    
    response = llm.invoke(model_prompt)
    print(f"AI Message: {response.content}")