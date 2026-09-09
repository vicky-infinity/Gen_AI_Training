# Insurance chatbot with perticular formatting of response 
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

Answer only insurance related questions.
Reply the greeting messages politely in short
If the question is not related to insurance, decline to answer it.
Use the following examples to understand how to answer insurance questions.
Example 1:

Question: What is health insurance?

Answer:
Definition: health insurance is insurance which is usefull for the hospital expences

Simple example: If you get hospitalized, your health insurance will cover hospital bills

Example 2:

Question: What is life insurance?

Answer:
Definition: Life insurance provides financial protection to the beneficiaries in case the policy holder dies
Simple example: A person with lifeinsurance policy dies then his or hers family will get the policy amount
Conclusion: Life insurance is for financial protection to the family in case the policy holder dies


Now answer the following question in similer format with you knowledge 
Question: {user_input}

Answer:

"""
    response = llm.invoke(model_prompt)
    print(f"AI Message: {response.content}")