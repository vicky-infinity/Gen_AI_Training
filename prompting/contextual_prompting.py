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

# This can be the histoy of the convo as well

context = """
Our company has introduced this new insurance called the plant insurance where the user can insure
their plants , or mini garden etc 
the premium is claculated based on the plant for flower plants its 5 percent of the amount per year and
for other plants its 2 percent 
"""


while True:
    user_input = input("You: ")
    prompt = f"""
    You are a insurance agent you have the knowledge of insurance and also the new type of 
    insurance that our company is providing this is the context of that insurance 
    Context:
    {context}
    Only answer when one ask about the plant insurce or special insurance of the 
    Company other wise just the normal quesion and answers


    User question:
    {user_input}
    """

    response = llm.invoke(prompt)

    print("AI: ", response.content)  