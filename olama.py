import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
import requests
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

response3 = requests.post()
# while True:
#     user_input = input("Enter your query")
#     if user_input == "bye":
#         exit
#     response = llm.invoke(user_input)
#     print(f"{response.content}")

iteration = 0
history = []

input_token = 0
output_token = 0
total_tokens = 0



#  Appends token counts
def respdetail(response, input_token, output_token, total_tokens):

    input_token += response.usage_metadata["input_tokens"]
    output_token += response.usage_metadata["output_tokens"]
    total_tokens += response.usage_metadata["total_tokens"]

    return input_token, output_token, total_tokens


def cost_calculater():
    
    ...


while True:
    print("*"*30)
    user_input = input("Enter your query: ")
    

    if user_input.lower() == "bye":
        print(f"Total questions: {iteration}")
        print(f"Total input tokens: {input_token}")
        print(f"Total output tokens: {output_token}")
        print(f"Total tokens: {total_tokens}")
        print(f"History: {history}")
        break

    model_input = f"""
You are a helpful, and conversational AI assistant.

Use the conversation history to maintain context and provide relevant answers.

Conversation history:
{history}

Current user message:
{user_input}
"""
    response = llm.invoke(model_input)

    # Update token counters
    input_token, output_token, total_tokens = respdetail(
        response,
        input_token,
        output_token,
        total_tokens
    )

    iteration += 1

    history.append({
        "User Question number {iteration}": user_input,
        f"AI assistant answer number {iteration}": response.content
    })

    print(f"Ai Response: {response.content}")
    print("*"*30)


# Free usage can be used with the following cloud models:

# gemma4:31b
# gpt-oss:120b
# gpt-oss:20b
# nemotron-3-nano:30b
# nemotron-3-super
# nemotron-3-ultra

###################### Cloud versions ##############################

# model="gemma4:31b-cloud"
# model="gpt-oss:120b-cloud"
# model="gpt-oss:20b-cloud"
# model="nemotron-3-nano:30b-cloud"
# model="nemotron-3-super-cloud"
# model="nemotron-3-ultra-cloud"


# (myenv) PS C:\Users\VickyVijayPatil\Downloads\Projects\IDP> python .\olama.py
# Enter your queryhe
# <bound method BaseModel.model_dump_json of AIMessage(content='I\'m sorry, but "he" doesn\'t provide enough context to understand what you\'re asking about. Could you please provide more information or rephrase your question? I\'d be happy to help if you can clarify what you\'re looking for.', additional_kwargs={}, response_metadata={'model': 'qwen2.5:1.5b', 'created_at': '2026-08-27T11:30:04.0777953Z', 'done': True, 'done_reason': 'stop', 'total_duration': 4002409900, 'load_duration': 2239778900, 'prompt_eval_count': 30, 'prompt_eval_duration': 117948000, 'eval_count': 50, 'eval_duration': 1634963000, 'logprobs': None, 'model_name': 'qwen2.5:1.5b', 'model_provider': 'ollama'}, id='lc_run--01a042fb-def6-72c2-ad94-4323717c08d3-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 30, 'output_tokens': 50, 'total_tokens': 80})>


#print(f"{response.model_dump_json}")
#print(f"{response.content}")