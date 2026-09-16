import time
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv
load_dotenv()

# ---- MUST match the DB maker script ----
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PERSIST_DIR     = "chroma_db"
COLLECTION_NAME = "valuemomentum"
# ----------------------------------------

# Path to the locally saved embedding model
EMBEDDING_MODEL_PATH = f"./models/{EMBEDDING_MODEL}"  

# Load embedding model from LOCAL folder
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)

# Initialize Chroma vector store with the loaded embeddings
vector_store = Chroma(
    collection_name=COLLECTION_NAME, # we can see and check it inthe chroma_db folder inside the sqllite database
    embedding_function=embeddings, # the model we used to create the embeddings same we have to use it to query the embeddings
    persist_directory=PERSIST_DIR, # this is the folder where the embeddings are stored, we can see it in the chroma_db folder
)


# Initialize the LLM using the Ollama API
llm = ChatOllama(
    model="gemma4:31b-cloud",
    base_url="https://ollama.com",
    client_kwargs={
        "headers": {
            "Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"
        }
    }
)

def format_context_for_llm(user_question, retrieved_results):
    """
    Build the full prompt for the LLM from the retrieved chunks.
    Returns None if there is nothing to ground the answer on.
    """
    if not retrieved_results:
        return None

    formatted_results = []
    for i, (doc, score) in enumerate(retrieved_results, start=1):
        formatted_results.append(
            f"--- Result {i} ---\n"
            f"  distance : {score:.4f}\n"      # lower = more similar
            f"  metadata : {doc.metadata}\n"
            f"  content  : {doc.page_content}\n"
        )

    context = "\n".join(formatted_results)

    prompt = f"""You are a helpful Value Momentum assistant. Use the context below to answer the question.
If the context does not contain enough information, respond with "I don't know".
Provide your answer in a concise, clear manner. Avoid repeating the context verbatim — summarize and synthesize it into a coherent response.
Keep the answer short to medium length if possible.

Context: {context}

Question: {user_question}


Answer: The answer i want in very specific format in the answer format i need you to give the
urls as well 
for example 
Answer: <your answer here>
Source: <source URL here> Context 
"""

    return prompt



# This is the main loop
def main():
    while True:
        user_input = input("Enter your query: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "bye":
            break

        start_time = time.time()

        results = vector_store.similarity_search_with_score(user_input, k=5)

        prompt = format_context_for_llm(user_input, results)
        if prompt is None:
            print("No results found.\n")
            continue

        response = llm.invoke(prompt)
        print(f"\nAnswer: {response.content}\n")
        print(f"Query took {time.time() - start_time:.2f}s\n")

        # no filter — your ingestion script uses source="valuemomentum_sample.docx"

        # if not results:
        #     print("No results found.\n")
        #     continue

        # for i, (doc, score) in enumerate(results, start=1):
        #     print(f"--- Result {i} ---")
        #     print(f"  distance : {score:.4f}")      # lower = more similar
        #     print(f"  metadata : {doc.metadata}")
        #     print(f"  content  : {doc.page_content}")
        #     print()

if __name__ == "__main__":
    main()