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
print("Model loading...")
EMBEDDING_MODEL_PATH = f"./models/{EMBEDDING_MODEL}"  
print("Model loaded Sucessfull")

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

questlist= [
    "What does ValueMomentum do?",
    "Who does ValueMomentum serve?",
    "What industries does ValueMomentum focus on?",
    "What are ValueMomentum's key statistics?",
    "How many insurers has ValueMomentum served?",
    "How many associates does ValueMomentum have?",
    "What are ValueMomentum's core values?",
    "Where are ValueMomentum's offices located?",
    "Who is the CEO of ValueMomentum?",
    "Who is on ValueMomentum's leadership team?",
    "What is ValueMomentum's corporate social responsibility program?",
    "What is the SHEE CSR program?",
    "What does ValueMomentum's new brand identity represent?",
    "What partnerships does ValueMomentum have?",
    "What is ValueMomentum's P&C focus?",
    "What capabilities does ValueMomentum offer?",
    "What is included in Advisory & Innovation?",
    "What is included in Technology Transformation?",
    "What is included in Data & Advanced Analytics?",
    "What is included in Managed Services?",
    "What is Core Value Optimization?",
    "What is Core Value Transformation?",
    "What services are included in Application Services?",
    "What does ValueMomentum offer for Quality Assurance?",
    "What does ValueMomentum offer for Distribution & Marketing?",
    "What is Channel Management?",
    "What is Experience and Engagement?",
    "What does ValueMomentum offer for Claims?",
    "What does ValueMomentum offer for Underwriting Precision?",
    "What does ValueMomentum offer for Product Strategy and Portfolio Profitability?",
    "What does ValueMomentum offer for MLOps?",
    "What does ValueMomentum offer for Augmented Claims Intelligence?",
    "What does ValueMomentum offer for Cloud Services?",
    "What does ValueMomentum offer for Core Services?",
    "Is ValueMomentum a Guidewire partner?",
    "What level of Guidewire partnership does ValueMomentum have?",
    "How many Guidewire experts does ValueMomentum have?",
    "How many Guidewire projects has ValueMomentum completed?",
    "What Guidewire Cloud specialization did ValueMomentum achieve?",
    "What Guidewire services does ValueMomentum offer?",
    "What Guidewire marketplace accelerators has ValueMomentum published?",
    "What are common questions about Guidewire Cloud Platform implementation?",
    "What are best practices for Guidewire Cloud transition?",
    "How did Pekin Insurance use Guidewire Cloud for multi-state expansion?",
    "How did Farm Bureau Insurance Company of Idaho modernize with Guidewire?",
    "How did NJM implement BOP and CUMB on Guidewire PolicyCenter?",
    "How did a super-regional carrier use Guidewire Cloud Platform for commercial lines growth?",
    "What challenges do insurers face in Guidewire Cloud migration?",
    "Is ValueMomentum a Duck Creek partner?",
    "What is Duck Creek OnDemand?",
    "What are the advantages of Duck Creek OnDemand?",
    "What are the top FAQs for Duck Creek OnDemand upgrades?",
    "What prerequisites are needed before upgrading to Duck Creek OnDemand?",
    "How long does a Duck Creek OnDemand upgrade take?",
    "What are best practices for a smooth Duck Creek OnDemand transition?",
    "What is ValueMomentum's Cognitive Underwriter Virtual Assistant?",
    "How did CUVA help underwriters?",
    "How did a usage-based insurer expand geographically on Duck Creek OnDemand?",
    "What are the four elements of severity conservation?",
    "How can insurers control claims leakage?",
    "How can improved subrogation management reduce claims leakage?",
    "How is AI used for fraud detection in claims?",
    "How can insurers predict litigation risk early?",
    "How do agentic systems transform claims processing from FNOL to settlement?",
    "How does intelligent document processing reduce LAE?",
    "How does graph-based AI improve claims assignment?",
    "What are the three steps to redefine the claims journey?",
    "How are leading insurers modernizing the claims experience?",
    "What are best practices for intelligent automation in claims?",
    "How do AI and telematics improve auto claims?",
    "What is an analytics-driven claims knowledge solution?",
    "What is the impact of social inflation on insurers?",
    "How can insurers mitigate social inflation?",
    "What does a modern claims organization look like in 2026?",
    "How does advanced analytics reshape the claims workflow?",
    "What is claims maturity in P&C insurance?",
    "What are the 7 key capabilities of modern insurance underwriting?",
    "How can AI improve underwriting?",
    "How can generative AI be used in underwriting?",
    "What are 5 trends shaping the future of underwriting?",
    "How is agentic AI changing underwriting?",
    "How do you operationalize an underwriting platform?",
    "What questions should insurers consider for commercial underwriting success?",
    "What is augmented underwriting?",
    "How does AI-enabled submission triage improve efficiency?",
    "What are the priorities and challenges in specialty insurance underwriting?",
    "How does product content management help commercial lines insurers?",
    "How does pricing modernization improve underwriting?",
    "How can insurers improve underwriting with Power BI?",
    "How does AI aerial imagery improve homeowners underwriting?",
    "Why is a modern data platform important for insurers?",
    "What are the challenges of data modernization in insurance?",
    "How can insurers enact effective data governance?",
    "What is an ROI approach to data modernization?",
    "How can insurers standardize BI operations with Power BI?",
    "How do you migrate from Tableau to Power BI?",
    "What are modern data apps in P&C insurance?",
    "What are 5 use cases for insurance analytics?",
    "How does advanced analytics help claims?",
    "How do you build an AWS architecture for claims analytics?",
    "What is MLOps and how does ValueMomentum help?",
    "What data services does ValueMomentum offer?",
    "How does Snowflake help insurers?",
    "How does Databricks help insurers?",
    "How can insurers drive insights with critical data capabilities?",
    "How can insurers democratize data?",
    "How can generative AI be applied in insurance?",
    "What are generative AI use cases in underwriting?",
    "What are generative AI use cases in claims?",
    "What are generative AI use cases in distribution?",
    "What are generative AI use cases in product development?",
    "How do you integrate generative AI with legacy systems?",
    "How can insurers mitigate the risks of generative AI?",
    "How can insurers master AI data privacy and security?",
    "What is retrieval-augmented generation and how does it drive decisions?",
    "How do RAG and agentic workflows supercharge insurance operations?",
    "Why are AI guardrails core infrastructure for P&C insurance?",
    "What are multi-agent systems in P&C insurance?",
    "How is agentic AI changing career pathways in P&C insurance?",
    "Why might open-source software be better for insurance AI and GenAI?",
    "What are the pros and cons of generative AI in insurance?",
    "How does AI-powered DevSecOps work?",
    "What are 4 steps toward breaking AI innovation paralysis?",
    "How does AI improve customer experience and profitability in underwriting?",
    "How can insurers make agent experience a growth engine?",
    "What are 7 ways modern agent portals drive growth?",
    "What are 3 questions to guide a modern agent portal strategy?",
    "What are the three tiers of a successful agent portal experience?",
    "How do you build a mobile-first solution for insurance?",
    "How can mobile apps enhance customer experience?",
    "How do you evolve an insurance mobile app strategy?",
    "How do you turn a digital experience platform into a growth platform?",
    "What are insurance distribution models driving growth?",
    "How do APIs enhance the insurance agent digital experience?",
    "What is embedded insurance?",
    "What are API marketplaces in insurance?",
    "What are insurance aggregators?",
    "What are agency platforms?",
    "What are digital use cases for customer experience in insurance?",
    "How do you build a seamless insurance digital experience?",
    "How does customer communication management improve CX?",
    "What are CCM trends in insurance for 2026?",
    "How do you modernize customer communication management?",
    "What are cloud migration strategies for insurers?",
    "What are the 4 key cloud adoption trends?",
    "How do you choose a cloud readiness assessment?",
    "What are the challenges of a multi-cloud strategy?",
    "What are the 6 pillars of cloud platform engineering?",
    "How does FinOps help manage cloud spend?",
    "How does unified observability unlock cloud value?",
    "How do internal developer platforms make insurers more agile?",
    "What is cloud observability in insurance?",
    "What are the stages of cloud observability maturity?",
    "How does AI-powered DevSecOps accelerate delivery securely?",
    "What are the 5 phases of implementing AI-powered DevSecOps?",
    "How does application engineering increase scalability?",
    "What is configuration-driven application engineering?",
    "How do you build a sustainable API integration foundation?",
    "What are API best practices for insurers?",
    "What are common API integration challenges?",
    "How do you take API automated testing to the next level?",
    "What does a modern pricing function look like?",
    "Why can't pricing modernization wait?",
    "How do you modernize an insurance pricing and rating ecosystem?",
    "What are key challenges with legacy pricing and rating methods?",
    "How does the Earnix partnership help insurers?",
    "What is the anatomy of a modern pricing function?",
    "What is the anatomy of a modern pricing workflow?",
    "How does pricing modernization improve speed and profitability?",
    "How do you get pricing modernization right?",
    "How does quality engineering drive insurance success?",
    "What is adaptive sprint automation?",
    "How does continuous test orchestration help insurers?",
    "How is AI used in software testing?",
    "How does GenAI improve the insurance testing life cycle?",
    "Why is a shift-left approach to non-functional testing vital?",
    "How does extreme automation help insurers scale?",
    "What are key approaches to automated testing in insurance?",
    "What are QA/QE trends for insurance core transformation?",
    "How does quality engineering improve customer experience?",
    "How does test automation help core transformation?",
    "How do you infuse quality and speed in application testing?",
    "What is a digital core for insurance?",
    "How does a digital core help insurers through uncertainty?",
    "What are core modernization strategies for insurers?",
    "What are insurance core transformation challenges?",
    "How do you leverage insurance core systems for business growth?",
    "How do you drive growth with modern insurance core systems?",
    "How do you optimize run and grow for core systems?",
    "What is a bundled approach to core innovation?",
    "How do you modernize legacy website migration with Sitecore?",
    "How do you apply application management services to cloud-based core?",
    "How do you automate requirements gathering in core modernization?",
    "How do you use state expansion to drive growth via modernized core?",
    "How does Guidewire Cloud support core modernization?",
    "How does Duck Creek OnDemand support core modernization?",
    "How does OneShield support core modernization?",
    "What results did a Fortune 500 insurer achieve with AI and aerial imagery?",
    "How did a Fortune 500 insurer optimize subrogation recovery with predictive modeling?",
    "How did Hiscox drive business agility with enterprise quality engineering?",
    "How did HAI Group modernize its core and improve customer experience?",
    "How did Acuity automate customer correspondence with a CCM platform?",
    "How did Pekin Insurance adopt enterprise CCM for automated document generation?",
    "How did Pekin Insurance enhance statistical reporting and data verification?",
    "How did Pekin Insurance modernize core systems to improve data reporting?",
    "How did Erie Insurance optimize data operations with MarkLogic on AWS?",
    "How did Red River Mutual launch a scalable DevSecOps framework?",
    "How did Tokio Marine HCC CPLG create world-class digital distribution?",
    "How did a super-regional carrier accelerate commercial lines growth with Guidewire Cloud?",
    "How did a specialty carrier reduce LAE with intelligent document processing?",
    "How did a regional carrier revitalize its claims organization with a cloud-native platform?",
    "How did Farm Bureau Insurance Company of Idaho improve payments?",
    "How did a specialty carrier streamline infrastructure for distribution expansion?",
    "How did a specialty insurer launch an underwriting workbench?",
    "How did a multi-line insurer improve customer experience with SmartCOMM?",
    "How did a top-five insurer streamline customer communication management?",
    "How did a regional insurer drive CX and growth through a cloud-based CCM platform?",
    "What awards and recognitions has ValueMomentum received?",
    "What is the latest press release from ValueMomentum?",
    "What did ISG Provider Lens say about ValueMomentum?",
    "What did Everest Group say about ValueMomentum?",
    "What Guidewire services recognition has ValueMomentum received?",
    "What P&C Insurance IT Services recognition has ValueMomentum received?",
    "Who joined ValueMomentum's leadership?",
    "What partnerships has ValueMomentum announced?",
    "What is ValueMomentum's new brand identity?",
    "What events does ValueMomentum attend?",
    "What is ITC Vegas 2026?",
    "What is Guidewire Connections?",
    "What is Snowflake Summit?",
    "What is Databricks Data + AI Summit?",
    "What is OpenText World?",
    "What is ValueMomentum's culture?",
    "What are ValueMomentum's values?",
    "Why should someone join ValueMomentum?",
    "What benefits does ValueMomentum offer?",
    "What opportunities does ValueMomentum offer for graduates?",
    "What does ValueMomentum look for in candidates?",
    "What is ValueMomentum's CSR committee?",
    "How does ValueMomentum give back to communities?",
    "What is ValueMomentum's SHEE program?",
    "What is it like to work at ValueMomentum?",
    "What are the four elements of severity conservation?",
    "What are the 7 key capabilities of modern insurance underwriting?",
    "What are the 6 pillars of cloud platform engineering?",
    "What are the 5 phases of implementing AI-powered DevSecOps?",
    "What are the 3 tiers of a successful agent portal experience?",
    "What are the 3 steps to redefine the claims journey?",
    "What are the 4 key cloud adoption trends?",
    "What are the 5 distribution trends reshaping P&C insurance in 2026?",
    "What are the 3 ways to enhance application development for better UX?",
    "What are the 7 questions to consider for commercial underwriting success?",
    "What are the 5 ways agentic AI is a strategic advantage in underwriting?",
    "What are the 4 use cases for intelligent automation?",
    "What are the 5 use cases for insurance analytics?",
    "What are the 3 key considerations for data modernization success?",
    "What are the 4 ways to turn a DXP into a growth platform?",
    "What are the 5 ways to enhance litigation risk prediction with AI?",
    "What are the 7 steps to creating an analytics-driven claims knowledge solution?",
    "What are the 5 key takeaways for intelligent automation in claims?",
    "What are the 3 questions to guide your modern agent portal strategy?",
    "What are the 3 ways product content management can drive commercial lines growth?",
    "What is ValueMomentum's stock price?",
    "What is ValueMomentum's exact annual revenue?",
    "Who are all of ValueMomentum's clients?",
    "What is the exact pricing of ValueMomentum's services?",
    "Which competitor is better than ValueMomentum?",
    "What are today's current job openings?",
    "What is the current weather in Hyderabad?",
    "What is the latest news outside the provided corpus?",
    "What are private employee salaries?",
    "What are internal financial projections?",
    "What is the exact number of employees in each office?",
    "What are current live insurance regulations?",
    "What is the best insurance platform for every insurer?",
    "Can you guarantee ROI for a specific insurer?",
    "What is the exact implementation timeline for my company?"
  ]


import asyncio
import json
import time
import sys
import itertools


# ─────────────────────────────────────────────────────────────
# ⚙️  Concurrency limit (tune this based on your provider)
# ─────────────────────────────────────────────────────────────
MAX_CONCURRENT = 5      # try 3 if you still get 429s
RETRY_ATTEMPTS = 3      # auto-retry on rate limit
RETRY_BACKOFF  = 2.0    # seconds, doubles each retry


# ─────────────────────────────────────────────────────────────
# 🎨 Console helpers
# ─────────────────────────────────────────────────────────────
SPINNER = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])

def clear_line():
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()

def bar(done: int, total: int, width: int = 30) -> str:
    filled = int(width * done / total) if total else 0
    return "█" * filled + "░" * (width - filled)


# ─────────────────────────────────────────────────────────────
# 🧠 Answer a single question  (with semaphore + retry)
# ─────────────────────────────────────────────────────────────
async def answer_one(question: str, index: int, total: int,
                     state: dict, lock: asyncio.Lock,
                     sem: asyncio.Semaphore) -> dict:

    start = time.perf_counter()

    async with lock:
        state["running"][index] = question[:50]
        render(state, total)

    async with sem:                       # 🔒 throttle here
        try:
            # 1. Retrieve
            results = vector_store.similarity_search_with_score(question, k=5)

            # 2. Build prompt
            prompt = format_context_for_llm(question, results)
            if prompt is None:
                async with lock:
                    state["running"].pop(index, None)
                    state["done"][index] = ("⚠️  no context", 0)
                    render(state, total)
                return {"question": question, "answer": "No results found."}

            # 3. Call LLM with retry on rate-limit
            last_err = None
            for attempt in range(RETRY_ATTEMPTS):
                try:
                    response = await llm.ainvoke(prompt)
                    break
                except Exception as e:
                    last_err = e
                    msg = str(e).lower()
                    if "rate" in msg or "429" in msg or "too many" in msg:
                        wait = RETRY_BACKOFF * (2 ** attempt)
                        async with lock:
                            state["running"][index] = f"⏳ rate-limited, retry in {wait:.0f}s"
                            render(state, total)
                        await asyncio.sleep(wait)
                        continue
                    raise
            else:
                raise last_err

            elapsed = time.perf_counter() - start

            async with lock:
                state["running"].pop(index, None)
                state["done"][index] = ("✅", elapsed)
                render(state, total)

            return {"question": question, "answer": response.content}

        except Exception as e:
            async with lock:
                state["running"].pop(index, None)
                state["done"][index] = (f"❌ {type(e).__name__}", 0)
                render(state, total)
            return {"question": question, "answer": f"ERROR: {e}"}


# ─────────────────────────────────────────────────────────────
# 🖥️  Render the live console
# ─────────────────────────────────────────────────────────────
def render(state: dict, total: int):
    done = len(state["done"])
    running = len(state["running"])
    pct = int(100 * done / total) if total else 0

    clear_line()
    sys.stdout.write(
        f"\r{next(SPINNER)}  "
        f"[{bar(done, total)}] {done}/{total} ({pct}%)  "
        f"🚀 running: {running}"
    )
    sys.stdout.flush()


# ─────────────────────────────────────────────────────────────
# 🚦 Main
# ─────────────────────────────────────────────────────────────
async def main():
    total = len(questlist)
    print(f"\n🎯 Firing {total} questions  (max {MAX_CONCURRENT} at a time)...\n")

    state = {"running": {}, "done": {}}
    lock = asyncio.Lock()
    sem  = asyncio.Semaphore(MAX_CONCURRENT)     # 🔒

    start_all = time.perf_counter()

    tasks = [
        answer_one(q, i, total, state, lock, sem)
        for i, q in enumerate(questlist)
    ]

    async def ticker():
        while len(state["done"]) < total:
            render(state, total)
            await asyncio.sleep(0.1)

    ticker_task = asyncio.create_task(ticker())

    results = await asyncio.gather(*tasks, return_exceptions=True)
    ticker_task.cancel()

    clear_line()
    total_time = time.perf_counter() - start_all

    # ── Build answers list ────────────────────────────────
    answers = []
    for r in results:
        if isinstance(r, Exception):
            answers.append({"question": "UNKNOWN", "answer": f"ERROR: {r}"})
        else:
            answers.append(r)

    # ── Save JSON ─────────────────────────────────────────
    with open("question_test.json", "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=2, ensure_ascii=False)

    # ── Summary ───────────────────────────────────────────
    ok     = sum(1 for _, (tag, _) in state["done"].items() if tag == "✅")
    failed = sum(1 for _, (tag, _) in state["done"].items() if tag.startswith("❌"))
    empty  = sum(1 for _, (tag, _) in state["done"].items() if tag.startswith("⚠️"))
    times  = [t for _, (tag, t) in state["done"].items() if tag == "✅"]
    avg    = sum(times) / len(times) if times else 0
    fastest = min(times) if times else 0
    slowest = max(times) if times else 0

    print("━" * 60)
    print(f"🏁  DONE in {total_time:.2f}s")
    print("━" * 60)
    print(f"  ✅ Successful : {ok}")
    print(f"  ⚠️  Empty     : {empty}")
    print(f"  ❌ Failed     : {failed}")
    print(f"  ⏱️  Avg       : {avg:.2f}s   (fastest {fastest:.2f}s · slowest {slowest:.2f}s)")
    print(f"  💾 Saved to   : question_test.json")
    print("━" * 60 + "\n")

    return answers


if __name__ == "__main__":
    asyncio.run(main())