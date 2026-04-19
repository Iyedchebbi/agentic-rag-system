import requests
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
# =========================
# 1. OPENROUTER EMBEDDINGS
# =========================

class OpenRouterEmbeddings(Embeddings):
    def __init__(self, api_key):
        self.api_key = api_key

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        res = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
                "input": text
            }
        )
        return res.json()["data"][0]["embedding"]


# =========================
# 2. LOAD DOCUMENT
# =========================

loader = TextLoader("cv.txt", encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)


# =========================
# 3. VECTOR STORE (FAISS)
# =========================

embeddings = OpenRouterEmbeddings(API_KEY)

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()


# =========================
# 4. OPENROUTER LLM
# =========================

llm = ChatOpenAI(
    openai_api_key=API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    model="openai/gpt-oss-120b:free",
    temperature=0.3
)


# =========================
# 5. TOOLS (AGENTIC RAG)
# =========================

def retrieve_tool(query: str):
    docs = retriever.invoke(query)    
    return "\n".join([d.page_content for d in docs])


# =========================
# 6. SIMPLE AGENT LOOP
# =========================

def agent(query: str):
    print("\n🔎 User Query:", query)

    # STEP 1: Retrieve context (tool usage)
    context = retrieve_tool(query)

    # STEP 2: Agent reasoning prompt
    prompt = f"""
You are an AI agent.

Use the context below to answer the question.
If context is not enough, say you are not sure.

CONTEXT:
{context}

QUESTION:
{query}

Answer step by step clearly:
"""

    # STEP 3: LLM response
    response = llm.invoke(prompt)

    return response.content


# =========================
# 7. RUN SYSTEM
# =========================

if __name__ == "__main__":
    while True:
        q = input("\nAsk something (or type exit): ")
        if q.lower() == "exit":
            break

        answer = agent(q)
        print("\n🤖 Answer:\n", answer)