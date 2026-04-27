"""
ESG RAG Agent — LangGraph ReAct Agent
"""

from langgraph.prebuilt import create_react_agent
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import os
import glob

load_dotenv()

embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    mistral_api_key=os.getenv("MISTRAL_API_KEY")
)
vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    collection_name="esg_documents_qdrant",
)

_documents = []
_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
for pdf in glob.glob("data/raw/*.pdf"):
    pages = PyPDFLoader(pdf).load()
    _documents.extend(_splitter.split_documents(pages))

_tokenized = [doc.page_content.lower().split() for doc in _documents]
_bm25 = BM25Okapi(_tokenized) if _documents else None


def _hybrid_search(query: str, k: int = 6) -> list:
    semantic = vectorstore.similarity_search(query, k=k)
    if _bm25 is None:
        return semantic
    semantic_ids = {doc.page_content for doc in semantic}
    tokens = query.lower().split()
    scores = _bm25.get_scores(tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    bm25_docs = [_documents[i] for i in top_idx]
    merged = list(semantic)
    for doc in bm25_docs:
        if doc.page_content not in semantic_ids:
            merged.append(doc)
    return merged[:k]


@tool
def search_esrs_documents(query: str) -> str:
    """Search ESRS regulatory documents for specific requirements or definitions."""
    docs = _hybrid_search(query, k=6)
    results = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown").split("/")[-1]
        results.append(f"[{i+1}] (Source: {source})\n{doc.page_content}")
    return "\n\n".join(results)


@tool
def compare_esrs_requirements(topic_a: str, topic_b: str) -> str:
    """Compare two different ESRS requirements or topics."""
    docs_a = _hybrid_search(topic_a, k=4)
    docs_b = _hybrid_search(topic_b, k=4)
    context_a = "\n".join([doc.page_content for doc in docs_a])
    context_b = "\n".join([doc.page_content for doc in docs_b])
    return f"=== Context for '{topic_a}' ===\n{context_a}\n\n=== Context for '{topic_b}' ===\n{context_b}"


@tool
def summarize_esrs_section(section: str) -> str:
    """Retrieve and summarize a specific ESRS section or disclosure requirement."""
    docs = _hybrid_search(section, k=8)
    context = "\n\n".join([doc.page_content for doc in docs])
    return f"Retrieved content for '{section}':\n{context}"


llm = ChatMistralAI(
    model="mistral-small-latest",
    mistral_api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0,
)

tools = [search_esrs_documents, compare_esrs_requirements, summarize_esrs_section]

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=(
        "You are an expert ESG analyst assistant with deep knowledge of ESRS standards. "
        "Always cite your sources and base your answers strictly on the retrieved documents. "
        "Never fabricate regulatory requirements."
    )
)

print("✅ ESG Agent initialized with 3 tools")


def run_agent(question: str) -> dict:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    final_message = result["messages"][-1]
    answer = final_message.content
    if isinstance(answer, list):
        answer = " ".join([item.get("text", "") for item in answer if isinstance(item, dict) and "text" in item])
    tool_calls = [m for m in result["messages"] if hasattr(m, "tool_calls") and m.tool_calls]
    return {
        "answer": answer,
        "reasoning_steps": len(tool_calls),
        "total_messages": len(result["messages"])
    }
