"""
ESG RAG Agent — LangGraph ReAct Agent
Capable de raisonner en plusieurs étapes sur les documents ESRS
"""

from langgraph.prebuilt import create_react_agent
from langchain_mistralai import ChatMistralAI
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import os
import glob

load_dotenv()

# Setup embeddings et vectorstore
embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    mistral_api_key=os.getenv("MISTRAL_API_KEY")
)
vectorstore = Chroma(
    persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"),
    embedding_function=embeddings,
    collection_name=os.getenv("COLLECTION_NAME", "esg_documents"),
)

# Charger les documents pour BM25
_documents = []
_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
for pdf in glob.glob("data/raw/*.pdf"):
    pages = PyPDFLoader(pdf).load()
    _documents.extend(_splitter.split_documents(pages))

_tokenized = [doc.page_content.lower().split() for doc in _documents]
_bm25 = BM25Okapi(_tokenized)


def _hybrid_search(query: str, k: int = 6) -> list:
    """Hybrid BM25 + semantic search"""
    semantic = vectorstore.similarity_search(query, k=k)
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


# Outils disponibles pour l'agent
@tool
def search_esrs_documents(query: str) -> str:
    """
    Search ESRS regulatory documents for information.
    Use this tool to find specific requirements, definitions,
    or disclosure obligations in ESG standards.
    Input: a specific search query about ESRS requirements.
    """
    docs = _hybrid_search(query, k=6)
    results = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown").split("/")[-1]
        results.append(f"[{i+1}] (Source: {source})\n{doc.page_content}")
    return "\n\n".join(results)


@tool
def compare_esrs_requirements(topic_a: str, topic_b: str) -> str:
    """
    Compare two different ESRS requirements or topics.
    Use this tool when asked to compare, contrast or find differences
    between two ESRS concepts, standards or disclosure requirements.
    Input: two topics to compare separately.
    """
    docs_a = _hybrid_search(topic_a, k=4)
    docs_b = _hybrid_search(topic_b, k=4)

    context_a = "\n".join([doc.page_content for doc in docs_a])
    context_b = "\n".join([doc.page_content for doc in docs_b])

    return f"=== Context for '{topic_a}' ===\n{context_a}\n\n=== Context for '{topic_b}' ===\n{context_b}"


@tool
def summarize_esrs_section(section: str) -> str:
    """
    Retrieve and summarize a specific ESRS section or disclosure requirement.
    Use this for broad questions about what a specific section covers.
    Input: the name or code of the ESRS section (e.g. 'S1-14', 'Appendix A').
    """
    docs = _hybrid_search(section, k=8)
    context = "\n\n".join([doc.page_content for doc in docs])
    return f"Retrieved content for '{section}':\n{context}"


# Créer l'agent
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
        "You have access to tools to search and analyze ESRS regulatory documents. "
        "Always cite your sources and base your answers strictly on the retrieved documents. "
        "For complex questions, use multiple tool calls to gather comprehensive information. "
        "Never fabricate regulatory requirements."
    )
)

print("✅ ESG Agent initialized with 3 tools")


def run_agent(question: str) -> dict:
    """Run the ESG agent on a question"""
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    
    # Extraire la réponse finale
    final_message = result["messages"][-1]
    answer = final_message.content
    
    # Compter les étapes de raisonnement
    tool_calls = [m for m in result["messages"] if hasattr(m, "tool_calls") and m.tool_calls]
    
    return {
        "answer": answer,
        "reasoning_steps": len(tool_calls),
        "total_messages": len(result["messages"])
    }


if __name__ == "__main__":
    print("\n🧪 Test 1 — Simple query:")
    result = run_agent("What does S1-14 require?")
    print(f"Answer: {result['answer'][:300]}...")
    print(f"Reasoning steps: {result['reasoning_steps']}")

    print("\n🧪 Test 2 — Complex comparison:")
    result = run_agent("What are the differences between S1-14 health and safety requirements and the general workforce disclosure requirements in ESRS S1?")
    print(f"Answer: {result['answer'][:300]}...")
    print(f"Reasoning steps: {result['reasoning_steps']}")
