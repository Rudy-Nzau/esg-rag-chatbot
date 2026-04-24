"""
Hybrid Retriever — BM25 + Semantic Search + CrossEncoder Reranking
"""

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()


def build_hybrid_retriever(documents, vectorstore):
    """
    Combine BM25 (lexical) + semantic search (vector)
    avec fusion RRF (Reciprocal Rank Fusion)
    """
    # BM25 sur les documents bruts
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 4

    # Semantic retriever depuis ChromaDB
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Fusion 50/50
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.5, 0.5]
    )

    print("✅ Hybrid retriever built (BM25 + Semantic)")
    return hybrid_retriever
