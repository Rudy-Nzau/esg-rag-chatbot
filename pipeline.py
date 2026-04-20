"""
Core RAG pipeline — ESG chatbot
Implements: hybrid retrieval → reranking → grounded generation
"""

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from rag.retriever import HybridRetriever
from rag.reranker import CrossEncoderReranker
from rag.prompts import ESG_RAG_PROMPT
import logging

logger = logging.getLogger(__name__)


class ESGRAGPipeline:
    def __init__(self, vectorstore, model_name: str = "gpt-4o"):
        self.retriever = HybridRetriever(vectorstore=vectorstore, top_k=10)
        self.reranker = CrossEncoderReranker(top_k=4)
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=ESG_RAG_PROMPT,
        )

    def query(self, question: str) -> dict:
        """
        Full RAG pipeline:
        1. Hybrid retrieval (semantic + BM25)
        2. CrossEncoder reranking
        3. Grounded generation with citations
        """
        logger.info(f"Processing query: {question[:100]}...")

        # Step 1: Retrieve candidates
        candidates = self.retriever.retrieve(question)
        logger.info(f"Retrieved {len(candidates)} candidates")

        # Step 2: Rerank
        reranked = self.reranker.rerank(question, candidates)
        logger.info(f"Reranked to top {len(reranked)} chunks")

        # Step 3: Build context with source metadata
        context_parts = []
        sources = []
        for i, doc in enumerate(reranked):
            source = doc.metadata.get("source", "unknown")
            context_parts.append(f"[{i+1}] {doc.page_content}")
            sources.append(source)

        context = "\n\n".join(context_parts)

        # Step 4: Generate grounded answer
        formatted_prompt = self.prompt.format(
            context=context,
            question=question
        )
        response = self.llm.invoke(formatted_prompt)

        return {
            "answer": response.content,
            "sources": list(set(sources)),
            "num_chunks_retrieved": len(candidates),
            "num_chunks_used": len(reranked),
        }
