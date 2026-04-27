"""
Core RAG pipeline — ESG chatbot
Hybrid retrieval (BM25 + Qdrant) → CrossEncoder Reranking → Mistral generation
"""

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from rag.prompts import ESG_RAG_PROMPT
from rag.reranker import CrossEncoderReranker
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import os
import glob
import logging

load_dotenv()
logger = logging.getLogger(__name__)


class ESGRAGPipeline:
    def __init__(self):
        self.embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            mistral_api_key=os.getenv("MISTRAL_API_KEY")
        )
        self.vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=self.embeddings,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name="esg_documents_qdrant",
        )
        self.llm = ChatMistralAI(
            model="mistral-small-latest",
            mistral_api_key=os.getenv("MISTRAL_API_KEY"),
            temperature=0,
        )
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=ESG_RAG_PROMPT,
        )
        self.reranker = CrossEncoderReranker(top_k=4)

        # Documents pour BM25
        self.documents = self._load_documents()
        if self.documents:
            tokenized = [doc.page_content.lower().split() for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized)
            print(f"✅ Hybrid RAG Pipeline (Qdrant) — {len(self.documents)} chunks")
        else:
            self.bm25 = None
            print("✅ Semantic-only RAG Pipeline (Qdrant)")

    def _load_documents(self):
        docs = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        for pdf in glob.glob("data/raw/*.pdf"):
            pages = PyPDFLoader(pdf).load()
            docs.extend(splitter.split_documents(pages))
        return docs

    def _hybrid_retrieve(self, question: str, k: int = 8):
        semantic_docs = self.vectorstore.similarity_search(question, k=k)
        semantic_ids = {doc.page_content for doc in semantic_docs}

        if self.bm25 is None:
            return semantic_docs

        tokens = question.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        bm25_docs = [self.documents[i] for i in top_idx]

        merged = list(semantic_docs)
        for doc in bm25_docs:
            if doc.page_content not in semantic_ids:
                merged.append(doc)
        return merged

    def query(self, question: str) -> dict:
        candidates = self._hybrid_retrieve(question, k=8)
        reranked = self.reranker.rerank(question, candidates)

        context_parts = []
        sources = []
        for i, doc in enumerate(reranked):
            context_parts.append(f"[{i+1}] {doc.page_content}")
            source = doc.metadata.get("source", "unknown")
            if source not in sources:
                sources.append(source)

        context = "\n\n".join(context_parts)
        formatted_prompt = self.prompt.format(context=context, question=question)
        response = self.llm.invoke(formatted_prompt)

        return {
            "answer": response.content,
            "sources": sources,
            "num_chunks_retrieved": len(candidates),
            "num_chunks_used": len(reranked),
        }
