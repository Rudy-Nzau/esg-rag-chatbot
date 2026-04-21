"""
Core RAG pipeline — ESG chatbot
Retrieval ChromaDB → génération Mistral
"""

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from rag.prompts import ESG_RAG_PROMPT
from dotenv import load_dotenv
import os
import logging

load_dotenv()
logger = logging.getLogger(__name__)


class ESGRAGPipeline:
    def __init__(self):
        embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            mistral_api_key=os.getenv("MISTRAL_API_KEY")
        )
        self.vectorstore = Chroma(
            persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"),
            embedding_function=embeddings,
            collection_name=os.getenv("COLLECTION_NAME", "esg_documents"),
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
        print("✅ RAG Pipeline initialized")

    def query(self, question: str) -> dict:
        # 1. Retrieval
        docs = self.vectorstore.similarity_search(question, k=4)
        logger.info(f"Retrieved {len(docs)} chunks")

        # 2. Build context
        context_parts = []
        sources = []
        for i, doc in enumerate(docs):
            context_parts.append(f"[{i+1}] {doc.page_content}")
            source = doc.metadata.get("source", "unknown")
            if source not in sources:
                sources.append(source)

        context = "\n\n".join(context_parts)

        # 3. Generate
        formatted_prompt = self.prompt.format(
            context=context,
            question=question
        )
        response = self.llm.invoke(formatted_prompt)

        return {
            "answer": response.content,
            "sources": sources,
            "num_chunks_retrieved": len(docs),
            "num_chunks_used": len(docs),
        }
