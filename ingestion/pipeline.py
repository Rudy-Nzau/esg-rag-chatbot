"""
Pipeline d'ingestion de documents ESG
PDF → chunks → embeddings Mistral → ChromaDB
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import logging

load_dotenv()
logger = logging.getLogger(__name__)


def ingest_documents(file_paths: list[str]) -> Chroma:
    # 1. Charger les PDFs
    docs = []
    for path in file_paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
    print(f"✅ Loaded {len(docs)} pages")

    # 2. Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks")

    # 3. Embeddings via Mistral API
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=os.getenv("MISTRAL_API_KEY")
    )
    print("✅ Mistral embeddings ready")

    # 4. Stockage ChromaDB
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
    collection = os.getenv("COLLECTION_NAME", "esg_documents")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection,
    )
    print(f"✅ Stored {len(chunks)} chunks in ChromaDB → {persist_dir}")
    return vectorstore


if __name__ == "__main__":
    import glob
    pdfs = glob.glob("data/raw/*.pdf")
    if not pdfs:
        print("❌ No PDFs found in data/raw/")
    else:
        print(f"Found {len(pdfs)} PDF(s): {pdfs}")
        ingest_documents(pdfs)
        print("🎉 Ingestion complete!")
