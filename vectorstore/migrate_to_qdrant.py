"""
Migration ChromaDB → Qdrant
Réingère tous les documents dans Qdrant
"""

from langchain_qdrant import QdrantVectorStore
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv
import os
import glob

load_dotenv()

COLLECTION_NAME = "esg_documents_qdrant"
VECTOR_SIZE = 1024  # Mistral embed dimension

def migrate():
    print("🚀 Starting migration ChromaDB → Qdrant...")

    # 1. Charger les documents
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    for pdf in glob.glob("data/raw/*.pdf"):
        pages = PyPDFLoader(pdf).load()
        chunks = splitter.split_documents(pages)
        docs.extend(chunks)
        print(f"✅ Loaded {len(chunks)} chunks from {pdf.split('/')[-1]}")

    print(f"📄 Total chunks: {len(docs)}")

    # 2. Setup Qdrant client
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    # 3. Créer la collection
    if client.collection_exists(COLLECTION_NAME):
        print(f"⚠️  Collection '{COLLECTION_NAME}' already exists — deleting and recreating")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"✅ Collection '{COLLECTION_NAME}' created")

    # 4. Embeddings
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=os.getenv("MISTRAL_API_KEY")
    )

    # 5. Ingérer dans Qdrant
    print("⏳ Ingesting documents into Qdrant (this may take a minute)...")
    vectorstore = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=COLLECTION_NAME,
    )

    print(f"✅ Migration complete — {len(docs)} chunks in Qdrant")
    
    # 6. Test rapide
    results = vectorstore.similarity_search("health and safety requirements", k=2)
    print(f"\n🧪 Test query returned {len(results)} results")
    print(f"First result preview: {results[0].page_content[:150]}...")

    return vectorstore

if __name__ == "__main__":
    migrate()
