from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import chat, ingest
import uvicorn

app = FastAPI(
    title="ESG RAG Chatbot API",
    description="Production-grade RAG pipeline for ESG document querying",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(ingest.router, prefix="/ingest", tags=["ingestion"])

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
