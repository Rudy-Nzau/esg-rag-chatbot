import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import agent, chat

app = FastAPI(
    title="ESG RAG Chatbot API",
    description="Production-grade RAG pipeline + LLM Agent for ESG document querying",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/chat", tags=["rag"])
app.include_router(agent.router, prefix="/agent", tags=["agent"])

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "2.0.0"}

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
