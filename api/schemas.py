from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    query: str
    collection_name: Optional[str] = "esg_documents"

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    num_chunks_retrieved: int
    num_chunks_used: int

class IngestResponse(BaseModel):
    status: str
    documents_ingested: int
    collection_name: str
