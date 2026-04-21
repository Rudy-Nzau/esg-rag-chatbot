from fastapi import APIRouter, HTTPException
from api.schemas import ChatRequest, ChatResponse
from rag.pipeline import ESGRAGPipeline

router = APIRouter()
pipeline = ESGRAGPipeline()

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    result = pipeline.query(request.query)
    return ChatResponse(**result)
