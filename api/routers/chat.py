from fastapi import APIRouter, HTTPException
from api.schemas import ChatRequest, ChatResponse

router = APIRouter()

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    return ChatResponse(
        answer=f"[Pipeline not yet connected] You asked: '{request.query}'.",
        sources=[],
        num_chunks_retrieved=0,
        num_chunks_used=0,
    )
