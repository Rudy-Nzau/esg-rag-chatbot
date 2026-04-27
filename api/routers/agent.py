from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from rag.agent import run_agent

router = APIRouter()

class AgentRequest(BaseModel):
    query: str

class AgentResponse(BaseModel):
    answer: str
    reasoning_steps: int
    total_messages: int

@router.post("/", response_model=AgentResponse)
async def agent_chat(request: AgentRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    result = run_agent(request.query)
    # Gérer le cas où answer est une liste
    answer = result["answer"]
    if isinstance(answer, list):
        answer = " ".join([item.get("text", "") for item in answer if isinstance(item, dict) and "text" in item])
    return AgentResponse(
        answer=answer,
        reasoning_steps=result["reasoning_steps"],
        total_messages=result["total_messages"]
    )
