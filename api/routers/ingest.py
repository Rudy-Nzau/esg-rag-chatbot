from fastapi import APIRouter, UploadFile, File, HTTPException
from api.schemas import IngestResponse
from typing import List
import os

router = APIRouter()

@router.post("/", response_model=IngestResponse)
async def ingest(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    supported = {".pdf", ".txt", ".json", ".csv"}
    ingested = 0
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in supported:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
        ingested += 1
    return IngestResponse(status="success", documents_ingested=ingested, collection_name="esg_documents")
