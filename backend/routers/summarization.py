# backend/app/routers/summarization.py
from fastapi import APIRouter
from app.schemas import SummarizeRequest, SummarizeResponse
from services import summarizer

router = APIRouter(prefix="/summarization", tags=["summarization"])


@router.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    result = summarizer.summarize_text(req.text)
    return {"summary": result}
