# backend/app/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class TranscriptRequest(BaseModel):
    text: str = Field(..., description="Full transcript or clinical dialog text")
    patient_name: Optional[str] = Field(None)
    metadata: Optional[Dict[str, Any]] = None


class TranscriptResponse(BaseModel):
    conversation_id: int
    transcript: str
    entities: Dict[str, Any]
    summary: Dict[str, Any]
    soap_note: Dict[str, Any]


class SummarizeRequest(BaseModel):
    text: str


class SummarizeResponse(BaseModel):
    summary: Dict[str, Any]


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    Sentiment: str
    Intent: str
    confidence: Optional[float] = None
    analysis_method: Optional[str] = None


class SOAPRequest(BaseModel):
    text: str


class SOAPResponse(BaseModel):
    soap: Dict[str, Any]
