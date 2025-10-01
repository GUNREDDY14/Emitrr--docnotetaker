# backend/app/routers/sentiment.py
from fastapi import APIRouter
from app.schemas import SentimentRequest, SentimentResponse
from services import sentiment_intent

router = APIRouter(prefix="/sentiment", tags=["sentiment"])


@router.post("/analyze", response_model=SentimentResponse)
def analyze(req: SentimentRequest):
    """Analyze sentiment and intent of patient dialogue using transformer models with fallback."""
    analysis = sentiment_intent.analyze_patient_dialogue(req.text)
    return SentimentResponse(**analysis)


@router.post("/analyze-utterance", response_model=SentimentResponse)
def analyze_utterance(req: SentimentRequest):
    """Analyze sentiment of a single utterance."""
    sentiment = sentiment_intent.classify_utterance_sentiment(req.text)
    intent = sentiment_intent.classify_intent(req.text)
    
    return SentimentResponse(
        Sentiment=sentiment,
        Intent=intent,
        analysis_method="BERT" if sentiment_intent._SENTIMENT_MODEL else "rule_based"
    )
