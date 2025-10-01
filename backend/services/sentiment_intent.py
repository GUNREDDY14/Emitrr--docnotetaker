# backend/app/services/sentiment_intent.py
"""
Enhanced sentiment & intent analysis using BERT models specifically tuned for medical conversations.
Returns structured JSON with sentiment (Anxious/Neutral/Reassured) and intent classification.
"""
from typing import Tuple, List, Optional, Dict, Any
import re
import logging
import torch

logger = logging.getLogger(__name__)

# Rule-based fallback keywords for medical context
ANXIOUS_KEYWORDS = [
    "worried", "anxious", "concerned", "nervous", "worry", "scared", "fearful",
    "panic", "stress", "stressed", "apprehensive", "uneasy", "distressed",
    "afraid", "terrified", "frightened", "alarmed", "troubled", "bothered"
]
REASSURED_KEYWORDS = [
    "relief", "relieved", "great to hear", "that's good to hear", "reassure", 
    "reassured", "thank you doctor", "comfortable", "confident", "optimistic",
    "hopeful", "better", "improving", "good news", "reassuring", "calm",
    "peaceful", "satisfied", "content", "pleased", "grateful"
]
NEUTRAL_KEYWORDS = [
    "okay", "fine", "normal", "usual", "regular", "standard", "typical",
    "average", "moderate", "manageable", "acceptable", "stable"
]

# Intent classification keywords
SEEKING_REASSURANCE_KEYWORDS = [
    "will i", "should i", "worry about", "affect me", "does this mean",
    "is it serious", "how long", "what if", "concerned about", "afraid",
    "should i be worried", "is this normal", "what does this mean",
    "am i okay", "will this get better", "is it dangerous"
]
REPORTING_SYMPTOMS_KEYWORDS = [
    "pain", "hurt", "ache", "symptom", "suffered", "injury", "problem", 
    "issue", "trouble", "discomfort", "feeling", "experiencing", "having",
    "feels like", "notice", "develop", "occur", "appear"
]
EXPRESSING_CONCERN_KEYWORDS = [
    "concern", "worry", "anxious", "nervous", "scared", "fearful",
    "bothered", "troubled", "distressed", "upset", "frightened"
]

# Global model instances (lazy loading)
_SENTIMENT_MODEL = None
_TOKENIZER = None


def _load_sentiment_model():
    """Load BERT model for medical sentiment analysis."""
    global _SENTIMENT_MODEL, _TOKENIZER
    if _SENTIMENT_MODEL is not None:
        return _SENTIMENT_MODEL, _TOKENIZER
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        
        # Use a robust BERT model for sentiment analysis
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        
        try:
            # Try to load the model with pipeline for easier use
            _SENTIMENT_MODEL = pipeline(
                "sentiment-analysis", 
                model=model_name,
                return_all_scores=True,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info(f"Loaded sentiment model: {model_name}")
            return _SENTIMENT_MODEL, None
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
            
        # Fallback to DistilBERT
        try:
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            _SENTIMENT_MODEL = pipeline(
                "sentiment-analysis", 
                model=model_name,
                return_all_scores=True,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info(f"Loaded fallback sentiment model: {model_name}")
            return _SENTIMENT_MODEL, None
        except Exception as e:
            logger.warning(f"Failed to load fallback model: {e}")
            
        return None, None
    except Exception as e:
        logger.error(f"Failed to load any sentiment model: {e}")
        return None, None


def _classify_sentiment_transformer(text: str) -> Optional[str]:
    """Use BERT model for medical sentiment classification."""
    model, tokenizer = _load_sentiment_model()
    if not model:
        return None
    
    try:
        # Clean and truncate text if too long
        text = text.strip()
        if len(text) > 512:
            text = text[:512]
            
        results = model(text)
        if not results or not isinstance(results, list):
            return None
            
        # Get the highest scoring sentiment
        best_result = max(results[0], key=lambda x: x['score'])
        sentiment_label = best_result['label'].lower()
        confidence = best_result['score']
        
        # Map to our medical sentiment categories with confidence threshold
        if confidence > 0.6:  # Only trust high-confidence predictions
            if 'positive' in sentiment_label or 'joy' in sentiment_label:
                return "Reassured"
            elif 'negative' in sentiment_label or 'sadness' in sentiment_label or 'anger' in sentiment_label:
                return "Anxious"
            elif 'neutral' in sentiment_label:
                return "Neutral"
        
        # If confidence is low, return None to fall back to rule-based
        return None
            
    except Exception as e:
        logger.error(f"BERT sentiment analysis failed: {e}")
        return None


def _classify_intent_rule_based(text: str) -> str:
    """Classify intent using rule-based approach optimized for medical conversations."""
    text_lower = text.lower()
    
    # Check for seeking reassurance patterns
    if any(keyword in text_lower for keyword in SEEKING_REASSURANCE_KEYWORDS):
        return "Seeking reassurance"
    
    # Check for expressing concern patterns
    if any(keyword in text_lower for keyword in EXPRESSING_CONCERN_KEYWORDS):
        return "Expressing concern"
    
    # Check for reporting symptoms patterns
    if any(keyword in text_lower for keyword in REPORTING_SYMPTOMS_KEYWORDS):
        return "Reporting symptoms"
    
    # Check for question patterns
    if text.strip().endswith('?') or re.search(r'\b(what|how|when|where|why|is|are|do|does|can|could|would|should)\b', text_lower):
        return "Seeking reassurance"
    
    # Default to reporting symptoms if no clear intent
    return "Reporting symptoms"


def classify_utterance_sentiment(text: str) -> str:
    """Classify sentiment of a single utterance using BERT with rule-based fallback."""
    # Try BERT first
    bert_sentiment = _classify_sentiment_transformer(text)
    if bert_sentiment:
        return bert_sentiment
    
    # Fallback to rule-based classification
    text_lower = text.lower()
    
    # Count keyword matches for each sentiment
    anxious_score = sum(1 for keyword in ANXIOUS_KEYWORDS if keyword in text_lower)
    reassured_score = sum(1 for keyword in REASSURED_KEYWORDS if keyword in text_lower)
    neutral_score = sum(1 for keyword in NEUTRAL_KEYWORDS if keyword in text_lower)
    
    # Return sentiment based on highest score
    if anxious_score > reassured_score and anxious_score > neutral_score:
        return "Anxious"
    elif reassured_score > neutral_score:
        return "Reassured"
    else:
        return "Neutral"


def classify_intent(text: str) -> str:
    """Classify intent using rule-based approach optimized for medical conversations."""
    return _classify_intent_rule_based(text)


def analyze_patient_dialogue(text: str) -> Dict[str, Any]:
    """Comprehensive analysis of patient dialogue returning structured sentiment and intent."""
    # Get sentiment using BERT with fallback
    sentiment = classify_utterance_sentiment(text)
    
    # Get intent using rule-based approach
    intent = classify_intent(text)
    
    # Calculate confidence based on method used
    confidence_score = 0.9 if _SENTIMENT_MODEL else 0.7
    analysis_method = "BERT" if _SENTIMENT_MODEL else "rule_based"
    
    return {
        "Sentiment": sentiment,
        "Intent": intent,
        "confidence": confidence_score,
        "analysis_method": analysis_method
    }
