# backend/tests/test_sentiment_enhanced.py
"""
Test enhanced sentiment and intent analysis with transformer models.
"""
import pytest
from backend.services.sentiment_intent import (
    classify_utterance_sentiment,
    extract_intents,
    classify_session,
    analyze_patient_dialogue
)


def test_sample_patient_dialogue():
    """Test the sample input from the requirements."""
    sample_text = "I'm a bit worried about my back pain, but I hope it gets better soon."
    
    # Test utterance-level sentiment
    sentiment = classify_utterance_sentiment(sample_text)
    assert sentiment in ["Anxious", "Neutral", "Reassured"]
    
    # Test intent extraction
    intents = extract_intents(sample_text)
    assert isinstance(intents, list)
    assert len(intents) > 0
    
    # Test session-level analysis
    session_sentiment, session_intents = classify_session(sample_text)
    assert session_sentiment in ["Anxious", "Neutral", "Reassured"]
    assert isinstance(session_intents, list)
    
    # Test comprehensive analysis
    analysis = analyze_patient_dialogue(sample_text)
    assert "sentiment" in analysis
    assert "intent" in analysis
    assert "confidence" in analysis
    assert "analysis_method" in analysis
    
    print(f"Sample analysis result: {analysis}")


def test_anxious_sentiment():
    """Test anxious sentiment detection."""
    anxious_texts = [
        "I'm really worried about this pain",
        "I'm anxious about the diagnosis",
        "I'm concerned this might be serious",
        "I'm nervous about the treatment"
    ]
    
    for text in anxious_texts:
        sentiment = classify_utterance_sentiment(text)
        assert sentiment == "Anxious", f"Expected Anxious for: {text}, got {sentiment}"


def test_reassured_sentiment():
    """Test reassured sentiment detection."""
    reassured_texts = [
        "That's great to hear, doctor",
        "I'm relieved to know it's not serious",
        "Thank you for reassuring me",
        "I feel much better now"
    ]
    
    for text in reassured_texts:
        sentiment = classify_utterance_sentiment(text)
        assert sentiment == "Reassured", f"Expected Reassured for: {text}, got {sentiment}"


def test_intent_detection():
    """Test intent detection."""
    # Seeking reassurance
    reassurance_text = "Will I be okay? Should I worry about this?"
    intents = extract_intents(reassurance_text)
    assert "Seeking reassurance" in intents
    
    # Reporting symptoms
    symptom_text = "I have pain in my neck and back"
    intents = extract_intents(symptom_text)
    assert "Reporting symptoms" in intents
    
    # Expressing concern
    concern_text = "I'm concerned about my condition"
    intents = extract_intents(concern_text)
    assert "Expressing concern" in intents


def test_medical_context():
    """Test medical-specific sentiment analysis."""
    medical_texts = [
        "Doctor, I'm worried about my whiplash injury",
        "The physiotherapy sessions are helping",
        "I'm anxious about the prognosis",
        "Thank you doctor, I feel reassured"
    ]
    
    for text in medical_texts:
        analysis = analyze_patient_dialogue(text)
        assert analysis["sentiment"] in ["Anxious", "Neutral", "Reassured"]
        assert len(analysis["intent"]) > 0


if __name__ == "__main__":
    # Run a quick test
    test_sample_patient_dialogue()
    print("✅ Enhanced sentiment analysis tests passed!")

