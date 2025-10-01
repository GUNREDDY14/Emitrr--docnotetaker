#!/usr/bin/env python3
"""
Simple test for enhanced sentiment analysis without pytest dependency.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.services.sentiment_intent import (
    classify_utterance_sentiment,
    extract_intents,
    classify_session,
    analyze_patient_dialogue
)


def test_sample_patient_dialogue():
    """Test the sample input from the requirements."""
    sample_text = "I'm a bit worried about my back pain, but I hope it gets better soon."
    
    print(f"Testing sample text: '{sample_text}'")
    
    # Test utterance-level sentiment
    sentiment = classify_utterance_sentiment(sample_text)
    print(f"Sentiment: {sentiment}")
    assert sentiment in ["Anxious", "Neutral", "Reassured"]
    
    # Test intent extraction
    intents = extract_intents(sample_text)
    print(f"Intents: {intents}")
    assert isinstance(intents, list)
    assert len(intents) > 0
    
    # Test session-level analysis
    session_sentiment, session_intents = classify_session(sample_text)
    print(f"Session sentiment: {session_sentiment}")
    print(f"Session intents: {session_intents}")
    assert session_sentiment in ["Anxious", "Neutral", "Reassured"]
    assert isinstance(session_intents, list)
    
    # Test comprehensive analysis
    analysis = analyze_patient_dialogue(sample_text)
    print(f"Comprehensive analysis: {analysis}")
    assert "sentiment" in analysis
    assert "intent" in analysis
    assert "confidence" in analysis
    assert "analysis_method" in analysis
    
    return analysis


def test_anxious_sentiment():
    """Test anxious sentiment detection."""
    anxious_texts = [
        "I'm really worried about this pain",
        "I'm anxious about the diagnosis",
        "I'm concerned this might be serious"
    ]
    
    print("\nTesting anxious sentiment detection:")
    for text in anxious_texts:
        sentiment = classify_utterance_sentiment(text)
        print(f"'{text}' -> {sentiment}")
        assert sentiment == "Anxious", f"Expected Anxious for: {text}, got {sentiment}"


def test_reassured_sentiment():
    """Test reassured sentiment detection."""
    reassured_texts = [
        "That's great to hear, doctor",
        "I'm relieved to know it's not serious",
        "Thank you for reassuring me"
    ]
    
    print("\nTesting reassured sentiment detection:")
    for text in reassured_texts:
        sentiment = classify_utterance_sentiment(text)
        print(f"'{text}' -> {sentiment}")
        assert sentiment == "Reassured", f"Expected Reassured for: {text}, got {sentiment}"


def test_intent_detection():
    """Test intent detection."""
    print("\nTesting intent detection:")
    
    # Seeking reassurance
    reassurance_text = "Will I be okay? Should I worry about this?"
    intents = extract_intents(reassurance_text)
    print(f"'{reassurance_text}' -> {intents}")
    assert "Seeking reassurance" in intents
    
    # Reporting symptoms
    symptom_text = "I have pain in my neck and back"
    intents = extract_intents(symptom_text)
    print(f"'{symptom_text}' -> {intents}")
    assert "Reporting symptoms" in intents


if __name__ == "__main__":
    try:
        print("🧪 Testing Enhanced Sentiment Analysis")
        print("=" * 50)
        
        # Test the main sample
        result = test_sample_patient_dialogue()
        
        # Test other cases
        test_anxious_sentiment()
        test_reassured_sentiment()
        test_intent_detection()
        
        print("\n✅ All tests passed!")
        print(f"\n📊 Sample Result for '{result['sentiment']}' sentiment:")
        print(f"   Intent: {result['intent']}")
        print(f"   Method: {result['analysis_method']}")
        print(f"   Confidence: {result['confidence']}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)

