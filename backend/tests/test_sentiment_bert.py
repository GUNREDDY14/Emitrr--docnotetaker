#!/usr/bin/env python3
"""
Test script for the enhanced BERT-based sentiment analysis.
Tests the sentiment analysis with the example provided by the user.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.services.sentiment_intent import analyze_patient_dialogue

def test_sentiment_analysis():
    """Test the sentiment analysis with various medical conversation examples."""
    
    test_cases = [
        {
            "input": "I'm a bit worried about my back pain, but I hope it gets better soon.",
            "expected_sentiment": "Anxious",
            "expected_intent": "Seeking reassurance"
        },
        {
            "input": "Thank you doctor, that's very reassuring to hear.",
            "expected_sentiment": "Reassured", 
            "expected_intent": "Expressing concern"
        },
        {
            "input": "I have been experiencing chest pain for the past week.",
            "expected_sentiment": "Neutral",
            "expected_intent": "Reporting symptoms"
        },
        {
            "input": "Should I be worried about this headache?",
            "expected_sentiment": "Anxious",
            "expected_intent": "Seeking reassurance"
        },
        {
            "input": "The medication is working well and I feel much better now.",
            "expected_sentiment": "Reassured",
            "expected_intent": "Reporting symptoms"
        }
    ]
    
    print("🧪 Testing Enhanced BERT Sentiment Analysis")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}:")
        print(f"Input: {test_case['input']}")
        
        try:
            result = analyze_patient_dialogue(test_case['input'])
            
            print(f"✅ Result: {result}")
            
            # Check if the result matches expected format
            if 'Sentiment' in result and 'Intent' in result:
                print(f"✅ Format: Correct JSON structure")
                
                # Check sentiment
                if result['Sentiment'] == test_case['expected_sentiment']:
                    print(f"✅ Sentiment: {result['Sentiment']} (Expected: {test_case['expected_sentiment']})")
                else:
                    print(f"⚠️  Sentiment: {result['Sentiment']} (Expected: {test_case['expected_sentiment']})")
                
                # Check intent
                if result['Intent'] == test_case['expected_intent']:
                    print(f"✅ Intent: {result['Intent']} (Expected: {test_case['expected_intent']})")
                else:
                    print(f"⚠️  Intent: {result['Intent']} (Expected: {test_case['expected_intent']})")
                    
            else:
                print(f"❌ Format: Missing required fields")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Testing Complete!")

if __name__ == "__main__":
    test_sentiment_analysis()
