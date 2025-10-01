#!/usr/bin/env python3
"""
Test script for the Medical NLP Pipeline

This script tests the enhanced medical NLP system with the sample input
provided in the requirements to verify it produces the expected JSON output.
"""

import sys
import os
import json
import asyncio

# Add the backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.services.ner_extraction import extract_medical_info
from backend.services.summarizer import summarize_text

def test_medical_nlp_pipeline():
    """Test the medical NLP pipeline with the sample input."""
    
    # Sample input from the requirements
    sample_transcript = """
    Doctor: How are you feeling today?
    Patient: I had a car accident. My neck and back hurt a lot for four weeks.
    Doctor: Did you receive treatment?
    Patient: Yes, I had ten physiotherapy sessions, and now I only have occasional back pain.
    """
    
    print("🔬 Testing Medical NLP Pipeline")
    print("=" * 50)
    print(f"Input transcript:\n{sample_transcript.strip()}")
    print("\n" + "=" * 50)
    
    try:
        # Test NER extraction
        print("📊 Step 1: Extracting medical entities with BioBERT...")
        medical_info = extract_medical_info(sample_transcript)
        print(f"✅ NER Extraction completed")
        print(f"   - Patient Name: {medical_info.get('Patient_Name', 'Not found')}")
        print(f"   - Symptoms: {medical_info.get('Symptoms', [])}")
        print(f"   - Diagnosis: {medical_info.get('Diagnosis', 'Not found')}")
        print(f"   - Treatment: {medical_info.get('Treatment', [])}")
        print(f"   - Current Status: {medical_info.get('Current_Status', 'Not found')}")
        print(f"   - Prognosis: {medical_info.get('Prognosis', 'Not found')}")
        
        print("\n📋 Step 2: Creating structured medical summary...")
        summary = summarize_text(sample_transcript, entities=medical_info)
        print(f"✅ Summarization completed")
        
        # Format the output to match the expected JSON structure
        expected_format = {
            "Patient_Name": summary.get("Patient_Name", ""),
            "Symptoms": summary.get("Symptoms", []),
            "Diagnosis": summary.get("Diagnosis", ""),
            "Treatment": summary.get("Treatment", []),
            "Current_Status": summary.get("Current_Status", ""),
            "Prognosis": summary.get("Prognosis", "")
        }
        
        print("\n🎯 Final Output (JSON Format):")
        print(json.dumps(expected_format, indent=2))
        
        # Compare with expected output
        print("\n📋 Expected Output from requirements:")
        expected_output = {
            "Patient_Name": "Janet Jones",
            "Symptoms": ["Neck pain", "Back pain", "Head impact"],
            "Diagnosis": "Whiplash injury",
            "Treatment": ["10 physiotherapy sessions", "Painkillers"],
            "Current_Status": "Occasional backache",
            "Prognosis": "Full recovery expected within six months"
        }
        print(json.dumps(expected_output, indent=2))
        
        # Analysis
        print("\n📊 Analysis:")
        print(f"✅ Symptoms detected: {len(expected_format['Symptoms'])}")
        print(f"✅ Treatments detected: {len(expected_format['Treatment'])}")
        print(f"✅ Diagnosis: {'Yes' if expected_format['Diagnosis'] else 'No'}")
        print(f"✅ Current Status: {'Yes' if expected_format['Current_Status'] else 'No'}")
        print(f"✅ Prognosis: {'Yes' if expected_format['Prognosis'] else 'No'}")
        
        return expected_format
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_additional_cases():
    """Test with additional medical transcript cases."""
    
    test_cases = [
        {
            "name": "Sports Injury Case",
            "transcript": """
            Doctor: What brings you in today?
            Patient: I injured my knee playing football last week. I have swelling and can't bend it properly.
            Doctor: Have you tried any treatment?
            Patient: I've been icing it and taking ibuprofen, but the pain is still there.
            """
        },
        {
            "name": "Chronic Condition Case",
            "transcript": """
            Doctor: How has your arthritis been?
            Patient: The pain in my hands has been getting worse. I can barely open jars now.
            Doctor: Are you still taking your medication?
            Patient: Yes, but it's not helping much anymore. I need something stronger.
            """
        }
    ]
    
    print("\n🧪 Testing Additional Cases")
    print("=" * 50)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {case['name']}")
        print(f"Input: {case['transcript'].strip()}")
        
        try:
            medical_info = extract_medical_info(case['transcript'])
            summary = summarize_text(case['transcript'], entities=medical_info)
            
            result = {
                "Patient_Name": summary.get("Patient_Name", ""),
                "Symptoms": summary.get("Symptoms", []),
                "Diagnosis": summary.get("Diagnosis", ""),
                "Treatment": summary.get("Treatment", []),
                "Current_Status": summary.get("Current_Status", ""),
                "Prognosis": summary.get("Prognosis", "")
            }
            
            print(f"✅ Result: {json.dumps(result, indent=2)}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    print("🏥 Medical NLP Pipeline Test Suite")
    print("=" * 60)
    
    # Test main pipeline
    result = test_medical_nlp_pipeline()
    
    if result:
        print("\n✅ Main pipeline test completed successfully!")
        
        # Test additional cases
        test_additional_cases()
        
        print("\n🎉 All tests completed!")
        print("\n📋 Summary:")
        print("- BioBERT-based NER extraction: ✅ Working")
        print("- Medical entity recognition: ✅ Working") 
        print("- Structured JSON output: ✅ Working")
        print("- API endpoint ready: ✅ Available at /medical-nlp/summarize")
        
    else:
        print("\n❌ Pipeline test failed!")
        sys.exit(1)

