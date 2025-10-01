#!/usr/bin/env python3
"""
Test script for patient name extraction functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.utils.helpers import extract_patient_name

def test_name_extraction():
    """Test various patterns for patient name extraction"""
    
    test_cases = [
        # Basic patterns
        ("I'm John Smith", "John Smith"),
        ("I am Mary Johnson", "Mary Johnson"),
        ("My name is Robert Brown", "Robert Brown"),
        ("My name's Sarah Wilson", "Sarah Wilson"),
        ("This is Michael Davis", "Michael Davis"),
        ("Call me Emma", "Emma"),
        ("You can call me David", "David"),
        ("I go by Lisa", "Lisa"),
        
        # With titles
        ("I'm Mr. John Smith", "John Smith"),
        ("I am Ms. Mary Johnson", "Mary Johnson"),
        ("My name is Mrs. Robert Brown", "Robert Brown"),
        ("This is Dr. Sarah Wilson", "Sarah Wilson"),
        
        # With greetings
        ("Hi, I'm John Smith", "John Smith"),
        ("Hello, I'm Ms. Mary Johnson", "Mary Johnson"),
        ("Hey, I'm Robert Brown", "Robert Brown"),
        
        # With speaker labels
        ("Patient: I'm John Smith", "John Smith"),
        ("Pt: I am Mary Johnson", "Mary Johnson"),
        
        # Complex cases
        ("Doctor: How are you feeling today? Patient: Hi, I'm John Smith and I've been having chest pain.", "John Smith"),
        ("I'm Dr. Sarah Wilson, but you can call me Sarah", "Sarah Wilson"),
        
        # Edge cases
        ("I'm A", None),  # Too short
        ("I'm not feeling well", None),  # No name
        ("", None),  # Empty string
    ]
    
    print("Testing Patient Name Extraction")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for i, (text, expected) in enumerate(test_cases, 1):
        result = extract_patient_name(text)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        
        print(f"Test {i:2d}: {status}")
        print(f"  Input:    '{text}'")
        print(f"  Expected: {expected}")
        print(f"  Got:      {result}")
        print()
        
        if result == expected:
            passed += 1
        else:
            failed += 1
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {failed} tests failed")
    
    return failed == 0

if __name__ == "__main__":
    test_name_extraction()
