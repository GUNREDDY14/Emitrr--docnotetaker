# backend/utils/helpers.py

from typing import Dict, Any, List, Optional
import re
import logging

logger = logging.getLogger(__name__)


def extract_patient_name(text: str) -> Optional[str]:
    """Extract patient name from conversation text using pattern matching."""
    if not text:
        return None
    
    # Patterns to match various ways people introduce themselves
    patterns = [
        # "I'm John Smith" / "I am John Smith"
        r"(?:I'?m|I am)\s+(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        
        # "My name is John Smith" / "My name's John Smith"
        r"(?:My name is|My name's)\s+(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        
        # "This is John Smith" / "This is Mr. John Smith"
        r"(?:This is)\s+(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        
        # "Call me John" / "You can call me John"
        r"(?:Call me|You can call me)\s+([A-Z][a-z]+)",
        
        # "I go by John" / "I go by Ms. Smith"
        r"(?:I go by)\s+(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        
        # "Patient: I'm John" (when there's a speaker label)
        r"(?:Patient|Pt):\s*(?:I'?m|I am)\s+(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        
        # "Hi, I'm John Smith" / "Hello, I'm Ms. Johnson"
        r"(?:Hi|Hello|Hey),?\s*(?:I'?m|I am)\s+(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
    ]
    
    # Try each pattern
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Take the first match and clean it up
            name = matches[0].strip()
            # Remove any remaining titles that might have been captured
            name = re.sub(r'^(Mr\.?|Mrs\.?|Ms\.?|Dr\.?)\s*', '', name, flags=re.IGNORECASE)
            if name and len(name) > 1:  # Ensure it's a reasonable name
                logger.info(f"Extracted patient name: '{name}' from text")
                return name
    
    logger.info("No patient name found in conversation text")
    return None


def clean_text(text: str) -> str:
    """Clean and normalize text input."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters that might interfere with processing
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    
    return text


def extract_key_phrases(text: str) -> List[str]:
    """Extract key phrases from text using simple pattern matching."""
    # This is a basic implementation - could be enhanced with NLP
    sentences = text.split('.')
    key_phrases = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 10:  # Filter out very short phrases
            key_phrases.append(sentence)
    
    return key_phrases


def format_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Format response data consistently."""
    return {
        "success": True,
        "data": data,
        "timestamp": data.get("timestamp", ""),
        "processing_time": data.get("processing_time", 0)
    }


def validate_text_input(text: str) -> bool:
    """Validate text input for processing."""
    if not text or not isinstance(text, str):
        return False
    
    if len(text.strip()) < 10:
        return False
    
    return True
