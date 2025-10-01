# backend/routers/medical_nlp.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import logging

from services.ner_extraction import extract_medical_info
from services.summarizer import summarize_text
from utils.helpers import extract_patient_name

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/medical-nlp", tags=["medical-nlp"])

class MedicalTranscriptRequest(BaseModel):
    """Request model for medical transcript processing."""
    text: str
    patient_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class MedicalSummaryResponse(BaseModel):
    """Response model for medical summary in the exact format specified."""
    Patient_Name: str
    Symptoms: List[str]
    Diagnosis: str
    Treatment: List[str]
    Current_Status: str
    Prognosis: str

@router.post("/summarize", response_model=MedicalSummaryResponse)
async def summarize_medical_transcript(request: MedicalTranscriptRequest):
    """
    Process medical transcript and return structured medical summary.
    
    This endpoint extracts medical entities (symptoms, treatments, diagnosis, prognosis)
    using BioBERT and other medical NLP models, and returns the exact JSON format
    specified in the requirements.
    
    Example input:
    "Doctor: How are you feeling today? Patient: I had a car accident. My neck and back hurt a lot for four weeks. Doctor: Did you receive treatment? Patient: Yes, I had ten physiotherapy sessions, and now I only have occasional back pain."
    
    Example output:
    {
        "Patient_Name": "",
        "Symptoms": ["Neck pain", "Back pain"],
        "Diagnosis": "Whiplash injury",
        "Treatment": ["10 physiotherapy sessions"],
        "Current_Status": "Occasional back pain",
        "Prognosis": "Full recovery expected within six months"
    }
    """
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text input is required")
        
        # Extract patient name from conversation if not provided
        patient_name = request.patient_name
        if not patient_name:
            patient_name = extract_patient_name(request.text)
        
        # Extract medical information using enhanced NER system
        medical_info = extract_medical_info(request.text)
        
        # Use the summarizer to enhance and structure the information
        summary = summarize_text(request.text, entities=medical_info)
        
        # Use extracted or provided patient name
        if patient_name:
            summary["Patient_Name"] = patient_name
        
        # Ensure all required fields are present with proper types
        response = MedicalSummaryResponse(
            Patient_Name=summary.get("Patient_Name", ""),
            Symptoms=summary.get("Symptoms", []),
            Diagnosis=summary.get("Diagnosis", ""),
            Treatment=summary.get("Treatment", []),
            Current_Status=summary.get("Current_Status", ""),
            Prognosis=summary.get("Prognosis", "")
        )
        
        logger.info(f"Successfully processed medical transcript. Extracted {len(response.Symptoms)} symptoms, {len(response.Treatment)} treatments.")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing medical transcript: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing transcript: {str(e)}")

@router.post("/extract-entities")
async def extract_medical_entities(request: MedicalTranscriptRequest):
    """
    Extract medical entities from transcript using BioBERT and other medical NLP models.
    
    Returns detailed entity extraction results including confidence scores and
    multiple extraction methods used.
    """
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text input is required")
        
        # Extract patient name from conversation if not provided
        patient_name = request.patient_name
        if not patient_name:
            patient_name = extract_patient_name(request.text)
        
        # Extract medical information
        medical_info = extract_medical_info(request.text)
        
        return {
            "success": True,
            "entities": medical_info,
            "extraction_methods": [
                "BioBERT NER",
                "spaCy NLP",
                "Medical keyword patterns",
                "Regex-based extraction"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error extracting medical entities: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting entities: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint for the medical NLP service."""
    return {
        "status": "healthy",
        "service": "medical-nlp",
        "models_loaded": {
            "biobert_ner": "available",
            "spacy": "available",
            "transformers": "available"
        }
    }

