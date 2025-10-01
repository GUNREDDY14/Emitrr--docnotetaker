# backend/app/routers/transcription.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.schemas import TranscriptRequest, TranscriptResponse
from app.database import get_db
from services import nlp_pipeline
from app.models import Patient, Conversation, Report, SOAPNote, SentimentRecord
from services.ner_extraction import extract_medical_info
from utils.helpers import extract_patient_name
 # 🔥 import extractor

router = APIRouter(prefix="/transcription", tags=["transcription"])


@router.post("/process", response_model=TranscriptResponse)
def process_transcript(payload: TranscriptRequest, db: Session = Depends(get_db)):
    """
    Main endpoint: ingest transcript text and return structured outputs.
    Automatically extracts patient name from conversation if not provided.
    """
    # Extract patient name from conversation if not provided
    patient_name = payload.patient_name
    if not patient_name:
        patient_name = extract_patient_name(payload.text)
    
    # create/find patient (very simple)
    patient = None
    if patient_name:
        patient = db.query(Patient).filter(Patient.name == patient_name).first()
        if not patient:
            patient = Patient(name=patient_name)
            db.add(patient)
            db.commit()
            db.refresh(patient)

    # save conversation
    conv = Conversation(
        patient_id=(patient.id if patient else None),
        transcript=payload.text,
        metadata_json=payload.metadata or {},
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)

    # run NLP pipeline
    outputs = nlp_pipeline.process_transcript(payload.text, metadata=payload.metadata or {})

    # 🔥 run Bio_ClinicalBERT extractor
    medical_entities = extract_medical_info(payload.text)

    # save report, soap, sentiment
    report = Report(conversation_id=conv.id, summary=outputs["summary"])
    db.add(report)
    db.flush()
    soap = SOAPNote(conversation_id=conv.id, soap_json=outputs["soap"])
    db.add(soap)
    sentiment = SentimentRecord(
        conversation_id=conv.id,
        sentiment=outputs["sentiment"]["session"],
        intent=outputs["sentiment"]["intent"]
    )
    db.add(sentiment)
    db.commit()

    return TranscriptResponse(
        conversation_id=conv.id,
        transcript=conv.transcript,
        entities=medical_entities,   # 🔥 return ONLY clean JSON entities
        summary=outputs["summary"],
        soap_note=outputs["soap"],
    )
