# backend/app/services/soap_generator.py
"""
Enhanced SOAP note generator that integrates comprehensive NLP pipeline outputs.
Takes inputs from summarizer, sentiment analysis, NER extraction, and original text.
"""
from typing import Dict, Any, List, Optional
import re


def generate_soap(
    text: str, 
    entities: Dict[str, Any] = None, 
    summary: Dict[str, Any] = None,
    sentiment_analysis: Dict[str, Any] = None,
    keywords: List[str] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive SOAP note using all NLP pipeline components.
    
    Args:
        text: Original transcript text
        entities: NER extraction results from ner_extraction.py
        summary: Medical summary from summarizer.py
        sentiment_analysis: Sentiment and intent analysis from sentiment_intent.py
        keywords: Extracted keywords from nlp_pipeline.py
    
    Returns:
        Structured SOAP note with enhanced medical information
    """
    if entities is None:
        entities = {}
    if summary is None:
        summary = {}
    if sentiment_analysis is None:
        sentiment_analysis = {}
    if keywords is None:
        keywords = []

    # Extract patient information
    patient_name = summary.get("Patient_Name") or entities.get("Patient_Name", "Not recorded")
    symptoms = summary.get("Symptoms", []) or entities.get("Symptoms", [])
    diagnosis = summary.get("Diagnosis") or entities.get("Diagnosis", "")
    treatments = summary.get("Treatment", []) or entities.get("Treatment", [])
    current_status = summary.get("Current_Status", "") or entities.get("Current_Status", "")
    prognosis = summary.get("Prognosis", "") or entities.get("Prognosis", "")
    
    # Extract sentiment and intent
    sentiment = sentiment_analysis.get("Sentiment", "Neutral")
    intent = sentiment_analysis.get("Intent", "Reporting symptoms")
    confidence = sentiment_analysis.get("confidence", 0.5)
    
    # Build comprehensive Subjective section
    chief_complaint = summary.get("Chief_Complaint", "")
    if not chief_complaint and symptoms:
        chief_complaint = f"Patient reports {', '.join(symptoms[:3])}"  # Top 3 symptoms
    if not chief_complaint:
        chief_complaint = "Patient consultation - detailed history to be obtained"
    
    # Enhanced History of Present Illness
    hpi_components = []
    if chief_complaint:
        hpi_components.append(f"Chief complaint: {chief_complaint}")
    
    if symptoms:
        symptoms_text = ', '.join(symptoms)
        hpi_components.append(f"Symptoms reported: {symptoms_text}")
    
    if current_status:
        hpi_components.append(f"Current status: {current_status}")
    
    # Add sentiment-based context
    if sentiment == "Anxious":
        hpi_components.append("Patient appears anxious about condition")
    elif sentiment == "Reassured":
        hpi_components.append("Patient appears reassured by discussion")
    
    if intent == "Seeking reassurance":
        hpi_components.append("Patient seeking reassurance about condition")
    elif intent == "Expressing concern":
        hpi_components.append("Patient expressing concerns about symptoms")
    
    history_of_present_illness = ". ".join(hpi_components) if hpi_components else "History to be obtained"

    # Build comprehensive Objective section
    physical_exam_findings = []
    observations = []
    
    # Extract physical findings from text and entities
    text_lower = text.lower()
    
    # Check for specific physical exam mentions
    if "pain" in text_lower or any("pain" in symptom.lower() for symptom in symptoms):
        physical_exam_findings.append("Patient reports pain in affected areas")
    
    if "stiffness" in text_lower or "stiff" in text_lower:
        physical_exam_findings.append("Stiffness noted in affected areas")
    
    if "swelling" in text_lower or "swollen" in text_lower:
        physical_exam_findings.append("Swelling may be present")
    
    if "bruising" in text_lower or "bruise" in text_lower:
        physical_exam_findings.append("Bruising may be present")
    
    # Default physical exam if no specific findings
    if not physical_exam_findings:
        physical_exam_findings.append("Physical examination findings to be documented")
        if "whiplash" in text_lower or "car accident" in text_lower:
            physical_exam_findings.append("Cervical and lumbar spine range of motion to be assessed")
    
    # Patient observations based on sentiment and symptoms
    if sentiment == "Anxious":
        observations.append("Patient appears anxious during consultation")
    elif sentiment == "Reassured":
        observations.append("Patient appears reassured and cooperative")
    
    if current_status and "improving" in current_status.lower():
        observations.append("Patient reports improvement in condition")
    elif current_status and "occasional" in current_status.lower():
        observations.append("Patient reports occasional symptoms")
    
    if not observations:
        observations.append("Patient cooperative during consultation")
    
    # Add treatment-related observations
    if treatments:
        treatment_observations = f"Patient has undergone: {', '.join(treatments[:3])}"
        observations.append(treatment_observations)

    # Build comprehensive Assessment section
    assessment_diagnosis = diagnosis if diagnosis else "Diagnosis pending further evaluation"
    
    # Determine severity based on symptoms and status
    severity_indicators = []
    if current_status and "occasional" in current_status.lower():
        severity_indicators.append("mild")
    if "improving" in text_lower or (current_status and "improving" in current_status.lower()):
        severity_indicators.append("improving")
    if "severe" in text_lower or "severe" in ' '.join(symptoms).lower():
        severity_indicators.append("severe")
    if "chronic" in text_lower or "chronic" in ' '.join(symptoms).lower():
        severity_indicators.append("chronic")
    
    severity = ", ".join(severity_indicators) if severity_indicators else "moderate"
    
    # Add prognosis to assessment
    assessment_notes = []
    if prognosis:
        assessment_notes.append(f"Prognosis: {prognosis}")
    
    # Add confidence-based assessment notes
    if confidence > 0.8:
        assessment_notes.append("High confidence in assessment")
    elif confidence < 0.5:
        assessment_notes.append("Assessment based on limited information")

    # Build comprehensive Plan section
    plan_treatments = []
    follow_up_plan = []
    
    # Extract treatment recommendations
    if treatments:
        plan_treatments.extend(treatments)
    
    # Add treatment recommendations based on diagnosis
    if "whiplash" in assessment_diagnosis.lower():
        if not any("physiotherapy" in t.lower() or "physio" in t.lower() for t in plan_treatments):
            plan_treatments.append("Physiotherapy sessions recommended")
        if not any("pain" in t.lower() for t in plan_treatments):
            plan_treatments.append("Pain management as needed")
    
    if not plan_treatments:
        plan_treatments.append("Treatment plan to be determined based on assessment")
    
    # Follow-up recommendations
    if prognosis and "months" in prognosis:
        follow_up_plan.append("Follow-up in 4-6 weeks to assess progress")
    elif "improving" in severity.lower():
        follow_up_plan.append("Follow-up in 2-4 weeks if symptoms persist")
    else:
        follow_up_plan.append("Return if symptoms worsen or persist")
    
    # Add specific follow-up based on sentiment
    if sentiment == "Anxious":
        follow_up_plan.append("Patient education and reassurance provided")
    elif intent == "Seeking reassurance":
        follow_up_plan.append("Address patient concerns and provide reassurance")

    # Construct final SOAP note
    soap_note = {
        "Subjective": {
            "Patient_Name": patient_name,
            "Chief_Complaint": chief_complaint,
            "History_of_Present_Illness": history_of_present_illness,
            "Patient_Sentiment": sentiment,
            "Patient_Intent": intent
        },
        "Objective": {
            "Physical_Exam": ". ".join(physical_exam_findings),
            "Observations": ". ".join(observations),
            "Vital_Signs": "To be documented during physical examination",
            "Assessment_Confidence": f"{confidence:.2f}"
        },
        "Assessment": {
            "Diagnosis": assessment_diagnosis,
            "Severity": severity,
            "Prognosis": prognosis if prognosis else "Prognosis to be determined",
            "Clinical_Notes": ". ".join(assessment_notes) if assessment_notes else "Standard assessment completed"
        },
        "Plan": {
            "Treatment": plan_treatments,
            "Follow_Up": follow_up_plan,
            "Patient_Education": "Patient education provided as appropriate",
            "Next_Steps": "Continue current treatment plan and monitor progress"
        }
    }
    
    # Add metadata section with NLP pipeline information
    soap_note["Metadata"] = {
        "Text_Length": len(text),
        "Keywords_Extracted": len(keywords),
        "Symptoms_Identified": len(symptoms),
        "Treatments_Identified": len(treatments),
        "Analysis_Method": sentiment_analysis.get("analysis_method", "rule_based"),
        "Processing_Timestamp": "Generated from NLP pipeline"
    }
    
    return soap_note


def generate_soap_from_pipeline(pipeline_output: Dict[str, Any], original_text: str) -> Dict[str, Any]:
    """
    Convenience function to generate SOAP note directly from NLP pipeline output.
    
    Args:
        pipeline_output: Complete output from nlp_pipeline.process_transcript()
        original_text: Original transcript text
    
    Returns:
        Enhanced SOAP note
    """
    return generate_soap(
        text=original_text,
        entities=pipeline_output.get("entities", {}),
        summary=pipeline_output.get("summary", {}),
        sentiment_analysis=pipeline_output.get("sentiment", {}),
        keywords=pipeline_output.get("keywords", [])
    )
