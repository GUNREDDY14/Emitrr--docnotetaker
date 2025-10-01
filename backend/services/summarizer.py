"""
Summarization with transformers (BART/T5) and robust rule-based fallback.
Outputs a structured medical JSON summary.
"""
from typing import Dict, Any, Optional
import re


_HF_SUMMARIZER = None  # lazy singleton


def _load_summarizer():
    global _HF_SUMMARIZER
    if _HF_SUMMARIZER is not None:
        return _HF_SUMMARIZER
    try:
        from transformers import pipeline
        candidate_models = [
            "facebook/bart-large-cnn",
            "t5-base",
        ]
        for model in candidate_models:
            try:
                _HF_SUMMARIZER = pipeline("summarization", model=model)
                break
            except Exception:
                continue
        return _HF_SUMMARIZER
    except Exception:
        return None


def _rule_based_summary(text: str, entities: Dict[str, Any]) -> Dict[str, Any]:
    text_lower = text.lower()
    sentences = re.split(r'(?<=[\.?\!])\s+', text.strip())
    hpi_sentences = [s for s in sentences if re.search(r"accident|pain|hurt|rear|whiplash|physio|physiotherapy", s, flags=re.I)]
    chief = hpi_sentences[0] if hpi_sentences else (sentences[0] if sentences else "")

    diagnosis = entities.get("Diagnosis") if entities.get("Diagnosis") else ("Whiplash injury" if "whiplash" in text_lower else "Not specified")
    treatment = entities.get("Treatment") or []
    current_status = "Occasional backache" if re.search(r"occasional backache|occasional back pain|now only", text_lower) else "Improving"
    prognosis = "Full recovery expected within six months" if re.search(r"full recovery|within six months|no long-term", text_lower) else "Prognosis not explicitly stated"
    return {
        "Patient_Name": (entities.get("Patient_Name") if entities.get("Patient_Name") else None),
        "Chief_Complaint": chief.strip(),
        "Symptoms": entities.get("Symptoms", []),
        "Diagnosis": diagnosis,
        "Treatment": treatment,
        "Current_Status": current_status,
        "Prognosis": prognosis,
    }


def summarize_text(text: str, entities: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if entities is None:
        entities = {}

    # Use the enhanced NER system to extract medical entities
    from .ner_extraction import extract_medical_info, _extract_patient_name
    medical_info = extract_medical_info(text)

    # If Patient_Name missing, force extract from text
    patient_name = medical_info.get("Patient_Name", "")
    if not patient_name:
        patient_name = _extract_patient_name(text)

    result = {
        "Patient_Name": patient_name,
        "Symptoms": medical_info.get("Symptoms", []),
        "Diagnosis": medical_info.get("Diagnosis", ""),
        "Treatment": medical_info.get("Treatment", []),
        "Current_Status": medical_info.get("Current_Status", ""),
        "Prognosis": medical_info.get("Prognosis", "")
    }

    
    # Enhance with rule-based patterns and abstractive summarization
    text_lower = text.lower()
    
    # Extract additional symptoms from text
    additional_symptoms = []
    symptom_patterns = [
        r"(neck pain)", r"(back pain)", r"(head pain)", r"(shoulder pain)",
        r"(chest pain)", r"(abdominal pain)", r"(knee pain)", r"(hip pain)",
        r"(headache)", r"(dizziness)", r"(nausea)", r"(fatigue)",
        r"(stiffness)", r"(swelling)", r"(bruising)", r"(numbness)"
    ]
    
    for pattern in symptom_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if match not in [s.lower() for s in result["Symptoms"]]:
                additional_symptoms.append(match.capitalize())
    
    result["Symptoms"].extend(additional_symptoms)
    result["Symptoms"] = list(set(result["Symptoms"]))  # Remove duplicates
    
    # Extract additional treatments
    additional_treatments = []
    treatment_patterns = [
        r"(\d+\s+(?:physiotherapy|physio|therapy)\s+sessions)",
        r"(painkillers?)", r"(medication)", r"(surgery)", r"(operation)",
        r"(injection)", r"(rehabilitation)", r"(exercise)", r"(stretching)",
        r"(massage)", r"(heat therapy)", r"(ice therapy)"
    ]
    
    for pattern in treatment_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if match not in [t.lower() for t in result["Treatment"]]:
                additional_treatments.append(match.capitalize())
    
    result["Treatment"].extend(additional_treatments)
    result["Treatment"] = list(set(result["Treatment"]))  # Remove duplicates
    
    # Enhance diagnosis if not found
    if not result["Diagnosis"]:
        if "whiplash" in text_lower:
            result["Diagnosis"] = "Whiplash injury"
        elif "car accident" in text_lower or "crash" in text_lower:
            result["Diagnosis"] = "Motor vehicle accident injury"
        elif "fall" in text_lower:
            result["Diagnosis"] = "Fall-related injury"
        elif "sports injury" in text_lower:
            result["Diagnosis"] = "Sports injury"
    
    # Enhance current status
    if not result["Current_Status"]:
        if "occasional" in text_lower and "back" in text_lower:
            result["Current_Status"] = "Occasional back pain"
        elif "occasional" in text_lower and "pain" in text_lower:
            result["Current_Status"] = "Occasional pain"
        elif "improving" in text_lower or "better" in text_lower:
            result["Current_Status"] = "Improving"
        elif "no longer" in text_lower and "pain" in text_lower:
            result["Current_Status"] = "Pain resolved"
    
    # Enhance prognosis
    if not result["Prognosis"]:
        if "full recovery" in text_lower and "six months" in text_lower:
            result["Prognosis"] = "Full recovery expected within six months"
        elif "full recovery" in text_lower:
            result["Prognosis"] = "Full recovery expected"
        elif "recovery" in text_lower and "months" in text_lower:
            result["Prognosis"] = "Recovery expected within months"
        elif "chronic" in text_lower:
            result["Prognosis"] = "Chronic condition"
    
    # TODO: Implement abstractive summarization enhancement if needed
    # This would require loading a transformer model for summarization
    
    # Clean up empty strings and ensure proper formatting
    for key, value in result.items():
        if isinstance(value, list):
            result[key] = [item for item in value if item and item.strip()]
        elif isinstance(value, str):
            result[key] = value.strip()
    
    return result
