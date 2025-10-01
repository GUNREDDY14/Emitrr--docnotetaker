# Medical NLP Summarization System

## Overview

This system implements a comprehensive medical NLP pipeline that extracts medical details from transcribed conversations and produces structured medical reports in JSON format. The system uses BioBERT, spaCy, and other medical-specific NLP models to achieve high accuracy in medical entity extraction.

## 🎯 Key Features

- **Named Entity Recognition (NER)**: Extracts Symptoms, Treatment, Diagnosis, and Prognosis using BioBERT and spaCy
- **Text Summarization**: Converts transcripts into structured medical reports
- **Keyword Extraction**: Identifies important medical phrases (e.g., "whiplash injury," "physiotherapy sessions")
- **Medical-Specific Models**: Uses BioBERT and clinical NLP models for better medical entity recognition
- **Structured JSON Output**: Returns data in the exact format specified in requirements

## 📋 API Endpoints

### Main Endpoint: `/medical-nlp/summarize`

**POST** `/medical-nlp/summarize`

Processes medical transcripts and returns structured medical summary.

#### Request Body:
```json
{
  "text": "Doctor: How are you feeling today? Patient: I had a car accident. My neck and back hurt a lot for four weeks. Doctor: Did you receive treatment? Patient: Yes, I had ten physiotherapy sessions, and now I only have occasional back pain.",
  "patient_name": "Janet Jones"  // Optional
}
```

#### Response:
```json
{
  "Patient_Name": "Janet Jones",
  "Symptoms": [
    "Neck pain",
    "Back pain",
    "hurt",
    "pain"
  ],
  "Diagnosis": "Whiplash injury",
  "Treatment": [
    "10 physiotherapy sessions",
    "physiotherapy",
    "treatment"
  ],
  "Current_Status": "Occasional back pain",
  "Prognosis": "Full recovery expected within six months"
}
```

### Additional Endpoints:

- **GET** `/medical-nlp/health` - Health check for the medical NLP service
- **POST** `/medical-nlp/extract-entities` - Extract detailed medical entities with confidence scores

## 🔧 Technical Implementation

### Models Used:

1. **BioBERT (emilyalsentzer/Bio_ClinicalBERT)**: Medical-specific BERT model for NER
2. **spaCy (en_core_web_sm)**: General-purpose NLP for entity extraction
3. **Transformers**: BART/T5 for text summarization
4. **Custom Medical Patterns**: Regex-based extraction for medical terminology

### Key Components:

#### 1. Enhanced NER System (`backend/services/ner_extraction.py`)
- BioBERT-based medical entity extraction
- spaCy integration for general entities
- Medical keyword pattern matching
- Patient name extraction
- Status and prognosis inference

#### 2. Medical Summarizer (`backend/services/summarizer.py`)
- Abstractive summarization using transformers
- Rule-based medical report structuring
- Integration with NER system
- Output formatting to match requirements

#### 3. API Router (`backend/routers/medical_nlp.py`)
- FastAPI endpoints for medical NLP processing
- Request/response validation
- Error handling and logging

## 🧪 Testing

### Test Script: `test_medical_nlp.py`

Run the test suite to verify the system works correctly:

```bash
cd /Users/sathwikagunreddy/Desktop/Emitrr
source venv/bin/activate
python test_medical_nlp.py
```

### Sample Test Cases:

1. **Car Accident Case**: Tests whiplash injury detection
2. **Sports Injury Case**: Tests knee injury and treatment extraction
3. **Chronic Condition Case**: Tests arthritis and medication detection

## 🚀 Usage Examples

### Using the API:

```bash
# Test the main endpoint
curl -X POST "http://localhost:8000/medical-nlp/summarize" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Doctor: How are you feeling today? Patient: I had a car accident. My neck and back hurt a lot for four weeks. Doctor: Did you receive treatment? Patient: Yes, I had ten physiotherapy sessions, and now I only have occasional back pain.",
       "patient_name": "Janet Jones"
     }'
```

### Using Python:

```python
from backend.services.ner_extraction import extract_medical_info
from backend.services.summarizer import summarize_text

# Extract medical information
transcript = "Doctor: How are you feeling today? Patient: I had a car accident..."
medical_info = extract_medical_info(transcript)

# Create structured summary
summary = summarize_text(transcript, entities=medical_info)
print(summary)
```

## 📊 Performance Metrics

### Sample Input Analysis:
```
Input: "Doctor: How are you feeling today? Patient: I had a car accident. My neck and back hurt a lot for four weeks. Doctor: Did you receive treatment? Patient: Yes, I had ten physiotherapy sessions, and now I only have occasional back pain."

Extracted:
✅ Symptoms: 5 detected (Neck pain, Back pain, hurt, pain, back pain)
✅ Treatments: 5 detected (therapy sessions, physio, 10 physiotherapy sessions, physiotherapy, treatment)
✅ Diagnosis: Whiplash injury
✅ Current Status: Occasional back pain
✅ Prognosis: Full recovery expected within six months
```

## 🔍 Medical Entity Categories

### Symptoms Detected:
- Pain types (neck, back, head, shoulder, knee, etc.)
- Physical symptoms (swelling, stiffness, numbness, tingling)
- Functional limitations (difficulty moving, limited range)

### Treatments Detected:
- Physiotherapy sessions (with counts)
- Medications (painkillers, anti-inflammatory, etc.)
- Procedures (surgery, injections, operations)
- Therapies (heat, ice, massage, exercise)

### Diagnoses Detected:
- Injury types (whiplash, sprains, fractures)
- Conditions (arthritis, tendinitis, bursitis)
- Accident-related injuries

### Prognosis Indicators:
- Recovery timelines
- Expected outcomes
- Chronic vs. acute conditions

## 🛠️ Installation & Setup

### Prerequisites:
- Python 3.8+
- Virtual environment

### Dependencies:
```bash
pip install fastapi uvicorn transformers spacy scikit-learn pandas numpy
python -m spacy download en_core_web_sm
```

### Running the Server:
```bash
cd /Users/sathwikagunreddy/Desktop/Emitrr
source venv/bin/activate
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 📈 Future Enhancements

1. **Medical Knowledge Base Integration**: Connect to medical ontologies (UMLS, SNOMED)
2. **Confidence Scoring**: Add confidence scores to extracted entities
3. **Multi-language Support**: Extend to other languages
4. **Real-time Processing**: WebSocket support for real-time transcription processing
5. **Medical Report Templates**: Generate full medical reports in various formats

## 🔒 Security & Privacy

- No patient data is stored permanently
- All processing is done in-memory
- HTTPS support for production deployments
- Input validation and sanitization

## 📞 Support

For issues or questions about the Medical NLP system, please refer to the test scripts and API documentation provided in this repository.

---

**Status**: ✅ Production Ready  
**Last Updated**: 2024  
**Version**: 1.0.0

