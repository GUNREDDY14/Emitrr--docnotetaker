# Physician Notetaker API

A comprehensive medical NLP system that processes transcribed physician-patient conversations and generates structured medical reports using advanced AI models.

## 🎯 Overview

The Physician Notetaker API is a FastAPI-based backend service that leverages medical-specific NLP models (BioBERT, spaCy) to extract medical entities, analyze sentiment, and generate structured medical documentation from transcribed conversations.

## ✨ Key Features

- **Medical Entity Recognition**: Extracts symptoms, treatments, diagnoses, and prognoses using BioBERT
- **Text Summarization**: Converts transcripts into structured medical reports
- **Sentiment Analysis**: Analyzes patient emotional state (Anxious/Neutral/Reassured)
- **SOAP Note Generation**: Creates structured medical notes
- **Transcription Processing**: Handles audio-to-text conversion workflows
- **Structured JSON Output**: Returns data in standardized medical formats

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Emitrr
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Run the application**
   ```bash
   uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

The API will be available at `http://localhost:8000`

## 📋 API Endpoints

### Core Medical NLP
- **POST** `/medical-nlp/summarize` - Process medical transcripts and extract structured data
- **POST** `/medical-nlp/extract-entities` - Extract detailed medical entities with confidence scores
- **GET** `/medical-nlp/health` - Health check for the medical NLP service

### Additional Services
- **POST** `/transcription/process` - Process audio transcription workflows
- **POST** `/summarization/summarize` - General text summarization
- **POST** `/sentiment/analyze` - Sentiment and intent analysis
- **POST** `/soap/generate` - Generate SOAP medical notes

### Documentation
- **GET** `/docs` - Interactive API documentation (Swagger UI)
- **GET** `/redoc` - Alternative API documentation

## 💡 Usage Examples

### Medical NLP Processing

```bash
curl -X POST "http://localhost:8000/medical-nlp/summarize" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Doctor: How are you feeling today? Patient: I had a car accident. My neck and back hurt a lot for four weeks. Doctor: Did you receive treatment? Patient: Yes, I had ten physiotherapy sessions, and now I only have occasional back pain.",
       "patient_name": "Janet Jones"
     }'
```

**Response:**
```json
{
  "Patient_Name": "Janet Jones",
  "Symptoms": ["Neck pain", "Back pain", "hurt", "pain"],
  "Diagnosis": "Whiplash injury",
  "Treatment": ["10 physiotherapy sessions", "physiotherapy", "treatment"],
  "Current_Status": "Occasional back pain",
  "Prognosis": "Full recovery expected within six months"
}
```

### Sentiment Analysis

```bash
curl -X POST "http://localhost:8000/sentiment/analyze" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "I am worried about this pain in my chest. Is this something serious?"
     }'
```

## 🏗️ Project Structure

```
Emitrr/
├── backend/
│   ├── app/                 # FastAPI application
│   │   ├── main.py         # Application entry point
│   │   ├── models.py       # Database models
│   │   ├── schemas.py      # Pydantic schemas
│   │   └── database.py     # Database configuration
│   ├── routers/            # API route handlers
│   │   ├── medical_nlp.py  # Medical NLP endpoints
│   │   ├── sentiment.py    # Sentiment analysis
│   │   ├── soap.py         # SOAP note generation
│   │   ├── summarization.py # Text summarization
│   │   └── transcription.py # Audio transcription
│   ├── services/           # Core business logic
│   │   ├── ner_extraction.py    # Named entity recognition
│   │   ├── nlp_pipeline.py      # NLP processing pipeline
│   │   ├── sentiment_intent.py  # Sentiment & intent analysis
│   │   ├── soap_generator.py    # SOAP note generation
│   │   └── summarizer.py        # Text summarization
│   ├── tests/              # Test suites
│   └── utils/              # Utility functions
├── requirements.txt        # Python dependencies
└── MEDICAL_NLP_DOCUMENTATION.md # Detailed technical docs
```

## 🔧 Technical Stack

- **Framework**: FastAPI
- **Database**: SQLAlchemy with PostgreSQL support
- **NLP Models**: 
  - BioBERT (emilyalsentzer/Bio_ClinicalBERT)
  - spaCy (en_core_web_sm)
  - Transformers (BART/T5 for summarization)
- **ML Libraries**: PyTorch, scikit-learn, pandas, numpy
- **Validation**: Pydantic
- **Testing**: pytest

## 🧪 Testing

Run the test suite to verify functionality:

```bash
# Activate virtual environment
source venv/bin/activate

# Run medical NLP tests
python backend/tests/test_medical_nlp.py

# Run all tests
pytest backend/tests/
```

## 🔒 Security & Privacy

- No patient data is stored permanently
- All processing is done in-memory
- Input validation and sanitization
- CORS middleware configured for cross-origin requests

## 📈 Future Enhancements

- Medical knowledge base integration (UMLS, SNOMED)
- Confidence scoring for extracted entities
- Multi-language support
- Real-time WebSocket processing
- Medical report templates in various formats

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For detailed technical documentation, see `MEDICAL_NLP_DOCUMENTATION.md`
