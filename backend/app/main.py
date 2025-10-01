# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import transcription, summarization, sentiment, soap, medical_nlp
from app.database import engine
from app.models import Base
from utils.logger import setup_logging

setup_logging()

# create DB tables if running against a DB accessible at startup
try:
    Base.metadata.create_all(bind=engine)
except Exception:
    # in local demos we might not have DB up; ignore if so
    pass

app = FastAPI(title="Physician Notetaker API", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down per environment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(transcription.router)
app.include_router(summarization.router)
app.include_router(sentiment.router)
app.include_router(soap.router)
app.include_router(medical_nlp.router)


@app.get("/")
def root():
    return {"message": "Physician Notetaker API — backend up"}
