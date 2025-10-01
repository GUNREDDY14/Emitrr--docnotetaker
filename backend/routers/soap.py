# backend/app/routers/soap.py
from fastapi import APIRouter
from app.schemas import SOAPRequest, SOAPResponse
from services import soap_generator

router = APIRouter(prefix="/soap", tags=["soap"])


@router.post("/generate", response_model=SOAPResponse)
def generate(req: SOAPRequest):
    soap = soap_generator.generate_soap(req.text)
    return {"soap": soap}
