import logging
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import CreditScoringRequest, CreditScoringResponse
from src.predict import CreditScoringModel

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CreditRiskAPI")

app = FastAPI(
    title="Bati Bank Credit Scoring API",
    description="Microservice for predicting credit risk using Alternative Data.",
    version="1.0.0"
)

model_engine = None

@app.on_event("startup")
def load_model():
    global model_engine
    try:
        model_path = "models/logisticregression.pkl"
        model_engine = CreditScoringModel(model_path=model_path)
        logger.info(f"Model loaded successfully from {model_path}")

    except Exception as e:
        logger.error(f"Failed to load model at {model_path}")
        raise

@app.get("/health")
def health_check():
    status = "healthy" if model_engine else "degraded (model missing.)"
    return {"status": status, "service": "credit-scoring"}

@app.post("/predict", response_model=CreditScoringResponse)
def predict_risk(customer_id: str, request: CreditScoringRequest):
    """
    Main prediction endpoint.
    """
    if not model_engine:
        raise HTTPException(status_code=503, detail="Scoring model is not initialized.")

    try:
        # Convert Pydantic model to Dict
        input_data = request.dict()
        
        # Run Inference
        result = model_engine.predict(input_data)
        
        # Define Approval Logic (Business Rule)
        # Example: Approve if Credit Score > 650 (Good/Excellent)
        is_approved = result['credit_score'] >= 650

        return CreditScoringResponse(
            customer_id=customer_id,
            probability_of_default=result['probability_of_default'],
            credit_score=result['credit_score'],
            risk_tier=result['risk_tier'],
            approved=is_approved
        )

    except Exception as e:
        logger.error(f"Prediction failed for {customer_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")