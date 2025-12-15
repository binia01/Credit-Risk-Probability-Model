from pydantic import BaseModel, Field

class CreditScoringRequest(BaseModel):
    """Defines the input features required by the model."""

    Recency: float = Field(..., description="Days since last transaction (or scaled value)")
    Frequency: float = Field(..., description="Number of transactions (or scaled value)")
    Monetary_Total: float = Field(..., description="Total spend amount (or scaled value)")
    Monetary_Mean: float = Field(..., description="Average transaction value")
    
    # Add other features if your model strictly requires them.
    # For a robust API, we often allow extra fields to pass through:
    class Config:
        extra = "allow" 

class CreditScoringResponse(BaseModel):
    customer_id: str
    probability_of_default: float
    credit_score: int
    risk_tier: str
    approved: bool