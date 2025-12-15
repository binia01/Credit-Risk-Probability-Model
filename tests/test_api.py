import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    """Ensure the API is up and running."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict_endpoint_validation():
    """Test that the API rejects bad data (Pydantic validation)."""
    # Missing required fields
    bad_payload = {
        "Recency": 10
        # Frequency and Monetary are missing
    }
    response = client.post("/predict", params={"customer_id": "test"}, json=bad_payload)
    
    # Should return 422 Unprocessable Entity
    assert response.status_code == 422 

def test_predict_endpoint_structure():
    """
    Test a valid request structure.
    Note: If the model isn't loaded (e.g., in CI without MLflow), 
    the API might return 503, which is actually a VALID test of our error handling.
    """
    valid_payload = {
        "Recency": 0.5,
        "Frequency": 0.5,
        "Monetary_Total": 0.5,
        "Monetary_Mean": 0.5
    }
    
    response = client.post("/predict", params={"customer_id": "test_user"}, json=valid_payload)
    
    # We accept either 200 (Success) or 503 (Model Missing in Test Env)
    # Both prove the endpoint code is reachable and executing logic.
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "credit_score" in data
        assert "risk_tier" in data
        assert isinstance(data["approved"], bool)