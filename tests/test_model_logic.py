import pytest
import sys
import os

# Ensure src is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import CreditScoringModel

class MockModel:
    """Mock MLflow model to bypass loading a real file."""
    def predict_proba(self, data):
        return [[0.2, 0.8]] # Fake return: 20% Good, 80% Bad

def test_credit_score_calculation():
    """Test the math converting Probability of Default (PD) to Score."""
    # We pass None for uri because we won't load the real model for this unit test
    # We bypass the __init__ loading logic by mocking or subclassing, 
    # but for simplicity, let's just test the standalone methods if possible.
    
    # Instantiate without loading model (Hack for testing methods only)
    engine = CreditScoringModel.__new__(CreditScoringModel)
    engine.MIN_SCORE = 300
    engine.MAX_SCORE = 850
    
    # Case 1: 0% Risk (Perfect) -> Should be 850
    assert engine.calculate_credit_score(0.0) == 850
    
    # Case 2: 100% Risk (Fail) -> Should be 300
    assert engine.calculate_credit_score(1.0) == 300
    
    # Case 3: 50% Risk -> Should be 575 (Midpoint)
    # Range is 550. Half is 275. 850 - 275 = 575.
    assert engine.calculate_credit_score(0.5) == 575

def test_risk_tier_mapping():
    """Test that scores map to the correct strings."""
    engine = CreditScoringModel.__new__(CreditScoringModel)
    
    assert engine.determine_risk_tier(820) == "Excellent"
    assert engine.determine_risk_tier(750) == "Very Good"
    assert engine.determine_risk_tier(680) == "Good"
    assert engine.determine_risk_tier(600) == "Fair"
    assert engine.determine_risk_tier(400) == "Poor"