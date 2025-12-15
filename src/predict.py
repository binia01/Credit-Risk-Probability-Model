import pandas as pd
import numpy as np
import pickle
import logging
import os
import sys
import mlflow.sklearn

# Setup Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ScoringEngine")

class CreditScoringModel:
    def __init__(self, model_uri):
        """Initialize the scoring engine by loading the trained model."""
        self.model_uri = model_uri
        self.model = self._load_model()

        self.MIN_SCORE = 300
        self.MAX_SCORE = 850

    def _load_model(self):
        """Loads the mode from the MLflow"""
        try:
            model = mlflow.sklearn.load_model(self.model_uri)
            logger.info(f"Model loaded successfully from {self.model_uri}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def calculate_credit_score(self, probability_of_default):
        """
        Converts a probability of Default (0-1) into a credit Score (300 - 850).
        Logic: Higher Risk (PD=1) -> Lower Score (300)
        """

        # Linear Mapping
        # Score = Max_Score - (PD * Range)
        score_range = self.MAX_SCORE - self.MIN_SCORE
        score = self.MAX_SCORE - (probability_of_default * score_range)
        return int(score)
    
    def determine_risk_tier(self, credit_score):
        """Maps credit score to a categorical tier."""
        if credit_score >= 800: return "Excellent"
        if credit_score >= 740: return "Very Good"
        if credit_score >= 670: return "Good"
        if credit_score >= 580: return "Fair"
        return "Poor"
    
    def predict(self, input_data: dict):
        """
        Main inference method.
        Args:
            input_data (dict): Dictionary containing feature values.
                               Example: {'Recency': 10, 'Frequency': 5, ...}
        Returns:
            dict: The scoring result.
        """
        try:
            # 1. Convert Input to DataFrame
            # Ensure input is wrapped in a list for single-row DataFrame
            df = pd.DataFrame([input_data])
            
            # 2. Alignment Check
            # The model expects specific columns. In a real production system,
            # you would load the scaler/encoder here to transform raw inputs.
            # For this step, we assume input_data matches the training columns.
            expected_cols = self.model.feature_names_in_ if hasattr(self.model, 'feature_names_in_') else None
            
            if expected_cols is not None:
                # Add missing cols with 0 (e.g. missing One-Hot categories)
                for col in expected_cols:
                    if col not in df.columns:
                        df[col] = 0
                # Reorder columns to match training
                df = df[expected_cols]

            # 3. Make Prediction
            # predict_proba returns [[prob_class_0, prob_class_1]]
            # We want class 1 (High Risk)
            probability_default = self.model.predict_proba(df)[:, 1][0]
            
            # 4. Calculate Score
            credit_score = self.calculate_credit_score(probability_default)
            risk_tier = self.determine_risk_tier(credit_score)
            
            logger.info(f"Scored Customer: PD={probability_default:.4f}, Score={credit_score}")
            
            return {
                "probability_of_default": round(float(probability_default), 4),
                "credit_score": credit_score,
                "risk_tier": risk_tier,
                "model_version": "1.0.0"
            }

        except Exception as e:
            logger.error(f"Prediction Error: {e}")
            raise

# if __name__ == "__main__":
#     # Example input (Mocking a processed customer row)
#     # NOTE: In reality, these values should be Scaled/WoE transformed if your model trained on that.
#     sample_input = {
#         'Recency': 0.5,           # Scaled value
#         'Frequency': 0.1,         # Scaled value
#         'Monetary_Total': 0.2,    # Scaled value
#         'Monetary_Mean': 0.2,
#         'Transaction_Count': 5
#         # Add other features your model expects
#     }
    
#     try:
#         engine = CreditScoringModel(model_path="models/randomforest.pkl")
#         result = engine.predict(sample_input)
#         print("\n--- Prediction Result ---")
#         print(result)
#     except Exception as e:
#         print(f"Test Failed: {e}")