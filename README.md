# Credit-Risk-Probability-Model

## Project Overview
This project focuses on the end-to-end development of a Credit Scoring Model for Bati Bank's new Buy-Now-Pay-Later (BNPL) partnership. Unlike traditional credit scoring which relies on credit bureau history, this project leverages **Alternative Data** (eCommerce transactions) to predict creditworthiness. By engineering behavioral features (RFM) and developing a proxy for default, we aim to calculate a Credit Risk Probability Score that informs loan approval decisions, optimizing the balance between risk (default) and reward (interest income).

## Project Structure
The project follows a modular, production-ready directory structure designed for MLOps maturity.

```text
credit-risk-model/
├── .github/workflows/ci.yml    # CI/CD pipeline for automated testing and deployment
├── data/
│   ├── raw/                    # Original Xente transaction dataset
│   └── processed/              # Data with RFM features and Weight of Evidence (WoE) transformation
├── notebooks/
│   └── eda.ipynb               # Exploratory Data Analysis & Feature Selection logic
├── src/
│   ├── __init__.py
│   ├── data_processing.py      # Pipelines for cleaning, RFM calculation, and WoE binning
│   ├── train.py                # Training script with MLFlow tracking and Hyperparameter tuning
│   ├── predict.py              # Inference engine for scoring new applicants
│   └── api/
│       ├── main.py             # FastAPI entry point
│       └── pydantic_models.py  # Request/Response schema validation
├── tests/
│   └── test_data_processing.py # Pytest modules for unit testing logic
├── Dockerfile                  # Container definition for reproducible environments
├── docker-compose.yml          # Orchestration for API and potential DB services
├── requirements.txt            # Python dependencies (scikit-learn, pandas, mlflow, fastapi)
├── .gitignore
└── README.md
```

---

## Credit Scoring Business Understanding

### 1. The Influence of Basel II Accord on Model Design
The Basel II Capital Accord is the global regulatory standard that dictates how much capital Bati Bank must hold to safeguard against insolvency. It moves beyond simple rules of thumb to a "Three Pillar" approach that directly impacts our data science workflow:

*   **Pillar 1 (Minimum Capital Requirements):**
    *   **Impact:** The model must output a calibrated **Probability of Default (PD)**, not just a binary classification. This PD is combined with *Loss Given Default (LGD)* and *Exposure at Default (EAD)* to calculate Expected Loss ($EL = PD \times LGD \times EAD$).
    *   **Constraint:** If our model is unstable or unverified, the regulator will force Bati Bank to use the "Standardized Approach," which usually requires holding significantly more capital than necessary, reducing the bank's profitability. To qualify for the "Internal Ratings-Based (IRB) Approach" (which saves the bank money), our model estimates must be precise and statistically valid.

*   **Pillar 2 (Supervisory Review):**
    *   **Impact:** "The Use Test." Regulators must see that the model is embedded in the bank's actual risk management process, not just a theoretical exercise.
    *   **Requirement:** This necessitates **Interpretability** and **Auditability**. We cannot deploy a "Black Box" where we cannot explain to an auditor *why* a specific applicant was rejected. Every feature transformation (like WoE) and model decision must be documented.

*   **Pillar 3 (Market Discipline):**
    *   **Impact:** The bank must disclose its risk assessment procedures.
    *   **Requirement:** Our code must be reproducible and our documentation (like this README) must clearly explain the methodology used to define "Risk," ensuring transparency for stakeholders.

### 2. The Proxy Variable: Necessity and Business Risks
In a standard credit modeling environment, we use "Supervised Learning" where the target variable $Y$ (Default) is known from historical loans (e.g., "Customer missed 3 payments in 12 months").
However, because this is a **Cold Start** problem with a new eCommerce partner, we have no loan history. We only have transaction history.

*   **Why a Proxy is Necessary:**
    We must infer creditworthiness from behavior. We assume a correlation between **financial stability** and **transactional consistency**. We will define a proxy target (e.g., `Risk_Label`) based on RFM (Recency, Frequency, Monetary) analysis.
    *   *Assumption:* A customer who buys frequently, recently, and with high value is likely "Good."
    *   *Assumption:* A customer with erratic, low-value, or ancient activity is likely "Bad."

*   **Business Risks of the Proxy Approach:**
    1.  **Behavior $\neq$ Repayment:** A customer might have high "Monetary" value (spending a lot) because they are impulsive and financially irresponsible. The proxy might label them "Low Risk" (Good) because they spend money, but they are actually the *highest* risk for default.
    2.  **The "Miser" Problem:** A financially prudent customer who saves money and buys rarely might be flagged as "High Risk" (Bad) by the proxy due to low Frequency/Monetary scores, causing the bank to lose a high-quality customer (Opportunity Cost).
    3.  **Fraud Vulnerability:** Fraudsters often mimic "Good" users (high frequency/value) to build a score before "busting out" (maxing credit and vanishing). Our proxy might unwittingly prioritize these fraudsters.
    4.  **Regulatory Bias:** If the proxy inadvertently correlates with protected characteristics (e.g., specific regions or demographics having lower transaction volumes), the model could violate Fair Lending laws.

### 3. Trade-offs: Logistic Regression (WoE) vs. Gradient Boosting
In the financial sector, the choice of algorithm is a strategic decision between **Explainability** (Regulatory Safety) and **Predictive Power** (Profitability).

#### **A. Logistic Regression with Weight of Evidence (WoE)**
*   **How it works:** Features are binned into groups (e.g., Age 20-30, 30-40). We calculate the WoE for each bin (measuring the separation of Good/Bad). These values are fed into a logistic regression.
*   **Pros:**
    *   **Industry Standard:** This is the gold standard for Basel II compliance. It produces a **Scorecard** (e.g., "If Age is 20-30, add 15 points").
    *   **Monotonicity:** We can force the model to follow logical rules (e.g., "Risk *must* decrease as Income increases").
    *   **Explainability:** We can tell a customer exactly why they were rejected ("Your transaction frequency score was too low").
*   **Cons:**
    *   **Linearity:** It assumes a linear relationship between features and risk. It fails to capture complex, non-linear patterns in behavioral data (e.g., high spending is good *only* if frequency is also moderate).
    *   **Performance:** Generally has lower accuracy (AUC) than tree-based models.

#### **B. Gradient Boosting (XGBoost / LightGBM / CatBoost)**
*   **How it works:** An ensemble of decision trees that iteratively correct the errors of previous trees.
*   **Pros:**
    *   **Predictive Power:** Excellent at capturing complex, non-linear relationships and edge cases in "Alternative Data" which LogReg might miss.
    *   **Feature Importance:** Automatically handles feature interactions.
*   **Cons:**
    *   **The "Black Box" Issue:** It is difficult to explain a specific prediction. "Tree #450 made a split based on Transaction Time" is not a valid explanation for a loan officer or regulator.
    *   **Overfitting:** Can memorize noise in the proxy variable, leading to poor generalization on real loans.
    *   **Stability:** Small changes in input data can lead to large jumps in the score, which is undesirable in banking.

#### **Conclusion for Bati Bank:**
Given we are launching a new product, we will likely adopt a **Hybrid Approach**:
1.  Use **Gradient Boosting** for feature selection and to set a "performance ceiling" (benchmark).
2.  Use **SHAP (SHapley Additive exPlanations)** values to attempt to explain the Boosting model.
3.  If regulatory pressure is high, distill the insights into a **Logistic Regression** scorecard for the final production deployment to ensure full Basel II compliance.