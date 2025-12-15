import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import mlflow
import mlflow.sklearn

# Setup Directories
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

def load_data(file_path):
    """
    Loads the processed model-ready data.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded. Shape: {df.shape}")
        return df
    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    
def eval_metrics(actual, pred, pred_proba):
    """Calculates comprehensive classification metrics."""
    return{
        "accuracy": accuracy_score(actual, pred),
        "precision": precision_score(actual, pred),
        "recall": recall_score(actual, pred),
        "f1": f1_score(actual, pred),
        "roc_auc": roc_auc_score(actual, pred_proba)
    }

def plot_confusion_matrix(y_test, y_pred, model_name):
    """Plots and saves confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    path = f"plots/cm-{model_name}.png"
    plt.savefig(path)
    plt.close
    return path

def train_model(model_name, model_obj, param_grid, X_train, X_test, y_train, y_test):
    """Generic funciton to train, tune and log any model."""
    with mlflow.start_run(run_name=f"Train_{model_name}"):
        print(f"Training {model_name}...")

        # Hyperparameter Tuning (Grid Search)
        print(f"Running Grid Search for {model_name}...")
        grid = GridSearchCV(model_obj, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        print(f"Best Params: {grid.best_params_}")
        
        # 2. Predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # 3. Metrics
        metrics = eval_metrics(y_test, y_pred, y_pred_proba)
        print(f"Performance: AUC={metrics['roc_auc']:.4f}, Accuracy={metrics['accuracy']:.4f}")
        
        # 4. Logging to MLflow
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)
        
        # Log Artifacts (Confusion Matrix)
        cm_path = plot_confusion_matrix(y_test, y_pred, model_name)
        mlflow.log_artifact(cm_path)
        
        # 5. Log Model and Register
        mlflow.sklearn.log_model(best_model, artifact_path="model", registered_model_name=f"CreditRisk_{model_name}")
        
        # Save locally
        pickle.dump(best_model, open(f"models/{model_name.lower()}.pkl", "wb"))
        
        return metrics['roc_auc']

def main():
    # 1. Load Data
    data_path = "./data/processed/model_ready_data.csv"
    df = load_data(data_path)
    if df is None: return

    # 2. Split Features and Target
    # Exclude ID columns and Target
    drop_cols = ['AccountId', 'RiskLabel', 'BatchId', 'SubscriptionId', 'CustomerId']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['RiskLabel']
    
    print(f"Training with {X.shape[1]} features.")
    
    # 3. Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 4. Setup MLflow
    mlflow.set_experiment("Bati_Bank_Credit_Scoring")
    
    # --- Model 1: Logistic Regression (Baseline) ---
    lr_params = {
        'C': [0.1, 1.0, 10.0],
        'solver': ['liblinear']
    }
    auc_lr = train_model("LogisticRegression", LogisticRegression(max_iter=1000), 
                         lr_params, X_train, X_test, y_train, y_test)

    # --- Model 2: Random Forest (Challenger) ---
    rf_params = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }
    auc_rf = train_model("RandomForest", RandomForestClassifier(random_state=42), 
                         rf_params, X_train, X_test, y_train, y_test)

    # --- Compare and Print Best ---
    print("\n=== Experiment Summary ===")
    print(f"Logistic Regression AUC: {auc_lr:.4f}")
    print(f"Random Forest AUC:       {auc_rf:.4f}")
    
    if auc_rf > auc_lr:
        print(">> Recommendation: Deploy Random Forest")
    else:
        print(">> Recommendation: Deploy Logistic Regression (Simpler & Better)")

if __name__ == "__main__":
    main()