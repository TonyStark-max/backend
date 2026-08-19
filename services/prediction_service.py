from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd

from services.explainability_service import explain_prediction

# Compatibility shims for loading scikit-learn 1.6.1 serialized pipelines
# across varied Python/scikit-learn runtime versions without altering model artifacts.
try:
    import sklearn.compose._column_transformer
    if not hasattr(sklearn.compose._column_transformer, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
except Exception:
    pass

try:
    import sklearn._loss
    if not hasattr(sklearn._loss, "CyHalfBinomialLoss"):
        sklearn._loss.CyHalfBinomialLoss = getattr(
            sklearn._loss, "HalfBinomialLoss", None
        )
    if "_loss" not in sys.modules:
        sys.modules["_loss"] = sklearn._loss
except Exception:
    pass

try:
    import sklearn.impute._base
    if not hasattr(sklearn.impute._base.SimpleImputer, "_fill_dtype"):
        sklearn.impute._base.SimpleImputer._fill_dtype = property(
            lambda self: getattr(self, "_fit_dtype", np.float64)
        )
except Exception:
    pass

from services.loan_analysis_service import generate_loan_amount_analysis
from services.suggestion_service import generate_suggestions


# ============================================================
# MODEL CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "loan_model.pkl"
)

MODEL_VERSION = "7-feature-gb-v2"


# ============================================================
# LOAD MODEL BUNDLE
# ============================================================

model = None
encoders = {}
FEATURES = []
model_load_error = None

# Track version increments
_version_counter = 2

def reload_model():
    global model, encoders, FEATURES, model_load_error, MODEL_VERSION, _version_counter
    try:
        model_bundle = joblib.load(MODEL_PATH)
        model = model_bundle["model"]
        encoders = model_bundle["encoders"]
        FEATURES = model_bundle["features"]
        model_load_error = None
        
        # Increment version on successful reload
        _version_counter += 1
        MODEL_VERSION = f"7-feature-gb-v{_version_counter}"
        print(f"Model reloaded successfully: {MODEL_VERSION}")
    except Exception as exc:
        model_load_error = (
            f"{type(exc).__name__}: "
            "failed to load model bundle"
        )
        print(model_load_error)

# Initial load
reload_model()


# ============================================================
# PREDICTION
# ============================================================

def predict_loan(application_data: dict[str, Any]) -> dict[str, Any]:
    """
    Generate a loan approval prediction and maximum eligible loan estimation
    using the trained 7-feature machine learning pipeline.
    """

    if model is None:
        raise RuntimeError(
            "Model is not loaded"
            if model_load_error is None
            else f"Model is not loaded: {model_load_error}"
        )

    # --------------------------------------------------------
    # Create input using the exact training feature names
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [
            {
                "Dependents": application_data["dependents"],
                "Employment_Type": application_data["employment_type"],
                "Credit_Score": application_data["credit_score"],
                "Annual_Income": application_data["annual_income"],
                "Loan_Amount": application_data["loan_amount"],
                "Loan_Tenure": application_data["loan_tenure"],
                "Education": application_data["education"],
            }
        ]
    )

    # --------------------------------------------------------
    # Apply the EXACT encoders used during training
    # --------------------------------------------------------

    for column, encoder in encoders.items():
        input_data[column] = encoder.transform(
            input_data[column]
        )

    # --------------------------------------------------------
    # Maintain EXACT feature order
    # --------------------------------------------------------

    input_data = input_data[FEATURES]

    # --------------------------------------------------------
    # ML prediction
    # --------------------------------------------------------

    prediction = model.predict(input_data)[0]

    # --------------------------------------------------------
    # Prediction probabilities
    # --------------------------------------------------------

    probabilities = model.predict_proba(input_data)[0]

    rejected_probability = float(probabilities[0])
    approved_probability = float(probabilities[1])

    # --------------------------------------------------------
    # Convert prediction to readable label
    # --------------------------------------------------------

    prediction_label = (
        "Approved"
        if prediction == 1
        else "Rejected"
    )

    # --------------------------------------------------------
    # Generate applicant guidance & explainability
    # --------------------------------------------------------

    suggestions = generate_suggestions(
        application_data
    )

    explanation = explain_prediction(
        application_data=application_data,
        encoded_df=input_data,
        model=model,
        features=FEATURES,
        is_approved=(prediction == 1),
    )

    # --------------------------------------------------------
    # Evaluate Loan Amount What-If Scenarios
    # --------------------------------------------------------

    loan_amount_analysis = generate_loan_amount_analysis(
        application_data=application_data,
        model=model,
        encoders=encoders,
        features=FEATURES,
    )

    requested_amount = int(application_data["loan_amount"])
    recommended_amount = loan_amount_analysis["recommendedAmount"]
    recommended_prob = loan_amount_analysis["recommendedApprovalProbability"] / 100.0

    return {
        "prediction": prediction_label,
        "approved_probability": round(
            approved_probability,
            4,
        ),
        "rejected_probability": round(
            rejected_probability,
            4,
        ),
        "suggestions": suggestions,
        "explanation": explanation,
        "loan_amount_analysis": loan_amount_analysis,
        "requested_loan_amount": requested_amount,
        "maximum_eligible_amount": recommended_amount,
        "maximum_eligible_prediction": "Approved" if recommended_amount is not None else "Rejected",
        "max_eligible_approved_probability": round(recommended_prob, 4),
        "max_loan_status": loan_amount_analysis.get("max_loan_status", "eligible" if recommended_amount is not None else "none_eligible"),
        "max_loan_message": loan_amount_analysis.get("max_loan_message", ""),
    }


def estimate_maximum_loan(application_data: dict[str, Any]) -> dict[str, Any]:
    """
    Dedicated calculation for loan amount what-if analysis.
    """
    analysis = generate_loan_amount_analysis(
        application_data=application_data,
        model=model,
        encoders=encoders,
        features=FEATURES,
    )
    requested_amount = int(application_data["loan_amount"])
    recommended_amount = analysis["recommendedAmount"]
    recommended_prob = analysis["recommendedApprovalProbability"] / 100.0
    return {
        "requested_loan_amount": requested_amount,
        "maximum_eligible_amount": recommended_amount,
        "maximum_eligible_prediction": "Approved" if recommended_amount is not None else "Rejected",
        "max_eligible_approved_probability": round(recommended_prob, 4),
        "max_loan_status": analysis.get("max_loan_status", "eligible" if recommended_amount is not None else "none_eligible"),
        "max_loan_message": analysis.get("max_loan_message", ""),
        "loan_amount_analysis": analysis,
    }


# ============================================================
# MODEL STATUS
# ============================================================

def get_model_status() -> dict:
    """
    Return information about the loaded ML model.
    """

    return {
        "model_loaded": model is not None,
        "model_type": type(model).__name__,
        "model_file": MODEL_PATH.name,
        "model_version": MODEL_VERSION,
        "features": FEATURES,
        "model_load_error": model_load_error,
    }
