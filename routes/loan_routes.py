from fastapi import APIRouter

from schemas.loan_schema import (
    LoanApplication,
    MaxLoanEstimateResponse,
    PredictionResponse,
    ValidationResponse,
)

from services.prediction_service import (
    estimate_maximum_loan,
    get_model_status,
    predict_loan,
)


router = APIRouter()


# ============================================================
# HEALTH
# ============================================================

@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "loan-approval-prediction",
    }


# ============================================================
# VALIDATION
# ============================================================

@router.post(
    "/validate",
    response_model=ValidationResponse,
)
def validate_application(
    application: LoanApplication,
):
    return {
        "status": "valid",
        "message": "Loan application data is valid",
        "data": application.model_dump(),
    }


# ============================================================
# MODEL STATUS
# ============================================================

@router.get("/model-status")
def model_status():
    return get_model_status()


# ============================================================
# PREDICTION
# ============================================================

@router.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_application(
    application: LoanApplication,
):
    return predict_loan(
        application.model_dump()
    )


# ============================================================
# MAXIMUM ELIGIBLE LOAN ESTIMATION
# ============================================================

@router.post(
    "/max-eligible-loan",
    response_model=MaxLoanEstimateResponse,
)
def max_eligible_loan_application(
    application: LoanApplication,
):
    return estimate_maximum_loan(
        application.model_dump()
    )