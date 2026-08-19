from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

STRONG_APPLICATION = {
    "dependents": 2,
    "employment_type": "Private",
    "annual_income": 6000000,
    "credit_score": 720,
    "loan_amount": 5000000,
    "loan_tenure": 10,
    "education": "Graduate",
}

MODERATE_REJECTED_APPLICATION = {
    "dependents": 2,
    "employment_type": "Private",
    "annual_income": 600000,
    "credit_score": 650,
    "loan_amount": 10000000,
    "loan_tenure": 5,
    "education": "Graduate",
}

WEAK_APPLICATION = {
    "dependents": 3,
    "employment_type": "Unemployed",
    "annual_income": 200000,
    "credit_score": 350,
    "loan_amount": 30000000,
    "loan_tenure": 5,
    "education": "High School",
}

def test_predict_endpoint_approved_increases_max_loan():
    response = client.post("/predict", json=STRONG_APPLICATION)
    assert response.status_code == 200
    data = response.json()
    
    assert data["prediction"] == "Approved"
    assert "loan_amount_analysis" in data
    analysis = data["loan_amount_analysis"]
    assert analysis["recommendedAmount"] is not None
    # For approved application, capacity searches upward (>= requested amount)
    assert analysis["recommendedAmount"] >= STRONG_APPLICATION["loan_amount"]
    assert analysis["mode"] == "UPWARD_CAPACITY"

def test_predict_endpoint_rejected_reduces_max_loan():
    response = client.post("/predict", json=MODERATE_REJECTED_APPLICATION)
    assert response.status_code == 200
    data = response.json()
    
    assert data["prediction"] == "Rejected"
    assert "loan_amount_analysis" in data
    analysis = data["loan_amount_analysis"]
    # For rejected application with decent profile, reduced eligible amount is found (< requested amount)
    assert analysis["recommendedAmount"] is not None
    assert analysis["recommendedAmount"] < MODERATE_REJECTED_APPLICATION["loan_amount"]
    assert analysis["mode"] == "DOWNWARD_IMPROVEMENT"

def test_predict_weak_application_no_eligible_amount():
    response = client.post("/predict", json=WEAK_APPLICATION)
    assert response.status_code == 200
    data = response.json()
    
    assert "loan_amount_analysis" in data
    analysis = data["loan_amount_analysis"]
    assert analysis["recommendedAmount"] is None

def test_predict_ntc_approved_and_rejected_capacity():
    # 1. Approved NTC profile
    ntc_strong = {
        "dependents": 0,
        "employment_type": "Private",
        "annual_income": 1200000,
        "loan_amount": 1000000,
        "loan_tenure": 10,
        "education": "Graduate",
        "monthly_expenses": 25000
    }
    res1 = client.post("/new-predict", json=ntc_strong)
    assert res1.status_code == 200
    d1 = res1.json()
    assert d1["prediction"] == "Approved"
    assert d1["maximum_eligible_amount"] is not None
    assert d1["maximum_eligible_amount"] >= ntc_strong["loan_amount"]
    assert "loan_amount_analysis" in d1

    # 2. Rejected NTC profile (excessive loan amount requested)
    ntc_rejected = {
        "dependents": 1,
        "employment_type": "Private",
        "annual_income": 600000,
        "loan_amount": 10000000,
        "loan_tenure": 5,
        "education": "Graduate",
        "monthly_expenses": 25000
    }
    res2 = client.post("/new-predict", json=ntc_rejected)
    assert res2.status_code == 200
    d2 = res2.json()
    assert d2["prediction"] == "Rejected"
    assert d2["maximum_eligible_amount"] is not None
    assert d2["maximum_eligible_amount"] < ntc_rejected["loan_amount"]
