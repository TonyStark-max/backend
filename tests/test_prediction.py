from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


APPROVAL_APPLICATION = {
    "dependents": 2,
    "employment_type": "Private",
    "annual_income": 6000000,
    "credit_score": 720,
    "loan_amount": 15000000,
    "loan_tenure": 10,
    "education": "Graduate",
}


REJECTION_APPLICATION = {
    "dependents": 3,
    "employment_type": "Unemployed",
    "annual_income": 200000,
    "credit_score": 350,
    "loan_amount": 30000000,
    "loan_tenure": 5,
    "education": "High School",
}


def test_prediction_returns_valid_response():

    response = client.post(
        "/predict",
        json=APPROVAL_APPLICATION,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in {
        "Approved",
        "Rejected",
    }

    assert 0 <= data["approved_probability"] <= 1

    assert 0 <= data["rejected_probability"] <= 1

    assert (
        round(
            data["approved_probability"]
            + data["rejected_probability"],
            4,
        )
        == 1.0
    )


def test_strong_application():

    response = client.post(
        "/predict",
        json=APPROVAL_APPLICATION,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == "Approved"


def test_weak_application():

    response = client.post(
        "/predict",
        json=REJECTION_APPLICATION,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == "Rejected"


def test_prediction_contains_suggestions():

    response = client.post(
        "/predict",
        json=REJECTION_APPLICATION,
    )

    assert response.status_code == 200

    data = response.json()

    assert "suggestions" in data

    assert isinstance(
        data["suggestions"],
        list,
    )


def test_invalid_prediction_input():

    application = APPROVAL_APPLICATION.copy()

    application["credit_score"] = 950

    response = client.post(
        "/predict",
        json=application,
    )

    assert response.status_code == 422


def test_prediction_contains_explainability():
    response = client.post(
        "/predict",
        json=REJECTION_APPLICATION,
    )
    assert response.status_code == 200
    data = response.json()

    assert "explanation" in data
    explanation = data["explanation"]
    assert explanation is not None

    assert "top_negative_factors" in explanation
    assert "positive_factors" in explanation
    assert "action_plan" in explanation
    assert "disclaimer" in explanation

    # Check top negative factors structure
    assert len(explanation["top_negative_factors"]) > 0
    top_factor = explanation["top_negative_factors"][0]
    assert "feature" in top_factor
    assert "label" in top_factor
    assert "user_value" in top_factor
    assert "impact_level" in top_factor
    assert "explanation" in top_factor
    assert "is_actionable" in top_factor

    # Check action plan structure
    assert len(explanation["action_plan"]) > 0
    first_action = explanation["action_plan"][0]
    assert first_action["priority"] == 1
    assert "title" in first_action
    assert "reason" in first_action
    assert "recommendation" in first_action


def test_profile_specific_explanations_differ():
    # Profile 1: Rejection driven by low CIBIL score
    bad_credit_app = {
        "dependents": 0,
        "employment_type": "Private",
        "annual_income": 3000000,
        "credit_score": 380,
        "loan_amount": 1000000,
        "loan_tenure": 15,
        "education": "Graduate",
    }
    res1 = client.post("/predict", json=bad_credit_app).json()
    assert res1["prediction"] == "Rejected"
    exp1 = res1["explanation"]
    assert exp1["top_negative_factors"][0]["feature"] == "Credit_Score"
    assert exp1["action_plan"][0]["factor_label"] == "Credit Score (CIBIL)"

    # Profile 2: Rejection driven by excessive loan amount & short tenure
    excessive_loan_app = {
        "dependents": 1,
        "employment_type": "Self-Employed",
        "annual_income": 1000000,
        "credit_score": 620,
        "loan_amount": 25000000,
        "loan_tenure": 2,
        "education": "Graduate",
    }
    res2 = client.post("/predict", json=excessive_loan_app).json()
    assert res2["prediction"] == "Rejected"
    exp2 = res2["explanation"]
    # In this profile, Loan_Amount is the dominant negative factor
    assert exp2["top_negative_factors"][0]["feature"] == "Loan_Amount"
    assert exp2["action_plan"][0]["factor_label"] == "Requested Loan Amount"

    # Confirm explanations are truly personalized and distinct
    assert exp1["top_negative_factors"][0]["feature"] != exp2["top_negative_factors"][0]["feature"]


def test_ntc_prediction_contains_shared_explainability_format():
    ntc_application = {
        "dependents": 2,
        "employment_type": "Private",
        "annual_income": 500000,
        "loan_amount": 600000,
        "loan_tenure": 10,
        "education": "Graduate",
        "monthly_expenses": 30000,
    }

    response = client.post("/new-predict", json=ntc_application)

    assert response.status_code == 200

    data = response.json()
    assert "explanation" in data
    assert data["explanation"] is not None
    assert "top_negative_factors" in data["explanation"]
    assert "positive_factors" in data["explanation"]
    assert "action_plan" in data["explanation"]
    assert "disclaimer" in data["explanation"]

    if data["explanation"]["top_negative_factors"]:
        first_negative = data["explanation"]["top_negative_factors"][0]
        assert "label" in first_negative
        assert "user_value" in first_negative
        assert "impact_level" in first_negative

    assert any(
        factor["label"] in {"Requested Loan Amount", "Annual Income", "Loan Tenure"}
        for factor in data["explanation"]["all_factors"]
    )
