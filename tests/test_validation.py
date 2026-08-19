from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


VALID_APPLICATION = {
    "dependents": 2,
    "employment_type": "Private",
    "annual_income": 6000000,
    "credit_score": 720,
    "loan_amount": 15000000,
    "loan_tenure": 10,
    "education": "Graduate",
}


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


def test_model_status():
    response = client.get("/model-status")

    assert response.status_code == 200

    data = response.json()

    assert data["model_loaded"] is True
    assert data["model_type"] == "GradientBoostingClassifier"
    assert data["model_file"] == "loan_model.pkl"
    assert data["model_version"].startswith("7-feature-gb-v")


def test_valid_application():
    response = client.post(
        "/validate",
        json=VALID_APPLICATION,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "valid"
    assert data["data"] == VALID_APPLICATION


def test_invalid_dependents():
    application = VALID_APPLICATION.copy()
    application["dependents"] = 5

    response = client.post(
        "/validate",
        json=application,
    )

    assert response.status_code == 422


def test_invalid_employment_type():
    application = VALID_APPLICATION.copy()
    application["employment_type"] = "Unknown"

    response = client.post(
        "/validate",
        json=application,
    )

    assert response.status_code == 422


def test_invalid_education():
    application = VALID_APPLICATION.copy()
    application["education"] = "Unknown"

    response = client.post(
        "/validate",
        json=application,
    )

    assert response.status_code == 422


def test_invalid_credit_score_low():
    application = VALID_APPLICATION.copy()
    application["credit_score"] = 299

    response = client.post(
        "/validate",
        json=application,
    )

    assert response.status_code == 422


def test_invalid_credit_score_high():
    application = VALID_APPLICATION.copy()
    application["credit_score"] = 901

    response = client.post(
        "/validate",
        json=application,
    )

    assert response.status_code == 422


def test_invalid_income():
    application = VALID_APPLICATION.copy()
    application["annual_income"] = 0

    response = client.post(
        "/validate",
        json=application,
    )

    assert response.status_code == 422


def test_invalid_loan_amount():
    application = VALID_APPLICATION.copy()
    application["loan_amount"] = 0

    response = client.post(
        "/validate",
        json=application,
    )

    assert response.status_code == 422


def test_invalid_loan_tenure():
    application = VALID_APPLICATION.copy()
    application["loan_tenure"] = 31

    response = client.post(
        "/validate",
        json=application,
    )

    assert response.status_code == 422


def test_missing_credit_score():
    application = VALID_APPLICATION.copy()
    del application["credit_score"]

    response = client.post(
        "/validate",
        json=application,
    )

    assert response.status_code == 422