from services.suggestion_service import generate_suggestions


def test_strong_profile():

    application = {
        "dependents": 2,
        "employment_type": "Private",
        "annual_income": 6000000,
        "credit_score": 720,
        "loan_amount": 15000000,
        "loan_tenure": 10,
        "education": "Graduate",
    }

    suggestions = generate_suggestions(application)

    assert isinstance(suggestions, list)

    assert len(suggestions) >= 1


def test_weak_profile():

    application = {
        "dependents": 3,
        "employment_type": "Unemployed",
        "annual_income": 200000,
        "credit_score": 350,
        "loan_amount": 30000000,
        "loan_tenure": 5,
        "education": "High School",
    }

    suggestions = generate_suggestions(application)

    assert len(suggestions) >= 4

    combined = " ".join(suggestions).lower()

    assert "credit score" in combined
    assert "loan amount" in combined
    assert "employment" in combined