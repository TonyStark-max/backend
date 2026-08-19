from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_assistant_suggestions_endpoint():
    response = client.get("/assistant/suggestions")
    assert response.status_code == 200
    data = response.json()
    assert "categories" in data
    assert "suggestions" in data
    assert len(data["categories"]) > 0
    assert len(data["suggestions"]) > 0

    # Verify categories exist
    assert "Loan Approval Queries" in data["categories"]
    assert "Credit Score Queries" in data["categories"]


def test_assistant_chat_endpoint():
    payload = {
        "message": "How can I improve my CIBIL credit score?",
        "history": [],
        "context": None,
    }
    response = client.post("/assistant/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert len(data["reply"]) > 0
    assert "suggestions" in data
    assert isinstance(data["suggestions"], list)
    assert data["status"] == "success"


def test_assistant_chat_with_context():
    payload = {
        "message": "Is my profile eligible for a home loan?",
        "history": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hello! How can I assist you with loans today?"}
        ],
        "context": {
            "credit_score": 760,
            "annual_income": 1200000,
            "loan_amount": 3500000,
            "loan_tenure": 15,
            "employment_type": "Private",
        },
    }
    response = client.post("/assistant/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert len(data["reply"]) > 0


def test_assistant_chat_empty_message_validation():
    payload = {
        "message": "",
        "history": [],
    }
    response = client.post("/assistant/chat", json=payload)
    assert response.status_code == 422
