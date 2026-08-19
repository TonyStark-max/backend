import sys
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def run():
    print("Testing /health...")
    resp = client.get("/health")
    assert resp.status_code == 200, resp.text
    print("OK: Health check passed")

    print("Testing /assistant/suggestions...")
    resp = client.get("/assistant/suggestions")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "categories" in data
    assert "suggestions" in data
    assert len(data["categories"]) >= 3
    print(f"OK: Found {len(data['suggestions'])} suggestions across categories: {data['categories']}")

    print("Testing /assistant/chat...")
    resp = client.post("/assistant/chat", json={
        "message": "How do I improve my CIBIL credit score?",
        "history": [],
    })
    assert resp.status_code == 200, resp.text
    chat_data = resp.json()
    assert "reply" in chat_data
    assert len(chat_data["reply"]) > 20
    assert "suggestions" in chat_data
    print("OK: Chat response generated successfully! Reply length:", len(chat_data["reply"]))

    print("Testing /assistant/chat with financial context...")
    resp = client.post("/assistant/chat", json={
        "message": "What is my approval likelihood?",
        "history": [],
        "context": {
            "credit_score": 750,
            "annual_income": 1200000,
            "loan_amount": 2500000,
            "loan_tenure": 10,
            "employment_type": "Private"
        }
    })
    assert resp.status_code == 200, resp.text
    assert "reply" in resp.json()
    print("OK: Contextual chat response generated successfully!")

    print("Testing /predict (Existing loan model)...")
    resp = client.post("/predict", json={
        "dependents": 2,
        "employment_type": "Private",
        "annual_income": 6000000,
        "credit_score": 720,
        "loan_amount": 15000000,
        "loan_tenure": 10,
        "education": "Graduate",
    })
    assert resp.status_code == 200, resp.text
    pred_data = resp.json()
    assert pred_data["prediction"] == "Approved"
    print("OK: Prediction response:", pred_data["prediction"])

    print("\n=============================================")
    print(" ALL BACKEND INTEGRATION TESTS PASSED 100%!")
    print("=============================================")

if __name__ == "__main__":
    run()
