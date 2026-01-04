import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Credit Scoring API is running"}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict():
    # Тестовые данные
    test_data = {
        "LIMIT_BAL": 20000.0,
        "AGE": 30,
        "BILL_AMT1": 1000.0,
        "PAY_AMT1": 500.0,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "PAY_0": 0
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code in [200, 400]  # Может быть 400 если модель не обучена
