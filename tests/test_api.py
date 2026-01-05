# tests/test_api.py

from fastapi.testclient import TestClient # Импортируем из fastapi, а не starlette
from src.api.app import app

def test_read_root():
    client = TestClient(app) # Передаём app как аргумент
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Credit Scoring API is running"}

def test_predict():
    client = TestClient(app)
    # Передайте ВСЕ "сырые" признаки, которые ожидает Pydantic модель (до FE), и которые были в исходном датасете
    test_data = {
        "ID": 1, # Необходимо?
        "LIMIT_BAL": 20000.0,
        "SEX": 1, # Добавлен
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 25,
        "PAY_0": 0, # PAY_0 соответствует PAY_1
        "PAY_2": 0, # Добавлен
        "PAY_3": 0, # Добавлен
        "PAY_4": 0, # Добавлен
        "PAY_5": 0, # Добавлен
        "PAY_6": 0, # Добавлен
        "BILL_AMT1": 3000.0,
        "BILL_AMT2": 2500.0,
        "BILL_AMT3": 2000.0,
        "BILL_AMT4": 1500.0,
        "BILL_AMT5": 1000.0,
        "BILL_AMT6": 500.0,
        "PAY_AMT1": 2000.0,
        "PAY_AMT2": 1800.0,
        "PAY_AMT3": 1600.0,
        "PAY_AMT4": 1400.0,
        "PAY_AMT5": 1200.0,
        "PAY_AMT6": 1000.0
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    # Проверим, что в ответе есть нужные поля
    json_response = response.json()
    assert "probability" in json_response
    assert "prediction" in json_response
    assert isinstance(json_response["probability"], float)
    assert json_response["prediction"] in [0, 1]
