from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import pandas as pd

app = FastAPI(title="Credit Scoring API", version="1.0.0")

class CreditApplication(BaseModel):
    LIMIT_BAL: float
    AGE: int
    BILL_AMT1: float
    PAY_AMT1: float
    EDUCATION: int
    MARRIAGE: int
    PAY_0: int

@app.get("/")
def read_root():
    return {"message": "Credit Scoring API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict(request: CreditApplication):
    # Загрузка модели
    model_path = os.path.join("models", "credit_default_model.pkl")
    if not os.path.exists(model_path):
        return {"error": "Model not found. Please train the model first."}
    
    model = joblib.load(model_path)

    # Преобразование данных в DataFrame
    input_data = request.dict()
    input_df = pd.DataFrame([input_data])

    # Предсказание
    probability = model.predict_proba(input_df)[0][1]
    prediction = int(model.predict(input_df)[0])

    return {
        "probability": float(probability),
        "prediction": prediction,
        "risk_level": "high" if probability > 0.5 else "low"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
