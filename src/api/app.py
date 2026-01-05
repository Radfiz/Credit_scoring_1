# src/api/app.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os
# Импортируем нашу функцию FE
from src.features.build_features import create_features

app = FastAPI(title="Credit Scoring API", version="1.0.0")

class CreditApplication(BaseModel):
    # Определим Pydantic модель с *всеми* признаками, которые поступают *до* FE, и которые есть в исходном датасете
    ID: int 
    LIMIT_BAL: float
    SEX: int 
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int 
    PAY_2: int 
    PAY_3: int 
    PAY_4: int 
    PAY_5: int 
    PAY_6: int 
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float

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
        return {"error": "Model file not found"}
    model = joblib.load(model_path)

    # Преобразование данных в DataFrame
    input_data = request.dict()
    input_df = pd.DataFrame([input_data])

    # Удаляем ID, если он не нужен для предсказания
    if 'ID' in input_df.columns:
        input_df = input_df.drop(columns=['ID'])

    input_df_fe = create_features(input_df)
    # --- КОНЕЦ FE ---

    # Предсказание
    try:
        probability = model.predict_proba(input_df_fe)[0][1]
        prediction = int(model.predict(input_df_fe)[0])
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

    return {
        "probability": float(probability),
        "prediction": prediction
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
