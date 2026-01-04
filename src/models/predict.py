import joblib
import pandas as pd
import os

def load_model(model_path="models/credit_default_model.pkl"):
    """Загружает обученную модель"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)

def predict(input_data, model_path="models/credit_default_model.pkl"):
    """Делает предсказание на новых данных"""
    model = load_model(model_path)
    
    # Проверяем, является ли input_data словарем или DataFrame
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        input_df = input_data
    else:
        raise TypeError("input_data должен быть словарем или pandas DataFrame")
    
    # Предсказание
    probability = model.predict_proba(input_df)[:, 1]
    prediction = model.predict(input_df)
    
    return {
        "predictions": prediction.tolist(),
        "probabilities": probability.tolist()
    }
