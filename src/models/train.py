import mlflow
import mlflow.sklearn
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from src.models.pipeline import create_pipeline

def train_model():
    """Обучает модель и логирует результаты в MLflow."""
    # Загрузка данных
    train_path = "data/processed/train.csv"
    df = pd.read_csv(train_path)
    X = df.drop(columns=["default.payment.next.month"])
    y = df["default.payment.next.month"]

    # Определение признаков
    numeric_features = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'PAY_AMT1']
    categorical_features = ['EDUCATION', 'MARRIAGE', 'PAY_0']

    # Создание пайплайна
    pipeline = create_pipeline(numeric_features, categorical_features)

    # Логирование в MLflow
    mlflow.set_tracking_uri("file:///./mlruns")
    mlflow.set_experiment("Credit_Default_Prediction")

    with mlflow.start_run():
        # Логирование параметров
        mlflow.log_param("model_type", "GradientBoosting")
        mlflow.log_params(pipeline.named_steps['classifier'].get_params())

        # Обучение модели
        pipeline.fit(X, y)

        # Расчёт метрик
        y_proba = pipeline.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_proba)
        y_pred = pipeline.predict(X)
        f1 = f1_score(y, y_pred)

        # Логирование метрик
        mlflow.log_metric("train_auc", auc)
        mlflow.log_metric("train_f1", f1)

        # Сохранение метрик в файл
        metrics = {
            "train_auc": float(auc),
            "train_f1": float(f1)
        }
        with open("metrics.json", "w") as f:
            json.dump(metrics, f)

        # Логирование модели
        mlflow.sklearn.log_model(pipeline, "model", registered_model_name="CreditDefaultModel")
        
        # Сохранение модели локально
        import joblib
        joblib.dump(pipeline, "models/credit_default_model.pkl")

        print(f"Train AUC: {auc:.4f}")
        print(f"Train F1: {f1:.4f}")

if __name__ == "__main__":
    train_model()
