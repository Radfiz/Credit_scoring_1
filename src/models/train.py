# src/models/train.py

import sys
import os
# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve
from src.models.pipeline import create_pipeline
import matplotlib.pyplot as plt

def create_features(df):
    """Применяет Feature Engineering к датафрейму."""
    df = df.copy()

    # Создание признака 'utilization'
    df['utilization'] = df['BILL_AMT1'] / (df['LIMIT_BAL'] + 1)
    df['utilization'] = df['utilization'].replace([np.inf, -np.inf], np.nan)

    # Создание признака 'pay_ratio'
    df['pay_ratio'] = df['PAY_AMT1'] / (df['BILL_AMT1'] + 1)
    df['pay_ratio'] = df['pay_ratio'].replace([np.inf, -np.inf], np.nan)

    # Создание признака 'bill_trend' (упрощённо)
    bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
    df['bill_trend'] = df[bill_cols].diff(axis=1).mean(axis=1, skipna=True).fillna(0)

    # Биннинг возраста
    df['age_group'] = pd.cut(df['AGE'], bins=[0, 25, 35, 50, 100], labels=['Young', 'Adult', 'Senior', 'Elder'], right=False)

    return df

def train_model():
    """Обучает модель и логирует результаты в MLflow."""
    # Загрузка данных
    train_path = "data/processed/train.csv"
    df = pd.read_csv(train_path)

    # Применение Feature Engineering
    df_fe = create_features(df)

    # Разделение на X и y
    X = df_fe.drop(columns=["default.payment.next.month"])
    y = df_fe["default.payment.next.month"]

    print("Columns in X after FE:", X.columns.tolist())

    # Определение признаков (теперь они должны существовать после FE)
    numeric_features = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'utilization', 'pay_ratio', 'bill_trend']
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE', 'age_group', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

    # Убедимся, что все перечисленные признаки существуют в X
    missing_features = set(numeric_features + categorical_features) - set(X.columns)
    if missing_features:
        print(f"ERROR: Missing features in X: {missing_features}")
        return

    # Создание пайплайна
    pipeline = create_pipeline(numeric_features, categorical_features)

    # --- НАСТРОЙКА ПОДБОРА ГИПЕРПАРАМЕТРОВ ---
    # Пример параметров для GradientBoostingClassifier
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__max_depth': [3, 5]
    }

    # Разделим X_train, X_val, y_train, y_val для подбора
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3, # 3 фолда
        scoring='roc_auc',
        n_jobs=-1, # Использовать все ядра
        verbose=1 # Печатать прогресс
    )

    print("Starting hyperparameter tuning...")
    grid_search.fit(X_train_split, y_train_split)

    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")
    # --- КОНЕЦ НАСТРОЙКИ ---

    # Оценка лучшей модели на валидационной выборке (для логирования)
    y_val_proba = best_pipeline.predict_proba(X_val_split)[:, 1]
    val_auc = roc_auc_score(y_val_split, y_val_proba)
    y_val_pred = best_pipeline.predict(X_val_split)
    val_f1 = f1_score(y_val_split, y_val_pred)
    val_precision = precision_score(y_val_split, y_val_pred)
    val_recall = recall_score(y_val_split, y_val_pred)

    # --- ПОСТРОЕНИЕ ROC-КРИВОЙ ---
    fpr, tpr, _ = roc_curve(y_val_split, y_val_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {val_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    roc_plot_path = "roc_curve.png"
    plt.savefig(roc_plot_path)
    plt.close() # Закрываем фигуру, чтобы освободить память
    # --- КОНЕЦ ПОСТРОЕНИЯ ---

    mlflow.set_experiment("Credit_Default_Prediction_HPT")

    with mlflow.start_run():
        # Логирование параметров
        mlflow.log_param("model_type", "GradientBoosting")
        mlflow.log_params(best_params)

        # Логирование метрик
        mlflow.log_metric("val_auc", val_auc)
        mlflow.log_metric("val_f1", val_f1)
        mlflow.log_metric("val_precision", val_precision)
        mlflow.log_metric("val_recall", val_recall)

        # --- ЛОГИРОВАНИЕ ГРАФИКА ---
        mlflow.log_artifact(roc_plot_path, artifact_path="plots")
        # --- КОНЕЦ ЛОГИРОВАНИЯ ---

        # Логирование модели
        mlflow.sklearn.log_model(best_pipeline, "model")

        # Сохранение модели локально
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", "credit_default_model.pkl")
        joblib.dump(best_pipeline, model_path)

        # Сохранение метрик в JSON
        metrics = {
            "val_auc": val_auc,
            "val_f1": val_f1,
            "val_precision": val_precision,
            "val_recall": val_recall
        }
        import json
        with open("metrics.json", "w") as f:
            json.dump(metrics, f)

        print(f"Validation AUC: {val_auc:.4f}")
        print(f"Validation F1: {val_f1:.4f}")
        print(f"Validation Precision: {val_precision:.4f}")
        print(f"Validation Recall: {val_recall:.4f}")

    # Удаляем временный файл графика после логирования (опционально)
    # os.remove(roc_plot_path)

if __name__ == "__main__":
    train_model()