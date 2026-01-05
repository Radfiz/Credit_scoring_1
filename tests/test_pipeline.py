# tests/test_pipeline.py
import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from src.models.pipeline import create_pipeline

def test_create_pipeline_returns_pipeline():
    numeric_features = ['LIMIT_BAL', 'AGE']
    categorical_features = ['EDUCATION', 'MARRIAGE']
    pipeline = create_pipeline(numeric_features, categorical_features)
    assert isinstance(pipeline, Pipeline)

def test_pipeline_fits_and_predicts():
    numeric_features = ['LIMIT_BAL', 'AGE']
    categorical_features = ['EDUCATION', 'MARRIAGE']
    pipeline = create_pipeline(numeric_features, categorical_features)

    X = pd.DataFrame({
        'LIMIT_BAL': [10000, 20000, 30000],
        'AGE': [25, 35, 45],
        'EDUCATION': [1, 2, 3],
        'MARRIAGE': [1, 2, 1]
    })
    # Явно преобразуем категориальные признаки в строки
    X['EDUCATION'] = X['EDUCATION'].astype(str)
    X['MARRIAGE'] = X['MARRIAGE'].astype(str)
    
    y = pd.Series([0, 1, 0])

    # Проверим, что обучение не вызывает ошибок
    pipeline.fit(X, y)

    # Проверим, что предсказание не вызывает ошибок
    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)

    # Проверим, что предсказания имеют правильную форму
    assert predictions.shape[0] == X.shape[0]
    assert probabilities.shape[0] == X.shape[0]
    assert probabilities.shape[1] == 2  # Бинарная классификация

def test_pipeline_handles_missing_values():
    numeric_features = ['LIMIT_BAL', 'AGE']
    categorical_features = ['EDUCATION', 'MARRIAGE']
    pipeline = create_pipeline(numeric_features, categorical_features)

    X = pd.DataFrame({
        'LIMIT_BAL': [10000, np.nan, 30000],
        'AGE': [25, 35, np.nan],
        'EDUCATION': ['1', '2', np.nan],  # Строки для категориальных
        'MARRIAGE': ['1', np.nan, '1']    # Строки для категориальных
    })
    
    y = pd.Series([0, 1, 0])

    # Проверим, что обучение с пропусками не вызывает ошибок
    pipeline.fit(X, y)

    # Проверим, что предсказание с пропусками не вызывает ошибок
    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)

    assert predictions.shape[0] == X.shape[0]
    assert probabilities.shape[0] == X.shape[0]