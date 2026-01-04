import pytest
import pandas as pd
from src.models.pipeline import create_pipeline

def test_create_pipeline():
    """Тестирует создание пайплайна."""
    numeric_features = ['LIMIT_BAL', 'AGE']
    categorical_features = ['EDUCATION', 'MARRIAGE']
    pipeline = create_pipeline(numeric_features, categorical_features)
    assert pipeline is not None
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0][0] == 'preprocessor'
    assert pipeline.steps[1][0] == 'classifier'

def test_pipeline_fit():
    """Тестирует обучение пайплайна."""
    numeric_features = ['LIMIT_BAL', 'AGE']
    categorical_features = ['EDUCATION', 'MARRIAGE']
    pipeline = create_pipeline(numeric_features, categorical_features)

    # Создаем фиктивные данные
    X = pd.DataFrame({
        'LIMIT_BAL': [10000, 20000],
        'AGE': [30, 40],
        'EDUCATION': [1, 2],
        'MARRIAGE': [1, 2]
    })
    y = pd.Series([0, 1])

    pipeline.fit(X, y)
    assert hasattr(pipeline, 'named_steps')
    
def test_pipeline_predict():
    """Тестирует предсказания пайплайна."""
    numeric_features = ['LIMIT_BAL', 'AGE']
    categorical_features = ['EDUCATION', 'MARRIAGE']
    pipeline = create_pipeline(numeric_features, categorical_features)

    # Создаем фиктивные данные
    X = pd.DataFrame({
        'LIMIT_BAL': [10000, 20000],
        'AGE': [30, 40],
        'EDUCATION': [1, 2],
        'MARRIAGE': [1, 2]
    })
    y = pd.Series([0, 1])

    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    assert len(predictions) == 2
    assert predictions[0] in [0, 1]
