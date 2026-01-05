from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

# Кастомный трансформер для преобразования категориальных признаков в строки
class CategoricalToStringTransformer(BaseEstimator, TransformerMixin):
    """Преобразует категориальные признаки в строки"""
    
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                # Преобразуем в строку, но сохраняем NaN
                X[col] = X[col].astype(object).where(pd.notnull(X[col]), None)
        return X

def create_pipeline(numeric_features, categorical_features):
    """
    Создает пайплайн для обработки данных и обучения модели.
    
    Parameters
    ----------
    numeric_features : list
        Список числовых признаков
    categorical_features : list
        Список категориальных признаков
    
    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Обученный пайплайн
    """
    
    # Для числовых признаков
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Для категориальных признаков
    categorical_transformer = Pipeline(steps=[
        # Сначала преобразуем в строки
        ('to_string', CategoricalToStringTransformer(categorical_features)),
        # Затем импутируем
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Комбинируем трансформеры
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Полный пайплайн
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ))
    ])
    
    return pipeline