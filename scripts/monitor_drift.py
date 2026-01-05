# scripts/monitor_drift.py

import pandas as pd
import numpy as np
from scipy import stats
import joblib
import os

def calculate_psi(expected_array, actual_array, buckets=10):
    """Рассчитывает PSI (Population Stability Index)."""
    def get_bucket_boundaries(array, buckets):
        return np.histogram(array, bins=buckets)[1]

    def calculate_proportions(array, boundaries):
        counts, _ = np.histogram(array, bins=boundaries)
        # Добавим небольшое значение, чтобы избежать деления на 0
        counts = counts.astype(float)
        counts[counts == 0] = 0.0001
        return counts / counts.sum()

    bucket_boundaries = get_bucket_boundaries(expected_array, buckets)

    expected_proportions = calculate_proportions(expected_array, bucket_boundaries)
    actual_proportions = calculate_proportions(actual_array, bucket_boundaries)

    psi_values = (actual_proportions - expected_proportions) * np.log(actual_proportions / expected_proportions)
    psi = np.sum(psi_values)
    return psi


def monitor_drift():
    """Имитирует поступление новых данных и рассчитывает PSI."""
    # Загрузка тренировочных данных (после FE)
    train_path = "data/processed/train.csv"
    train_df = pd.read_csv(train_path)

    # Имитация новых данных (берём часть тестовой выборки)
    test_path = "data/processed/test.csv"
    test_df = pd.read_csv(test_path)

    # Применяем Feature Engineering к тестовым данным
    # Копируем функцию из train.py или импортируем
    def create_features(df):
        df = df.copy()
        df['utilization'] = df['BILL_AMT1'] / (df['LIMIT_BAL'] + 1)
        df['utilization'] = df['utilization'].replace([np.inf, -np.inf], np.nan)
        df['pay_ratio'] = df['PAY_AMT1'] / (df['BILL_AMT1'] + 1)
        df['pay_ratio'] = df['pay_ratio'].replace([np.inf, -np.inf], np.nan)
        bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
        df['bill_trend'] = df[bill_cols].diff(axis=1).mean(axis=1, skipna=True).fillna(0)
        df['age_group'] = pd.cut(df['AGE'], bins=[0, 25, 35, 50, 100], labels=['Young', 'Adult', 'Senior', 'Elder'], right=False)
        return df

    test_df_fe = create_features(test_df)

    # Выберем признаки для анализа дрифта
    numeric_features = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'utilization', 'pay_ratio', 'bill_trend']
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE', 'age_group', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

    # Рассчитаем PSI для числовых признаков
    print("PSI for numeric features:")
    for feature in numeric_features:
        if feature in train_df.columns and feature in test_df_fe.columns:
            psi = calculate_psi(train_df[feature].dropna(), test_df_fe[feature].dropna())
            print(f"  {feature}: {psi:.4f}")
        else:
            print(f"  {feature}: Feature not found in both datasets.")

    # Для категориальных можно использовать Chi-Square тест или PSI
    print("\nChi-square test p-values for categorical features:")
    for feature in categorical_features:
        if feature in train_df.columns and feature in test_df_fe.columns:
            # Удалим строки с NaN, если есть
            train_cat = train_df[feature].dropna()
            test_cat = test_df_fe[feature].dropna()
            observed = pd.crosstab(train_cat, test_cat)
            if observed.shape[0] > 1 and observed.shape[1] > 1: # Проверим, что есть смысл в тесте
                chi2, p_value, dof, expected = stats.chi2_contingency(observed)
                print(f"  {feature}: p-value = {p_value:.4f}")
            else:
                print(f"  {feature}: Cannot perform Chi-square test (insufficient categories).")
        else:
            print(f"  {feature}: Feature not found in both datasets.")

if __name__ == "__main__":
    monitor_drift()