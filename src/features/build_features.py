# src/features/build_features.py

import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

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

def build_features():
    """Загружает train/test, применяет FE, сохраняет обратно."""
    train_path = os.path.join("data", "processed", "train.csv")
    test_path = os.path.join("data", "processed", "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df_fe = create_features(train_df)
    test_df_fe = create_features(test_df)

    # Пересохраняем файлы с новыми признаками
    train_df_fe.to_csv(train_path, index=False)
    test_df_fe.to_csv(test_path, index=False)

    print(f"Features built and saved to {train_path} and {test_path}")

if __name__ == "__main__":
    build_features()