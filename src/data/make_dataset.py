# src/data/make_dataset.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_and_split_data():
    """Загружает данные и разделяет их на тренировочную и тестовую выборки."""
    data_path = os.path.join("data", "raw", "UCI_Credit_Card.csv")
    df = pd.read_csv(data_path)

    # Разделение данных
    X = df.drop(columns=["default.payment.next.month"])
    y = df["default.payment.next.month"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Сохраняем
    os.makedirs("data/processed", exist_ok=True)
    train_path = os.path.join("data", "processed", "train.csv")
    test_path = os.path.join("data", "processed", "test.csv")

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train data saved to {train_path}")
    print(f"Test data saved to {test_path}")

if __name__ == "__main__":
    load_and_split_data()