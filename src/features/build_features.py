# Файл для построения признаков
import pandas as pd
import numpy as np

def create_features(df):
    """Создает новые признаки из исходных данных"""
    # Пример создания новых признаков
    if 'BILL_AMT1' in df.columns and 'PAY_AMT1' in df.columns:
        df['BILL_PAY_RATIO'] = df['PAY_AMT1'] / (df['BILL_AMT1'] + 1)
    
    if 'LIMIT_BAL' in df.columns:
        df['LIMIT_BAL_LOG'] = np.log(df['LIMIT_BAL'] + 1)
    
    return df
