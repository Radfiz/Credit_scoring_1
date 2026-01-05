# src/data/run_validation.py

import pandas as pd
import great_expectations as ge
from great_expectations.data_context import DataContext
from great_expectations.core.batch import RuntimeBatchRequest

def run_validation():
    """Загружает train.csv и запускает валидацию с помощью GE."""
    # Загрузка данных
    df = pd.read_csv("data/processed/train.csv")

    # Загрузка контекста GE
    context = DataContext()

    # Загрузка suite
    suite_name = "credit_data_suite"
    suite = context.get_expectation_suite(suite_name)

    # --- ИСПРАВЛЕНИЕ СОЗДАНИЯ VALIDATOR ---
    # Создание RuntimeBatchRequest для DataFrame
    # Используем имя datasource и data_connector из great_expectations.yml
    batch_request = RuntimeBatchRequest(
        datasource_name="my_datasource", # Имя из great_expectations.yml
        data_connector_name="default_runtime_data_connector_name", # Имя из great_expectations.yml
        data_asset_name="my_runtime_asset_name", # Имя ассета из great_expectations.yml
        runtime_parameters={"batch_data": df}, # Передаем сам DataFrame
        batch_identifiers={"runtime_batch_identifier_name": "validation_batch"} # batch_identifiers из ассета
    )

    # Создание валидатора через контекст с RuntimeBatchRequest
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite=suite
    )
    # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

    # Запуск валидации
    results = validator.validate()

    print(f"Validation Results: {results.success}")
    if not results.success:
        print("Validation Failed!")
        for result in results.results:
            if not result.success:
                print(f"  Failed expectation: {result.expectation_config}")
        # Важно: вызвать exit(1), чтобы DVC понял, что стадия провалилась
        exit(1)
    else:
        print("Validation Passed!")

if __name__ == "__main__":
    run_validation()