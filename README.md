# Credit Scoring Model

Проект для предсказания дефолта по кредитным картам на основе датасета UCI Credit Card.

## Структура проекта
credit-scoring-model/
├── data/ # Данные
│ ├── raw/ # Исходные данные
│ └── processed/ # Обработанные данные
├── models/ # Сохраненные модели
├── notebooks/ # Jupyter ноутбуки для анализа
├── src/ # Исходный код
│ ├── data/ # Загрузка и обработка данных
│ ├── features/ # Построение признаков
│ ├── models/ # Модели и обучение
│ └── api/ # API для предсказаний
├── tests/ # Тесты
├── scripts/ # Вспомогательные скрипты
└── .github/ # CI/CD конфигурация
## Установка

1. Клонируйте репозиторий:
git clone <repository-url>
cd credit-scoring-model

2. Установите зависимости:
pip install -r requirements.txt

3. Установите DVC (опционально, для управления данными):
pip install dvc

## Использование

### Подготовка данных
python src/data/make_dataset.py

### Обучение модели
python src/models/train.py

### Запуск API
uvicorn src.api.app:app --reload

### Запуск тестов
pytest tests/

## API Endpoints

- `GET /` - Проверка работоспособности
- `GET /health` - Проверка здоровья
- `POST /predict` - Предсказание дефолта

Пример запроса:
{
"LIMIT_BAL": 20000.0,
"AGE": 30,
"BILL_AMT1": 1000.0,
"PAY_AMT1": 500.0,
"EDUCATION": 2,
"MARRIAGE": 1,
"PAY_0": 0
}
