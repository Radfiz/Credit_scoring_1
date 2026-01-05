# Credit Scoring Model Pipeline

## Описание

Этот проект представляет собой автоматизированный пайплайн для разработки, тестирования, развёртывания и мониторинга модели машинного обучения, предназначенной для предсказания вероятности дефолта клиента (PD-модель) в области кредитного скоринга. Проект реализует сквозной (end-to-end) процесс, включая управление версиями кода и данных, валидацию данных, экспериментирование с моделями, логирование, тестирование, контейнеризацию и мониторинг.

**Домен:** Финансы / Кредитный скоринг
**Цель модели:** Предсказание вероятности дефолта клиента.

## Стек технологий

*   **Язык программирования:** Python
*   **Машинное обучение:** Scikit-learn
*   **Автоматизация пайплайна:** DVC (Data Version Control)
*   **Управление экспериментами:** MLflow
*   **Валидация данных:** Great Expectations
*   **Тестирование:** pytest
*   **Контейнеризация:** Docker
*   **API:** FastAPI
*   **CI/CD:** GitHub Actions
*   **Визуализация:** Matplotlib, Seaborn (для анализа и ROC-кривой)
*   **Управление зависимостями:** pip, requirements.txt

## Структура проекта
```
├── Dockerfile                       # Файл для сборки Docker-образа API
├── README.md                        # Этот файл
├── dvc.yaml                         # Определение DVC пайплайна (prepare, validate, train)
├── dvc.lock                         # Файл блокировок DVC (генерируется, отслеживается в git)
├── pyproject.toml                   # Конфигурация для setuptools, pytest, black и т.д.
├── requirements.txt                 # Список зависимостей Python
├── .gitignore                       # Правила игнорирования файлов для Git
├── .github/                         # Конфигурации GitHub Actions
│ └── workflows/
│ └── ci-cd.yml                      # Файл определения CI/CD пайплайна
├── great_expectations/              # Конфигурации и наборы ожиданий Great Expectations
│ ├── great_expectations.yml         # Основной файл конфигурации GE
│ └── expectations/
│ └── credit_data_suite.json         # Набор ожиданий для валидации данных
├── scripts/
│ └── monitor_drift.py               # Скрипт для мониторинга дрифта данных (PSI)
├── src/                             # Исходный код проекта
│ ├── api/
│ │ └── app.py                       # FastAPI приложение с endpoint /predict
│ ├── data/
│ │ ├── make_dataset.py              # Скрипт для подготовки и разделения данных
│ │ └── run_validation.py            # Скрипт для запуска валидации данных с GE
│ ├── features/
│ │ └── build_features.py            # Скрипт/модуль для Feature Engineering
│ ├── models/
│ │ ├── pipeline.py                  # Определение Sklearn Pipeline
│ │ ├── train.py                     # Скрипт для обучения модели с подбором гиперпараметров
│ │ └── predict.py                   # (Опционально) Скрипт для предсказания вне API
├── tests/                           # Unit-тесты
│ ├── test_api.py                    # Тесты для API
│ └── test_pipeline.py               # Тесты для пайплайна и его компонентов
├── data/                            # Папка для данных (управляемая DVC)
│ ├── raw/                           # Исходные данные
│ └── processed/                     # Обработанные данные
└── models/                          # Папка для обученных моделей
└── credit_default_model.pkl         # Обученная модель
```

# Установка и запуск

### Требования

*   Python 3.11
*   Git
*   Conda или venv (рекомендуется)
*   Docker Desktop
*   DVC (устанавливается через pip)

### 1. Клонирование репозитория

```bash
git clone https://github.com/<ваш_логин>/Credit_scoring_1.git
cd Credit_scoring_1
``

2. Создание и активация виртуального окружения (рекомендуется через conda)
```bash
conda create -n credit-scoring python=3.11
conda activate credit-scoring
```

3. Установка зависимостей
```bash
pip install -r requirements.txt
```

4. Инициализация DVC (если ещё не инициализирован в репозитории)
```bash
dvc init
```
5. Запуск DVC пайплайна
```bash
dvc repro
```
6. Запуск тестов
```bash
pytest tests/ -v
```
7. Сборка и запуск Docker-образа API
```bash
docker build -t credit-scoring-api:latest .
docker run -p 8000:8000 credit-scoring-api:latest
```
8. Запуск скрипта мониторинга дрифта
```bash
python scripts/monitor_drift.py
```
9. Запуск MLflow UI
```bash
mlflow ui
```
Открыть http://127.0.0.1:5000 в браузере.

# CI/CD
Проект настроен на использование GitHub Actions. Файл .github/workflows/ci-cd.yml определяет пайплайн, который автоматически запускает тесты, линтинг, форматирование и валидацию данных при каждом пуше в репозиторий.














