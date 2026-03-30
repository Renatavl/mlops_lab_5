# Лабораторна робота №5: Оркестрація ML-пайплайнів (CI/CD + Continuous Training)

## Що реалізовано

- **ML pipeline code**: `src/prepare.py` + `src/train.py`.
- **DVC pipeline**: `dvc.yaml` зі stage `prepare` і `train`.
- **Обов'язкові артефакти тренування**:
  - `model.pkl`
  - `metrics.json`
  - `confusion_matrix.png`
- **Airflow DAG**: `dags/ml_training_pipeline.py` із кроками:
  1. Sensor/Check доступності даних + перевірка `dvc status`
  2. Data preparation
  3. Model training
  4. Evaluation & Branching (`accuracy_test >= 0.85`)
  5. Model registration у MLflow Model Registry (Staging)
- **Docker multi-stage build**: `Dockerfile`.
- **Локальний Airflow через Docker Compose**: `docker-compose.yaml`.
- **CI (GitHub Actions)**: `.github/workflows/main.yaml`:
  - lint,
  - DAG integrity test,
  - Docker build test.
- **Тести**:
  - `tests/test_pretrain.py`
  - `tests/test_posttrain.py`
  - `tests/test_dag_integrity.py`

## Структура

- `src/` — код підготовки, навчання, реєстрації моделі.
- `dags/` — Airflow DAG.
- `tests/` — pretrain/posttrain + DAG integrity.
- `config/` — Hydra-конфіг.
- `airflow/Dockerfile` — кастомний образ Airflow з потрібними залежностями.
- `docker-compose.yaml` — підняття локального Airflow.

## Передумови

- Python 3.11+
- Docker + Docker Compose
- (опційно) DVC 3+

Дані очікуються у файлі:

- `data/raw/weatherAUS.csv`

## Локальний запуск (без Airflow)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

dvc init
dvc repro prepare
dvc repro train

pytest -q tests/test_pretrain.py
F1_THRESHOLD=0.70 pytest -q tests/test_posttrain.py
```

## Запуск Airflow (Docker Compose)

```bash
docker compose up airflow-init
docker compose up -d airflow-webserver airflow-scheduler
```

Airflow UI:

- URL: `http://localhost:8080`
- Login: `admin`
- Password: `admin`

Далі в UI:

1. Увімкнути DAG `ml_training_pipeline`.
2. Запустити `Trigger DAG`.

## Що робить DAG

1. `check_data_availability` — сенсор перевіряє наявність `data/raw/weatherAUS.csv`.
2. `check_dvc_status` — перевірка стану DVC.
3. `prepare_data` — запуск `python src/prepare.py`.
4. `train_model` — запуск `python src/train.py`.
5. `evaluate_and_branch`:
   - якщо `accuracy_test >= 0.85` → `register_model`;
   - інакше → `stop_pipeline`.
6. `register_model` — логування моделі й реєстрація версії в MLflow Registry зі стадією `Staging`.

## CI (GitHub Actions)

Workflow файл: `.github/workflows/main.yaml`

Кроки:

1. Встановлення залежностей.
2. Lint (`flake8`).
3. DAG integrity (`pytest tests/test_dag_integrity.py`).
4. Docker build (`docker build ...`).

## Відтворюваність

- Фіксований `seed` у конфігу
- Логування `git_commit_hash` і `raw_data_md5` у MLflow
- DVC stage-орієнтований запуск підготовки та тренування
