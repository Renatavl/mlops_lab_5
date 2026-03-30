from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.sensors.python import PythonSensor
from mlflow.tracking import MlflowClient

PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/opt/airflow/project"))
RAW_DATA_PATH = PROJECT_DIR / "data" / "raw" / "weatherAUS.csv"
METRICS_PATH = PROJECT_DIR / "metrics.json"
MODEL_PATH = PROJECT_DIR / "model.pkl"
TRAIN_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "MLOps_Lab5_Airflow")
REGISTRY_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_REGISTRY_EXPERIMENT_NAME", "MLOps_Lab5_Registry"
)


def is_data_available() -> bool:
    return RAW_DATA_PATH.exists()


def choose_branch(**kwargs) -> str:
    threshold = float(os.getenv("ACCURACY_THRESHOLD", "0.85"))
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    accuracy = float(metrics.get("accuracy_test", 0.0))
    kwargs["ti"].xcom_push(key="accuracy_test", value=accuracy)
    if accuracy >= threshold:
        return "register_model"
    return "stop_pipeline"


def register_model_task() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", (PROJECT_DIR / "mlruns").resolve().as_uri())
    model_name = os.getenv("MLFLOW_MODEL_NAME", "WeatherPredictionModel")
    target_stage = os.getenv("MLFLOW_TARGET_STAGE", "Staging")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(REGISTRY_EXPERIMENT_NAME)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    model = joblib.load(MODEL_PATH)

    with mlflow.start_run(run_name="airflow_registration") as run:
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        mlflow.sklearn.log_model(model, artifact_path="model")
        model_uri = f"runs:/{run.info.run_id}/model"

    version = mlflow.register_model(model_uri=model_uri, name=model_name)
    client = MlflowClient(tracking_uri=tracking_uri)
    client.transition_model_version_stage(
        name=model_name,
        version=version.version,
        stage=target_stage,
        archive_existing_versions=True,
    )


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
}

with DAG(
    dag_id="ml_training_pipeline",
    description="MLOps Lab 5 DAG: DVC prepare/train + branching + MLflow registration",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args=default_args,
    tags=["mlops", "ct", "lab5"],
) as dag:
    check_data = PythonSensor(
        task_id="check_data_availability",
        python_callable=is_data_available,
        poke_interval=30,
        timeout=300,
        mode="reschedule",
    )

    check_dvc_status = BashOperator(
        task_id="check_dvc_status",
        bash_command=f"cd {PROJECT_DIR} && dvc status || true",
    )

    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "python3 src/prepare.py "
            "mlflow.tracking_uri=${MLFLOW_TRACKING_URI:-None}"
        ),
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "python3 src/train.py "
            "mlflow.tracking_uri=${MLFLOW_TRACKING_URI:-None} "
            f"mlflow.experiment_name={TRAIN_EXPERIMENT_NAME}"
        ),
    )

    evaluate_and_branch = BranchPythonOperator(
        task_id="evaluate_and_branch",
        python_callable=choose_branch,
    )

    register_model = PythonOperator(
        task_id="register_model",
        python_callable=register_model_task,
    )

    stop_pipeline = EmptyOperator(task_id="stop_pipeline")
    finish = EmptyOperator(task_id="finish", trigger_rule="none_failed_min_one_success")

    check_data >> check_dvc_status >> prepare_data >> train_model >> evaluate_and_branch
    evaluate_and_branch >> register_model >> finish
    evaluate_and_branch >> stop_pipeline >> finish
