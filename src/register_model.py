from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


def register_model(
    model_path: Path,
    metrics_path: Path,
    tracking_uri: str,
    model_name: str,
    stage: str,
) -> int:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("MLOps_Lab5_Registry")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    model = joblib.load(model_path)

    with mlflow.start_run(run_name="airflow_registration") as run:
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        mlflow.sklearn.log_model(model, artifact_path="model")
        model_uri = f"runs:/{run.info.run_id}/model"

    version = mlflow.register_model(model_uri=model_uri, name=model_name)
    client = MlflowClient(tracking_uri=tracking_uri)
    client.transition_model_version_stage(
        name=model_name,
        version=version.version,
        stage=stage,
        archive_existing_versions=True,
    )
    return int(version.version)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="model.pkl")
    parser.add_argument("--metrics-path", default="metrics.json")
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--model-name", default="WeatherPredictionModel")
    parser.add_argument("--stage", default="Staging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    version = register_model(
        model_path=Path(args.model_path),
        metrics_path=Path(args.metrics_path),
        tracking_uri=args.tracking_uri,
        model_name=args.model_name,
        stage=args.stage,
    )
    print(f"Registered model version: {version}")


if __name__ == "__main__":
    main()
