from pathlib import Path

import pytest

try:
    from airflow.models import DagBag

    AIRFLOW_AVAILABLE = True
except ModuleNotFoundError:
    AIRFLOW_AVAILABLE = False


@pytest.mark.skipif(not AIRFLOW_AVAILABLE, reason="apache-airflow is not installed")
def test_dag_import_has_no_errors() -> None:
    dags_path = Path(__file__).resolve().parents[1] / "dags"
    dag_bag = DagBag(dag_folder=str(dags_path), include_examples=False)
    assert dag_bag.import_errors == {}, f"DAG import errors: {dag_bag.import_errors}"


@pytest.mark.skipif(not AIRFLOW_AVAILABLE, reason="apache-airflow is not installed")
def test_main_dag_exists() -> None:
    dags_path = Path(__file__).resolve().parents[1] / "dags"
    dag_bag = DagBag(dag_folder=str(dags_path), include_examples=False)
    assert "ml_training_pipeline" in dag_bag.dags
