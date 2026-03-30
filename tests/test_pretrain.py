import os

import pandas as pd

REQUIRED_COLUMNS = {
    "MinTemp",
    "MaxTemp",
    "Rainfall",
    "Humidity3pm",
    "Pressure3pm",
    "RainTomorrow",
}


def test_raw_data_exists() -> None:
    data_path = os.getenv("DATA_PATH", "data/raw/weatherAUS.csv")
    assert os.path.exists(data_path), f"Data not found: {data_path}"


def test_data_schema_basic() -> None:
    data_path = os.getenv("DATA_PATH", "data/raw/weatherAUS.csv")
    df = pd.read_csv(data_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    assert not missing, f"Missing columns: {sorted(missing)}"
    assert df.shape[0] >= 1000, "Dataset is too small for this lab"
