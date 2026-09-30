import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

PIMA_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]
PIMA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(data_path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} records, {len(df.columns)} columns")
    return df


def validate_schema(df: pd.DataFrame, config: dict) -> None:
    required = config["data"]["feature_columns"] + [config["data"]["target_column"]]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    nulls = df.isnull().sum()
    if nulls.any():
        logger.warning(f"Null values detected:\n{nulls[nulls > 0]}")
    logger.info("Schema validation passed")


def download_pima_dataset(output_path: str) -> None:
    logger.info(f"Downloading Pima dataset to {output_path}")
    df = pd.read_csv(PIMA_URL, names=PIMA_COLUMNS, header=None)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} records")
