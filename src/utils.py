"""
utils.py — shared helpers used across the pipeline.
"""

import logging
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def get_logger(name: str) -> logging.Logger:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level, logging.INFO),
    )
    return logging.getLogger(name)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_path(filename: str = "diabetes.csv") -> Path:
    return project_root() / "data" / filename


def models_path() -> Path:
    path = project_root() / "models"
    path.mkdir(exist_ok=True)
    return path


def logs_path() -> Path:
    path = project_root() / "logs"
    path.mkdir(exist_ok=True)
    return path


FEATURE_COLS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

TARGET_COL = "Outcome"

FEATURE_DESCRIPTIONS = {
    "Pregnancies": "Number of times pregnant (0–17)",
    "Glucose": "Plasma glucose concentration in mg/dL (0–199)",
    "BloodPressure": "Diastolic blood pressure in mm Hg (0–122)",
    "SkinThickness": "Triceps skin fold thickness in mm (0–99)",
    "Insulin": "2-hour serum insulin in mu U/ml (0–846)",
    "BMI": "Body mass index in kg/m² (0–67.1)",
    "DiabetesPedigreeFunction": "Diabetes pedigree function (0.078–2.42)",
    "Age": "Age in years (21–81)",
}
