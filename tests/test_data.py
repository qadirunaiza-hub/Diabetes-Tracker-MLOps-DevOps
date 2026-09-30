import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import impute_zeros
from src.features.engineering import engineer


@pytest.fixture
def raw_df():
    return pd.DataFrame({
        "Pregnancies": [2, 8, 0],
        "Glucose": [0, 120, 85],
        "BloodPressure": [70, 0, 64],
        "SkinThickness": [0, 20, 15],
        "Insulin": [0, 0, 80],
        "BMI": [33.6, 0.0, 27.4],
        "DiabetesPedigreeFunction": [0.627, 0.210, 0.351],
        "Age": [50, 33, 29],
        "Outcome": [1, 0, 0],
    })


def test_impute_zeros_removes_physiological_zeros(raw_df):
    df = impute_zeros(raw_df)
    for col in ["Glucose", "BloodPressure", "SkinThickness", "BMI"]:
        assert df[col].min() > 0, f"{col} still contains zero after imputation"


def test_engineer_adds_expected_columns(raw_df):
    df = engineer(raw_df.drop(columns=["Outcome"]))
    for col in ["BMI_Category", "Glucose_Category", "Age_Group", "Glucose_BMI"]:
        assert col in df.columns, f"Expected column '{col}' missing"


def test_engineer_bmi_category_range(raw_df):
    df = engineer(raw_df.drop(columns=["Outcome"]))
    assert df["BMI_Category"].between(0, 3).all()
