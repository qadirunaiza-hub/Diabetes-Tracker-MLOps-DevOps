import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _bmi_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["BMI_Category"] = pd.cut(
        df["BMI"], bins=[0, 18.5, 25.0, 30.0, np.inf], labels=[0, 1, 2, 3]
    ).astype(int)
    return df


def _glucose_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Glucose_Category"] = pd.cut(
        df["Glucose"], bins=[0, 70, 100, 125, np.inf], labels=[0, 1, 2, 3]
    ).astype(int)
    return df


def _age_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Age_Group"] = pd.cut(
        df["Age"], bins=[0, 30, 45, 60, np.inf], labels=[0, 1, 2, 3]
    ).astype(int)
    return df


def _interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Glucose_BMI"] = df["Glucose"] * df["BMI"]
    df["Insulin_Glucose_Ratio"] = df["Insulin"] / (df["Glucose"] + 1e-6)
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = _bmi_category(df)
    df = _glucose_category(df)
    df = _age_group(df)
    df = _interaction_features(df)
    logger.info(f"Feature engineering complete: {df.shape[1]} columns")
    return df
