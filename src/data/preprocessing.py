import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Columns where 0 is physiologically impossible and must be treated as missing
ZERO_AS_NULL = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def impute_zeros(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ZERO_AS_NULL:
        if col not in df.columns:
            continue
        median = df[col].replace(0, np.nan).median()
        zeros = (df[col] == 0).sum()
        df[col] = df[col].replace(0, median)
        logger.info(f"  {col}: replaced {zeros} zeros with median={median:.2f}")
    return df


def split(df: pd.DataFrame, target: str, test_size: float, seed: int):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def fit_scale(X_train: pd.DataFrame, X_test: pd.DataFrame, scaler_path: str = None):
    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_te = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    if scaler_path:
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler persisted to {scaler_path}")
    return X_tr, X_te, scaler


def run_pipeline(df: pd.DataFrame, config: dict, persist: bool = True):
    logger.info("Starting preprocessing pipeline")
    df = impute_zeros(df)

    cfg_d = config["data"]
    X_train, X_test, y_train, y_test = split(
        df, cfg_d["target_column"], cfg_d["test_size"], cfg_d["random_state"]
    )

    scaler_path = f"{config['model']['model_path']}scaler.pkl" if persist else None
    X_train, X_test, scaler = fit_scale(X_train, X_test, scaler_path)

    logger.info(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler
