"""
preprocessing.py
----------------
MLOps Step 2: Data Preprocessing

Why this matters:
  Raw data is rarely model-ready. We handle zero-coded missingness (zeros
  in medical fields that actually mean "not measured"), scale features to
  the same numeric range so distance-based models converge properly, and
  split data into train/test sets in a reproducible way.

  The fitted scaler is saved alongside the model so inference always uses
  the same transformation the model was trained on — a common MLOps gotcha
  called "training-serving skew."
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.utils import get_logger, FEATURE_COLS, TARGET_COL, models_path, data_path

logger = get_logger(__name__)

ZERO_AS_NAN_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_data(csv_path=None) -> pd.DataFrame:
    path = csv_path or data_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: python data/generate_data.py"
        )
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows from {path}")
    return df


def replace_zero_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    In medical datasets zeros often encode missing values (e.g., Glucose=0
    is physiologically impossible). Replace with NaN so the imputer handles
    them explicitly rather than biasing the model toward zero.
    """
    df = df.copy()
    for col in ZERO_AS_NAN_COLS:
        if col in df.columns:
            n_zeros = (df[col] == 0).sum()
            if n_zeros > 0:
                logger.info(f"  {col}: replaced {n_zeros} zeros with NaN")
                df.loc[df[col] == 0, col] = np.nan
    return df


def build_pipeline(df: pd.DataFrame) -> dict:
    """
    Full preprocessing pipeline. Returns a dict with:
      X_train, X_test, y_train, y_test  — numpy arrays
      feature_names                      — column order (important for inference)
      scaler                             — fitted StandardScaler
      imputer                            — fitted SimpleImputer
    """
    df = replace_zero_with_nan(df)

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].values

    # --- Train / test split (stratified to preserve class ratio) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Split: {len(X_train)} train | {len(X_test)} test")

    # --- Impute missing values with column median (robust to outliers) ---
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp  = imputer.transform(X_test)

    # --- Scale features to zero mean, unit variance ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled  = scaler.transform(X_test_imp)

    logger.info("Preprocessing complete: imputation + standard scaling applied")

    return {
        "X_train": X_train_scaled,
        "X_test":  X_test_scaled,
        "y_train": y_train,
        "y_test":  y_test,
        "feature_names": FEATURE_COLS,
        "scaler": scaler,
        "imputer": imputer,
    }


def save_preprocessors(scaler, imputer) -> None:
    """
    Persist the fitted preprocessors. The API must load these exact objects
    at inference time — otherwise predictions will be on a different scale
    than what the model was trained on (training-serving skew).
    """
    out = models_path()
    joblib.dump(scaler, out / "scaler.joblib")
    joblib.dump(imputer, out / "imputer.joblib")
    logger.info(f"Scaler and imputer saved to {out}/")


def load_preprocessors():
    out = models_path()
    scaler  = joblib.load(out / "scaler.joblib")
    imputer = joblib.load(out / "imputer.joblib")
    return scaler, imputer


def preprocess_single(raw_input: dict, scaler: StandardScaler, imputer: SimpleImputer) -> np.ndarray:
    """
    Transform a single inference request dict into a scaled feature vector.
    Used by the Flask API for online inference.
    """
    df = pd.DataFrame([raw_input])[FEATURE_COLS]
    df = replace_zero_with_nan(df)
    imputed = imputer.transform(df)
    scaled  = scaler.transform(imputed)
    return scaled


if __name__ == "__main__":
    df = load_data()
    result = build_pipeline(df)
    save_preprocessors(result["scaler"], result["imputer"])
    print(f"X_train shape: {result['X_train'].shape}")
    print(f"X_test  shape: {result['X_test'].shape}")
    print(f"Class balance (train): {np.bincount(result['y_train'])}")
