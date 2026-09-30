"""
generate_data.py
----------------
Generates a synthetic dataset that mirrors the structure and statistical
properties of the Pima Indians Diabetes Dataset (UCI ML Repository).

Feature definitions:
  Pregnancies              - Number of times pregnant (integer, 0–17)
  Glucose                  - Plasma glucose concentration (mg/dL, 0–199)
  BloodPressure            - Diastolic blood pressure (mm Hg, 0–122)
  SkinThickness            - Triceps skin fold thickness (mm, 0–99)
  Insulin                  - 2-Hour serum insulin (mu U/ml, 0–846)
  BMI                      - Body mass index (kg/m², 0–67.1)
  DiabetesPedigreeFunction - Diabetes pedigree function (0.078–2.42)
  Age                      - Age in years (21–81)
  Outcome                  - 1 = diabetic, 0 = non-diabetic

NOTE: This is synthetic educational data. Do NOT use for clinical decisions.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_SAMPLES = 768  # same as original Pima dataset size


def generate_diabetes_data(n_samples: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Diabetic class (~35% prevalence, matching the original dataset)
    n_positive = int(n_samples * 0.348)
    n_negative = n_samples - n_positive

    def _clip(arr, lo, hi):
        return np.clip(arr, lo, hi)

    # --- Non-diabetic group ---
    neg = {
        "Pregnancies": rng.integers(0, 6, n_negative),
        "Glucose": _clip(rng.normal(110, 18, n_negative), 50, 165).astype(int),
        "BloodPressure": _clip(rng.normal(68, 10, n_negative), 40, 110).astype(int),
        "SkinThickness": _clip(rng.normal(24, 8, n_negative), 0, 60).astype(int),
        "Insulin": _clip(rng.exponential(60, n_negative), 0, 400).astype(int),
        "BMI": _clip(rng.normal(30.0, 5.5, n_negative), 15.0, 55.0).round(1),
        "DiabetesPedigreeFunction": _clip(rng.exponential(0.35, n_negative), 0.08, 1.5).round(3),
        "Age": _clip(rng.integers(21, 55, n_negative), 21, 81),
        "Outcome": np.zeros(n_negative, dtype=int),
    }

    # --- Diabetic group ---
    pos = {
        "Pregnancies": rng.integers(1, 10, n_positive),
        "Glucose": _clip(rng.normal(143, 22, n_positive), 80, 199).astype(int),
        "BloodPressure": _clip(rng.normal(74, 12, n_positive), 40, 122).astype(int),
        "SkinThickness": _clip(rng.normal(33, 10, n_positive), 0, 99).astype(int),
        "Insulin": _clip(rng.exponential(150, n_positive), 0, 846).astype(int),
        "BMI": _clip(rng.normal(35.5, 6.5, n_positive), 18.0, 67.1).round(1),
        "DiabetesPedigreeFunction": _clip(rng.exponential(0.55, n_positive), 0.08, 2.42).round(3),
        "Age": _clip(rng.integers(25, 75, n_positive), 21, 81),
        "Outcome": np.ones(n_positive, dtype=int),
    }

    df = pd.concat([pd.DataFrame(neg), pd.DataFrame(pos)], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Inject realistic zero-coded missingness (e.g., Glucose=0 means "not recorded")
    zero_mask_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_mask_cols:
        zero_idx = rng.choice(df.index, size=int(n_samples * 0.04), replace=False)
        df.loc[zero_idx, col] = 0

    return df


if __name__ == "__main__":
    output_path = Path(__file__).parent / "diabetes.csv"
    df = generate_diabetes_data()
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Class distribution:\n{df['Outcome'].value_counts()}")
    print(f"\nFirst 5 rows:\n{df.head()}")
