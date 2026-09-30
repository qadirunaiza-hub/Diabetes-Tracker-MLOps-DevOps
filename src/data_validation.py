"""
data_validation.py
------------------
MLOps Step 1: Data Validation

Why this matters:
  Garbage in → garbage model. Before any preprocessing or training, we
  verify that the incoming dataset meets our contract: correct columns,
  expected dtypes, acceptable value ranges, and no catastrophic missingness.
  This acts as a gate — the pipeline halts if data quality is unacceptable.
"""

import pandas as pd
from src.utils import get_logger, FEATURE_COLS, TARGET_COL

logger = get_logger(__name__)

# ── Validation contract ────────────────────────────────────────────────────── #

EXPECTED_COLUMNS = FEATURE_COLS + [TARGET_COL]

COLUMN_RANGES = {
    "Pregnancies":              (0,   17),
    "Glucose":                  (0,  200),
    "BloodPressure":            (0,  130),
    "SkinThickness":            (0,  100),
    "Insulin":                  (0,  850),
    "BMI":                      (0,   68),
    "DiabetesPedigreeFunction": (0.0,  3.0),
    "Age":                      (21,   82),
    "Outcome":                  (0,    1),
}

MAX_MISSING_FRACTION = 0.05   # flag if >5% of any column is truly NaN
MAX_ZERO_FRACTION    = 0.50   # flag if >50% of a medical column is zero-coded


def validate(df: pd.DataFrame) -> dict:
    """
    Run all validation checks. Returns a report dict with a top-level
    'passed' boolean and per-check details.
    """
    report = {"checks": {}, "passed": True}

    # 1. Schema check
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    report["checks"]["schema"] = {
        "passed": len(missing_cols) == 0,
        "missing_columns": missing_cols,
    }
    if missing_cols:
        report["passed"] = False
        logger.error(f"Schema check FAILED — missing columns: {missing_cols}")
        return report  # can't proceed without columns

    # 2. Row count
    n = len(df)
    enough_rows = n >= 100
    report["checks"]["row_count"] = {"passed": enough_rows, "n_rows": n}
    if not enough_rows:
        report["passed"] = False
        logger.warning(f"Row count check FAILED — only {n} rows")

    # 3. Null values
    null_fractions = (df[EXPECTED_COLUMNS].isnull().sum() / n).to_dict()
    null_violations = {c: v for c, v in null_fractions.items() if v > MAX_MISSING_FRACTION}
    report["checks"]["null_values"] = {
        "passed": len(null_violations) == 0,
        "null_fractions": {c: round(v, 4) for c, v in null_fractions.items()},
        "violations": null_violations,
    }
    if null_violations:
        report["passed"] = False
        logger.warning(f"Null check FAILED — high nulls in: {list(null_violations.keys())}")

    # 4. Value ranges
    range_violations = {}
    for col, (lo, hi) in COLUMN_RANGES.items():
        if col not in df.columns:
            continue
        out_of_range = df[(df[col] < lo) | (df[col] > hi)].shape[0]
        if out_of_range > 0:
            range_violations[col] = out_of_range
    report["checks"]["value_ranges"] = {
        "passed": len(range_violations) == 0,
        "violations": range_violations,
    }
    if range_violations:
        report["passed"] = False
        logger.warning(f"Range check FAILED — {range_violations}")

    # 5. Target column — binary
    unique_targets = set(df[TARGET_COL].dropna().unique())
    valid_target = unique_targets.issubset({0, 1})
    report["checks"]["target_binary"] = {
        "passed": valid_target,
        "unique_values": list(unique_targets),
    }
    if not valid_target:
        report["passed"] = False
        logger.error(f"Target check FAILED — non-binary values: {unique_targets}")

    # 6. Class balance (warn only — doesn't fail pipeline)
    class_dist = df[TARGET_COL].value_counts(normalize=True).to_dict()
    minority_frac = min(class_dist.values())
    severely_imbalanced = minority_frac < 0.10
    report["checks"]["class_balance"] = {
        "passed": True,   # informational
        "distribution": {int(k): round(v, 4) for k, v in class_dist.items()},
        "warning": "Severe imbalance detected" if severely_imbalanced else None,
    }
    if severely_imbalanced:
        logger.warning(f"Class imbalance warning — minority class is {minority_frac:.1%}")

    # 7. Zero-coded missingness in medical columns
    medical_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    zero_fractions = {c: (df[c] == 0).mean() for c in medical_cols if c in df.columns}
    zero_violations = {c: v for c, v in zero_fractions.items() if v > MAX_ZERO_FRACTION}
    report["checks"]["zero_coded_missingness"] = {
        "passed": True,   # informational — handled in preprocessing
        "zero_fractions": {c: round(v, 4) for c, v in zero_fractions.items()},
        "note": f"Columns with >50% zeros: {list(zero_violations.keys())}" if zero_violations else "OK",
    }

    status = "PASSED" if report["passed"] else "FAILED"
    logger.info(f"Data validation {status} | {n} rows | {len(EXPECTED_COLUMNS)} columns")
    return report


def print_report(report: dict) -> None:
    print("\n" + "=" * 60)
    print("  DATA VALIDATION REPORT")
    print("=" * 60)
    overall = "✓ PASSED" if report["passed"] else "✗ FAILED"
    print(f"  Overall: {overall}\n")
    for check_name, details in report["checks"].items():
        status = "✓" if details.get("passed", True) else "✗"
        print(f"  {status} {check_name.replace('_', ' ').title()}")
        for k, v in details.items():
            if k != "passed" and v is not None and v != {} and v != []:
                print(f"      {k}: {v}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import sys
    from src.utils import data_path

    csv = data_path()
    if not csv.exists():
        print(f"Dataset not found at {csv}. Run: python data/generate_data.py")
        sys.exit(1)

    df = pd.read_csv(csv)
    report = validate(df)
    print_report(report)
    sys.exit(0 if report["passed"] else 1)
