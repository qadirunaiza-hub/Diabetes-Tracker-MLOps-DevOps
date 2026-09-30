"""
run_pipeline.py
---------------
Top-level pipeline orchestrator.

Runs the full MLOps workflow in order:
  1. Generate dataset
  2. Validate data
  3. Train all models + track with MLflow
  4. Register best model to MLflow Registry (Production)

After this script completes, start the API:
  python -m api.app
Or via Docker:
  docker compose up
"""

import sys
from pathlib import Path

# Make src importable when run from project root
sys.path.insert(0, str(Path(__file__).parent))

from data.generate_data import generate_diabetes_data
from src.data_validation import validate, print_report
from src.train import run_training_pipeline
from src.utils import get_logger, data_path

logger = get_logger("pipeline")


def main() -> None:
    logger.info("=" * 60)
    logger.info("  DIABETES TRACKER — FULL MLOps PIPELINE")
    logger.info("=" * 60)

    # ── Step 1: Data Generation ──────────────────────────────── #
    csv = data_path()
    if csv.exists():
        logger.info(f"Dataset already exists at {csv} — skipping generation.")
    else:
        logger.info("Generating synthetic diabetes dataset...")
        df = generate_diabetes_data()
        df.to_csv(csv, index=False)
        logger.info(f"Dataset saved: {csv} ({len(df)} rows)")

    import pandas as pd
    df = pd.read_csv(csv)

    # ── Step 2: Data Validation ──────────────────────────────── #
    logger.info("\nRunning data validation...")
    report = validate(df)
    print_report(report)

    if not report["passed"]:
        logger.error("Data validation FAILED. Fix the data before proceeding.")
        sys.exit(1)

    # ── Steps 3–6: Train, Track, Register ────────────────────── #
    logger.info("Starting training pipeline...\n")
    run_training_pipeline()

    logger.info("\n" + "=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info("  View experiments: mlflow ui --backend-store-uri sqlite:///mlflow.db")
    logger.info("  Start the API:    python -m api.app")
    logger.info("  Or via Docker:    docker compose up")


if __name__ == "__main__":
    main()
