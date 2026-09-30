#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/config.yaml"
PYTHON="${PYTHON:-python}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== Diabetes Tracker — Training Pipeline ==="

log "[1/4] Downloading dataset..."
$PYTHON - <<'EOF'
from src.data.ingestion import load_config, download_pima_dataset
cfg = load_config()
download_pima_dataset(cfg["data"]["raw_path"])
EOF

log "[2/4] Validating schema..."
$PYTHON - <<'EOF'
from src.data.ingestion import load_config, load_data, validate_schema
cfg = load_config()
df = load_data(cfg["data"]["raw_path"])
validate_schema(df, cfg)
EOF

log "[3/4] Training with MLflow tracking..."
$PYTHON - <<'EOF'
from src.data.ingestion   import load_config, load_data
from src.data.preprocessing import run_pipeline
from src.features.engineering import engineer
from src.models.train import run as train_run

cfg = load_config()
df = load_data(cfg["data"]["raw_path"])
df = engineer(df)
X_train, X_test, y_train, y_test, _ = run_pipeline(df, cfg)
model, metrics, run_id = train_run(X_train, y_train, X_test, y_test, cfg)
print(f"Run ID : {run_id}")
print(f"Metrics: {metrics}")
EOF

log "[4/4] Done. MLflow UI -> http://localhost:5000"
