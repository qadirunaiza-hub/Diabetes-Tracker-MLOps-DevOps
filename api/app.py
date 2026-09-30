"""
app.py
------
MLOps Step 7: Flask Prediction API

Architecture:
  1. On startup, loads the Production model from the MLflow Model Registry
     (via the local mlruns directory — no remote server needed).
  2. Loads the saved scaler + imputer for consistent preprocessing.
  3. Exposes REST endpoints for prediction, health, and audit logging.
  4. Every prediction is persisted to SQLite for auditability.

Endpoints:
  GET  /            — serves the web UI
  GET  /health      — liveness check (used by Docker health check)
  GET  /model-info  — current model name, version, and stage
  POST /predict     — returns prediction + probability + risk level
  GET  /predictions — recent prediction log (last 50)
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path

import mlflow.sklearn
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

from src.utils import project_root, logs_path, FEATURE_COLS, FEATURE_DESCRIPTIONS
from src.preprocessing import load_preprocessors, preprocess_single

load_dotenv()

# ── App setup ─────────────────────────────────────────────────────────────── #

app = Flask(__name__, template_folder="templates")
CORS(app)

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{project_root() / 'mlflow.db'}")
MODEL_NAME  = os.getenv("MODEL_NAME", "DiabetesPredictor")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")  # MLflow alias (replaces deprecated stages)
DB_PATH     = logs_path() / "predictions.db"

# ── Global state (loaded once at startup) ─────────────────────────────────── #

_model       = None
_scaler      = None
_imputer     = None
_model_info  = {}


def load_model() -> None:
    global _model, _scaler, _imputer, _model_info

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # Use alias-based URI: models:/<name>@<alias>  (MLflow 2.9+ recommended pattern)
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

    logger.info(f"Loading model from registry: {model_uri}")
    _model = mlflow.sklearn.load_model(model_uri)

    from mlflow import MlflowClient
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    if mv:
        _model_info = {
            "name":    MODEL_NAME,
            "version": mv.version,
            "alias":   MODEL_ALIAS,
            "run_id":  mv.run_id,
        }

    _scaler, _imputer = load_preprocessors()
    logger.info(f"Model loaded: {MODEL_NAME} v{_model_info.get('version', '?')} ({MODEL_STAGE})")


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT NOT NULL,
            input_data    TEXT NOT NULL,
            prediction    INTEGER NOT NULL,
            probability   REAL NOT NULL,
            risk_level    TEXT NOT NULL,
            model_version TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"Prediction log DB initialised at {DB_PATH}")


def log_prediction(input_data: dict, prediction: int, probability: float,
                   risk_level: str, model_version: str) -> None:
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO predictions (timestamp, input_data, prediction, probability, risk_level, model_version) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                json.dumps(input_data),
                prediction,
                probability,
                risk_level,
                model_version,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning(f"Failed to log prediction: {exc}")


def risk_level(probability: float) -> str:
    if probability < 0.30:
        return "Low"
    if probability < 0.60:
        return "Moderate"
    return "High"


# ── Routes ────────────────────────────────────────────────────────────────── #

@app.route("/")
def index():
    return render_template(
        "index.html",
        feature_cols=FEATURE_COLS,
        feature_descriptions=FEATURE_DESCRIPTIONS,
        model_info=_model_info,
    )


@app.route("/health")
def health():
    return jsonify({
        "status": "ok" if _model is not None else "model_not_loaded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/model-info")
def model_info_endpoint():
    return jsonify(_model_info)


@app.route("/predict", methods=["POST"])
def predict():
    if _model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    # Validate all features present
    missing = [f for f in FEATURE_COLS if f not in data]
    if missing:
        return jsonify({"error": f"Missing features: {missing}"}), 400

    try:
        X = preprocess_single(data, _scaler, _imputer)
        pred  = int(_model.predict(X)[0])
        prob  = float(_model.predict_proba(X)[0][1])
        level = risk_level(prob)
        version = _model_info.get("version", "unknown")

        log_prediction(data, pred, prob, level, version)

        return jsonify({
            "prediction":        pred,
            "prediction_label":  "Diabetes Risk Detected" if pred == 1 else "No Diabetes Risk Detected",
            "probability":       round(prob, 4),
            "risk_level":        level,
            "model_version":     version,
            "disclaimer":        (
                "This is an educational ML demonstration. "
                "It is NOT a medical diagnostic tool. "
                "Consult a qualified healthcare professional for any health concerns."
            ),
        })

    except Exception as exc:
        logger.exception("Prediction error")
        return jsonify({"error": str(exc)}), 500


@app.route("/predictions")
def predictions_log():
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT timestamp, prediction, probability, risk_level, model_version, input_data "
            "FROM predictions ORDER BY id DESC LIMIT 50"
        ).fetchall()
        conn.close()
        return jsonify([
            {
                "timestamp":     r[0],
                "prediction":    r[1],
                "probability":   r[2],
                "risk_level":    r[3],
                "model_version": r[4],
                "input":         json.loads(r[5]),
            }
            for r in rows
        ])
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Startup ───────────────────────────────────────────────────────────────── #

def create_app() -> Flask:
    init_db()
    try:
        load_model()
    except Exception as exc:
        logger.warning(
            f"Could not load Production model: {exc}. "
            "Run the training pipeline first: python run_pipeline.py"
        )
    return app


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5001))
    create_app()
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_ENV") == "development")
