import logging
import os

import joblib
import mlflow.pyfunc
import pandas as pd
import yaml
from flask import Flask, jsonify, request
from flask_cors import CORS

from src.monitoring.verve_integration import VerveMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

with open("configs/config.yaml", "r") as _f:
    CONFIG = yaml.safe_load(_f)

FEATURES = CONFIG["data"]["feature_columns"]
monitor = VerveMonitor(CONFIG)

_model = None
_scaler = None


def _load_model() -> None:
    global _model, _scaler
    mlflow.set_tracking_uri(CONFIG["mlflow"]["tracking_uri"])
    logger.info(f"Loading model from {CONFIG['api']['model_uri']}")
    _model = mlflow.pyfunc.load_model(CONFIG["api"]["model_uri"])
    scaler_path = f"{CONFIG['model']['model_path']}scaler.pkl"
    if os.path.exists(scaler_path):
        _scaler = joblib.load(scaler_path)
    logger.info("Model ready")


def _get_model():
    if _model is None:
        _load_model()
    return _model


# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": _model is not None})


@app.route("/features", methods=["GET"])
def get_features():
    return jsonify({"feature_columns": FEATURES})


@app.route("/predict", methods=["POST"])
def predict_batch():
    try:
        payload = request.get_json(force=True)
        records = payload.get("features")
        if not records:
            return jsonify({"error": "Missing 'features' key"}), 400

        if isinstance(records, dict):
            records = [records]

        df = pd.DataFrame(records)[FEATURES]
        if _scaler is not None:
            df = pd.DataFrame(_scaler.transform(df), columns=FEATURES)

        model = _get_model()
        preds = model.predict(df)
        probs = (
            model.predict_proba(df)[:, 1].tolist()
            if hasattr(model, "predict_proba")
            else None
        )
        return jsonify({
            "predictions": preds.tolist() if hasattr(preds, "tolist") else list(preds),
            "probabilities": probs,
            "n_samples": len(df),
        })

    except Exception as exc:
        logger.exception("Batch prediction error")
        return jsonify({"error": str(exc)}), 500


@app.route("/predict/single", methods=["POST"])
def predict_single():
    try:
        payload = request.get_json(force=True)
        patient_id = payload.pop("patient_id", "unknown")

        df = pd.DataFrame([payload])[FEATURES]
        if _scaler is not None:
            df = pd.DataFrame(_scaler.transform(df), columns=FEATURES)

        model = _get_model()
        prediction = int(model.predict(df)[0])
        probability = (
            float(model.predict_proba(df)[0, 1])
            if hasattr(model, "predict_proba")
            else float(prediction)
        )

        risk = (
            "High" if probability >= CONFIG["monitoring"]["alert_threshold"]
            else "Medium" if probability >= 0.4
            else "Low"
        )

        monitor.log_prediction(patient_id, payload, prediction, probability)

        return jsonify({
            "patient_id": patient_id,
            "prediction": prediction,
            "probability": probability,
            "risk_level": risk,
            "label": "Diabetic" if prediction == 1 else "Non-Diabetic",
        })

    except Exception as exc:
        logger.exception("Single prediction error")
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _load_model()
    app.run(
        host=CONFIG["api"]["host"],
        port=CONFIG["api"]["port"],
        debug=CONFIG["api"]["debug"],
    )
