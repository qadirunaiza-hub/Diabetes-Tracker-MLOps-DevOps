import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .evaluate import compute_metrics

logger = logging.getLogger(__name__)

_ALGORITHMS = {
    "xgboost":            xgb.XGBClassifier,
    "random_forest":      RandomForestClassifier,
    "logistic_regression": LogisticRegression,
}


def _build_model(config: dict):
    algo = config["model"]["algorithm"]
    cls = _ALGORITHMS.get(algo)
    if cls is None:
        raise ValueError(f"Unknown algorithm '{algo}'. Valid: {list(_ALGORITHMS)}")
    return cls(**config["model"]["params"])


def run(X_train, y_train, X_test, y_test, config: dict):
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run() as run:
        model = _build_model(config)

        mlflow.log_param("algorithm", config["model"]["algorithm"])
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_params(config["model"]["params"])

        logger.info(f"Training {config['model']['algorithm']}...")
        model.fit(X_train, y_train)

        metrics = compute_metrics(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        logger.info(f"Metrics: {metrics}")

        model_name = config["mlflow"]["model_name"]
        if config["model"]["algorithm"] == "xgboost":
            mlflow.xgboost.log_model(
                model, artifact_path="model", registered_model_name=model_name
            )
        else:
            mlflow.sklearn.log_model(
                model, artifact_path="model", registered_model_name=model_name
            )

        logger.info(f"Run ID: {run.info.run_id}")
        return model, metrics, run.info.run_id
