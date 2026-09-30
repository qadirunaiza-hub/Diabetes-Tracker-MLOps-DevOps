"""
train.py
--------
MLOps Steps 3, 5, 6: Model Training + Experiment Tracking + Model Registry

How MLflow works here:
  1. mlflow.set_experiment()  — creates or reuses a named experiment bucket
  2. mlflow.start_run()       — begins a run; every run gets a unique run_id
  3. mlflow.log_param()       — log hyperparameters (searchable in UI)
  4. mlflow.log_metrics()     — log scalar metrics (accuracy, f1, etc.)
  5. mlflow.log_artifact()    — log files (plots, reports)
  6. mlflow.sklearn.log_model() — save the serialised model inside the run
  7. MlflowClient.create_registered_model() — create a named model in the registry
  8. MlflowClient.create_model_version()    — link the run's model to the registry
  9. MlflowClient.transition_model_version_stage() — promote to Production

The MLflow tracking server defaults to ./mlruns (local file-based).
In production you would point MLFLOW_TRACKING_URI at a remote server.
"""

import os
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from src.utils import get_logger, project_root, models_path
from src.preprocessing import load_data, build_pipeline, save_preprocessors
from src.evaluate import full_evaluation

logger = get_logger(__name__)

EXPERIMENT_NAME = "diabetes-prediction"
MODEL_REGISTRY_NAME = os.getenv("MODEL_NAME", "DiabetesPredictor")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{project_root() / 'mlflow.db'}")

MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=1.0, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150, max_depth=8, min_samples_split=5, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42
    ),
    "SVM": SVC(
        kernel="rbf", C=1.0, probability=True, random_state=42
    ),
}

MODEL_PARAMS = {
    "Logistic Regression": {"max_iter": 1000, "C": 1.0},
    "Random Forest":       {"n_estimators": 150, "max_depth": 8, "min_samples_split": 5},
    "Gradient Boosting":   {"n_estimators": 150, "learning_rate": 0.08, "max_depth": 4},
    "SVM":                 {"kernel": "rbf", "C": 1.0},
}


def setup_mlflow() -> MlflowClient:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"Experiment: {EXPERIMENT_NAME}")
    return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def train_and_track(data: dict) -> dict:
    """
    Train all models, log each run to MLflow, return a summary keyed by model name.
    """
    client = setup_mlflow()
    plot_dir = project_root() / "docs" / "plots"
    results = {}

    for name, model in MODELS.items():
        logger.info(f"\nTraining: {name}")
        with mlflow.start_run(run_name=name) as run:
            run_id = run.info.run_id

            # --- Train ---
            model.fit(data["X_train"], data["y_train"])

            # --- Evaluate ---
            eval_result = full_evaluation(
                model, data["X_test"], data["y_test"], name, plot_dir
            )
            metrics = eval_result["metrics"]

            # --- Log to MLflow ---
            mlflow.log_params(MODEL_PARAMS[name])
            mlflow.log_param("model_type", name)
            mlflow.log_metrics(metrics)

            for artifact_path in eval_result["artifacts"]:
                mlflow.log_artifact(artifact_path)

            # Log the model itself — MLflow serialises it with the run.
            # skops_trusted_types: required for tree-based models (Random Forest,
            # Gradient Boosting) that internally use sklearn.tree._tree.Tree.
            # We created these models ourselves so trusting them is safe here.
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                registered_model_name=None,
                skops_trusted_types=[
                    "sklearn.tree._tree.Tree",
                    "sklearn.tree._classes.DecisionTreeClassifier",
                    "sklearn.ensemble._forest.RandomForestClassifier",
                    "sklearn.ensemble._gb.GradientBoostingClassifier",
                    "sklearn.linear_model._logistic.LogisticRegression",
                    "sklearn.svm._classes.SVC",
                    "numpy.core.multiarray._reconstruct",
                    "numpy.ndarray",
                    "numpy.dtype",
                ],
            )

            logger.info(
                f"  run_id={run_id[:8]}  "
                f"acc={metrics['accuracy']:.4f}  "
                f"f1={metrics['f1']:.4f}  "
                f"roc_auc={metrics['roc_auc']:.4f}"
            )

            results[name] = {
                "run_id": run_id,
                "metrics": metrics,
                "model": model,
            }

    return results


def register_best_model(results: dict, client: MlflowClient) -> str:
    """
    MLOps Step 6: Model Registry

    Picks the best run by ROC-AUC and promotes it through the registry:
      None → Staging → Production

    The registry lets you:
      - Track which version is live
      - Roll back to a previous version instantly
      - Gate promotions with approval workflows (in production setups)
    """
    best_name = max(results, key=lambda n: results[n]["metrics"]["roc_auc"])
    best      = results[best_name]
    run_id    = best["run_id"]
    auc       = best["metrics"]["roc_auc"]

    logger.info(f"\nBest model: {best_name} (ROC-AUC={auc:.4f})")

    # Build the model URI — points to the artifact logged inside the run
    model_uri = f"runs:/{run_id}/model"

    # Register in the Model Registry
    mv = mlflow.register_model(model_uri=model_uri, name=MODEL_REGISTRY_NAME)
    version = mv.version

    logger.info(f"Registered '{MODEL_REGISTRY_NAME}' version {version}")

    # Tag this version as the "champion" (active production) alias.
    # MLflow aliases replace the older stage system (stages are deprecated since 2.9).
    # The alias "champion" is our convention for "what the API should serve."
    # Rolling back is instant: set_registered_model_alias to a previous version.
    client.set_registered_model_alias(
        name=MODEL_REGISTRY_NAME,
        alias="champion",
        version=version,
    )
    logger.info(f"'{MODEL_REGISTRY_NAME}' v{version} → alias 'champion' (Production)")

    return version


def print_leaderboard(results: dict) -> None:
    print("\n" + "=" * 70)
    print("  EXPERIMENT LEADERBOARD")
    print("=" * 70)
    print(f"  {'Model':<26} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>7}")
    print("-" * 70)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["metrics"]["roc_auc"], reverse=True)
    for name, r in sorted_results:
        m = r["metrics"]
        best_marker = " ◄ BEST" if name == sorted_results[0][0] else ""
        print(
            f"  {name:<26} {m['accuracy']:>6.4f} {m['precision']:>6.4f} "
            f"{m['recall']:>6.4f} {m['f1']:>6.4f} {m['roc_auc']:>7.4f}{best_marker}"
        )
    print("=" * 70 + "\n")


def run_training_pipeline() -> None:
    logger.info("=" * 60)
    logger.info("  DIABETES TRACKER — TRAINING PIPELINE")
    logger.info("=" * 60)

    # Step 1: Load and preprocess
    df = load_data()
    data = build_pipeline(df)
    save_preprocessors(data["scaler"], data["imputer"])

    # Step 2: Train all models, log to MLflow
    results = train_and_track(data)

    # Step 3: Print leaderboard
    print_leaderboard(results)

    # Step 4: Register best model
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    version = register_best_model(results, client)

    logger.info(f"\nPipeline complete. Model '{MODEL_REGISTRY_NAME}' v{version} is in Production.")
    logger.info(f"Start the MLflow UI with: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    run_training_pipeline()
