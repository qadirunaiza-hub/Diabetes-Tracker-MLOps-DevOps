"""
evaluate.py
-----------
MLOps Step 4: Model Evaluation

Computes a full classification report and generates diagnostic plots:
  - Confusion matrix
  - ROC curve
  - Feature importance (where available)

These artifacts are logged to MLflow so every experiment run has a
complete, reproducible audit trail — not just numbers, but visuals.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe in Docker/CI)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, classification_report,
)

from src.utils import get_logger, FEATURE_COLS

logger = get_logger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Returns a flat dict of evaluation metrics — structured for MLflow logging.

    Metrics explained:
      accuracy   — fraction of correct predictions (misleading on imbalanced data)
      precision  — of predicted positives, how many are truly positive
      recall     — of actual positives, how many did we catch (sensitivity)
      f1         — harmonic mean of precision and recall
      roc_auc    — area under the ROC curve; 0.5 = random, 1.0 = perfect
    """
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
    }


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          model_name: str, output_dir: Path) -> Path:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["No Diabetes", "Diabetes"],
        yticklabels=["No Diabetes", "Diabetes"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    out = output_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info(f"Confusion matrix saved: {out}")
    return out


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray,
                   model_name: str, output_dir: Path) -> Path:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = output_dir / f"roc_curve_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info(f"ROC curve saved: {out}")
    return out


def plot_feature_importance(model, model_name: str, output_dir: Path):
    """
    Only applicable to tree-based models that expose feature_importances_.
    Returns None if the model doesn't support it.
    """
    if not hasattr(model, "feature_importances_"):
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = [FEATURE_COLS[i] for i in indices]
    values   = importances[indices]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(features[::-1], values[::-1], color="#4C72B0")
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance — {model_name}")
    fig.tight_layout()
    out = output_dir / f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info(f"Feature importance saved: {out}")
    return out


def full_evaluation(model, X_test: np.ndarray, y_test: np.ndarray,
                    model_name: str, output_dir: Path) -> dict:
    """
    Run full evaluation suite for one model. Returns:
      metrics — dict of scalar metrics
      artifacts — list of paths to generated plot files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)

    artifacts = []
    artifacts.append(plot_confusion_matrix(y_test, y_pred, model_name, output_dir))
    artifacts.append(plot_roc_curve(y_test, y_prob, model_name, output_dir))
    fi_path = plot_feature_importance(model, model_name, output_dir)
    if fi_path:
        artifacts.append(fi_path)

    print(f"\n  {model_name} — Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")

    return {"metrics": metrics, "artifacts": [str(a) for a in artifacts]}
