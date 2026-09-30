import logging

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred
    )
    return {
        "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score":  round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc":   round(float(roc_auc_score(y_test, y_prob)), 4),
    }


def print_report(model, X_test, y_test) -> None:
    y_pred = model.predict(X_test)
    logger.info(
        "\n" + classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"])
    )
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
