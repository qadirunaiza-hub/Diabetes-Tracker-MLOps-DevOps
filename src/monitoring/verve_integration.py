import logging
import os
from datetime import datetime, timezone
from typing import Any

import requests

logger = logging.getLogger(__name__)


class VerveMonitor:
    """
    Pushes prediction and model-health events to the Verve real-time platform.
    High-risk predictions are flushed immediately; low-risk events are batched.
    """

    def __init__(self, config: dict):
        vcfg = config.get("monitoring", {}).get("verve", {})
        self.enabled = vcfg.get("enabled", False)
        self.endpoint = vcfg.get("endpoint", "")
        self.api_key = os.getenv("VERVE_API_KEY", vcfg.get("api_key", ""))
        self.alert_threshold = config.get("monitoring", {}).get("alert_threshold", 0.7)
        self.batch_size = config.get("monitoring", {}).get("batch_size", 100)
        self._buffer: list[dict] = []

    # ------------------------------------------------------------------
    def log_prediction(
        self,
        patient_id: str,
        features: dict,
        prediction: int,
        probability: float,
    ) -> None:
        event = {
            "event_type": "prediction",
            "patient_id": patient_id,
            "probability": probability,
            "prediction": prediction,
            "risk_level": "high" if probability >= self.alert_threshold else "normal",
            "timestamp": _now(),
        }
        self._buffer.append(event)

        if probability >= self.alert_threshold:
            logger.warning(
                f"HIGH RISK — patient {patient_id}: p={probability:.3f}"
            )
            self._emit(self._buffer.copy())
            self._buffer.clear()
        elif len(self._buffer) >= self.batch_size:
            self._flush()

    def log_model_metrics(self, metrics: dict[str, float], run_id: str) -> None:
        event = {
            "event_type": "model_metrics",
            "run_id": run_id,
            "metrics": metrics,
            "timestamp": _now(),
        }
        self._emit([event])

    def _flush(self) -> None:
        if self._buffer:
            self._emit(self._buffer.copy())
            self._buffer.clear()

    def _emit(self, events: list[dict]) -> bool:
        if not self.enabled or not self.endpoint:
            logger.debug("Verve not configured — event dropped")
            return False
        payload = {
            "source": "diabetes-tracker",
            "timestamp": _now(),
            "events": events,
        }
        try:
            resp = requests.post(
                self.endpoint,
                json=payload,
                headers={"X-API-Key": self.api_key, "Content-Type": "application/json"},
                timeout=5,
            )
            resp.raise_for_status()
            logger.info(f"Emitted {len(events)} event(s) to Verve")
            return True
        except requests.RequestException as exc:
            logger.warning(f"Verve emit failed: {exc}")
            return False

    def __del__(self) -> None:
        self._flush()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
