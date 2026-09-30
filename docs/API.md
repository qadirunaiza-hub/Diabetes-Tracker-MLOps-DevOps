# Diabetes Tracker — API Reference

Base URL: `http://localhost:5001`

---

## GET /health

Liveness check. Used by Docker health checks and load balancers.

**Response 200**
```json
{
  "status": "ok",
  "timestamp": "2026-09-30T10:00:00+00:00"
}
```

---

## GET /model-info

Returns metadata about the currently loaded model version.

**Response 200**
```json
{
  "name":    "DiabetesPredictor",
  "version": "3",
  "stage":   "Production",
  "run_id":  "a1b2c3d4e5f6..."
}
```

---

## POST /predict

Run a diabetes-risk prediction for a single patient record.

**Request body (JSON)**

| Field                     | Type  | Description                                 |
|---------------------------|-------|---------------------------------------------|
| `Pregnancies`             | float | Number of times pregnant (0–17)             |
| `Glucose`                 | float | Plasma glucose concentration mg/dL (0–199)  |
| `BloodPressure`           | float | Diastolic blood pressure mm Hg (0–122)      |
| `SkinThickness`           | float | Triceps skin fold thickness mm (0–99)       |
| `Insulin`                 | float | 2-hour serum insulin mu U/ml (0–846)        |
| `BMI`                     | float | Body mass index kg/m² (0–67.1)              |
| `DiabetesPedigreeFunction`| float | Diabetes pedigree function (0.078–2.42)     |
| `Age`                     | float | Age in years (21–81)                        |

**Example request**
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 2,
    "Glucose": 138,
    "BloodPressure": 78,
    "SkinThickness": 32,
    "Insulin": 140,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 45
  }'
```

**Response 200**
```json
{
  "prediction":        1,
  "prediction_label":  "Diabetes Risk Detected",
  "probability":       0.7834,
  "risk_level":        "High",
  "model_version":     "3",
  "disclaimer":        "This is an educational ML demonstration. It is NOT a medical diagnostic tool..."
}
```

| Field              | Description                                          |
|--------------------|------------------------------------------------------|
| `prediction`       | `0` = no risk detected, `1` = risk detected          |
| `prediction_label` | Human-readable label                                 |
| `probability`      | Model confidence (0.0 – 1.0) of the positive class  |
| `risk_level`       | `Low` (<30%), `Moderate` (30–60%), `High` (>60%)    |
| `model_version`    | MLflow registry version that served this request     |
| `disclaimer`       | Always present — this is not a diagnostic tool       |

**Error responses**

| Code | Reason                                  |
|------|-----------------------------------------|
| 400  | Missing required feature fields         |
| 500  | Internal prediction error               |
| 503  | Model not loaded (pipeline not run yet) |

---

## GET /predictions

Returns the 50 most recent prediction log entries from SQLite.

**Response 200**
```json
[
  {
    "timestamp":     "2026-09-30T10:05:23+00:00",
    "prediction":    1,
    "probability":   0.7834,
    "risk_level":    "High",
    "model_version": "3",
    "input": { "Pregnancies": 2, "Glucose": 138, ... }
  }
]
```
