# Diabetes Tracker — MLOps Educational Project

> **Disclaimer**: This is a learning project. It demonstrates MLOps concepts using a synthetic medical-style dataset. It is **not** a clinical tool and must **not** be used for medical decisions.

---

## What This Project Teaches

| MLOps Component       | Where It Appears                     | Key Concept                                                |
|-----------------------|--------------------------------------|------------------------------------------------------------|
| Data Generation       | `data/generate_data.py`              | Reproducible synthetic data with a fixed seed              |
| Data Validation       | `src/data_validation.py`             | Schema, range, null, and class-balance checks              |
| Preprocessing         | `src/preprocessing.py`               | Zero-imputation, StandardScaler, train/test split          |
| Model Training        | `src/train.py`                       | Multiple classifiers, hyperparameters                      |
| Experiment Tracking   | `src/train.py` → MLflow             | Params, metrics, artifact logging per run                  |
| Model Evaluation      | `src/evaluate.py`                    | Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix |
| Model Registry        | `src/train.py` → MLflow Registry    | Version control for models, stage transitions              |
| Serving               | `api/app.py`                         | Flask REST API loading the Production model                |
| Prediction Logging    | `api/app.py` → SQLite               | Auditability — every inference is recorded                 |
| Containerisation      | `Dockerfile`, `docker-compose.yml`   | Reproducible deployment, health checks                     |
| Web UI                | `api/templates/index.html`           | Feature form + live prediction log                         |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MLOps Pipeline                       │
│                                                         │
│  generate_data.py                                       │
│       │                                                 │
│       ▼                                                 │
│  data_validation.py  ── validates schema, ranges, nulls │
│       │                                                 │
│       ▼                                                 │
│  preprocessing.py  ─── impute zeros → scale features   │
│       │                                                 │
│       ▼                                                 │
│  train.py ──────────── 4 models trained                 │
│       │                                                 │
│       ├──► MLflow Tracking (./mlruns)                   │
│       │       params / metrics / plots / model artifact  │
│       │                                                 │
│       └──► MLflow Model Registry                        │
│               DiabetesPredictor → Production            │
│                                                         │
└─────────────────────────────────────────────────────────┘
              │
              ▼  (reads Production model at startup)
┌─────────────────────────┐     ┌──────────────────────┐
│   Flask API (port 5001) │     │  MLflow UI (port 5002)│
│                         │     │                      │
│  POST /predict          │     │  Experiment browser  │
│  GET  /health           │     │  Model registry      │
│  GET  /model-info       │     │  Run comparison      │
│  GET  /predictions      │     │                      │
│         │               │     └──────────────────────┘
│         ▼               │
│  SQLite predictions.db  │
│  (audit log)            │
│         │               │
│         ▼               │
│  HTML Web UI            │
│  (form + log table)     │
└─────────────────────────┘
```

---

## Quick Start

### Prerequisites

```bash
python 3.11+
pip
docker (optional, for containerised deployment)
```

### 1 — Clone and install

```bash
cd diabetes-tracker
cp .env.example .env
pip install -r requirements.txt
```

### 2 — Run the full pipeline

```bash
python run_pipeline.py
```

This will:
- Generate `data/diabetes.csv` (768 synthetic samples)
- Validate the dataset
- Train 4 models (Logistic Regression, Random Forest, Gradient Boosting, SVM)
- Log all runs to MLflow (`./mlruns`)
- Register the best model as `DiabetesPredictor` in Production

### 3 — Explore experiments in the MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5002
# Open: http://localhost:5002
```

You'll see:
- All 4 experiment runs with their metrics
- Downloadable plot artifacts (confusion matrices, ROC curves, feature importance)
- The Model Registry with the Production version highlighted

### 4 — Start the prediction API

```bash
python -m api.app
# Open: http://localhost:5001
```

### 5 — Run via Docker (recommended for production-like setup)

```bash
# First run the pipeline to populate ./mlruns and ./models
python run_pipeline.py

# Then start both services
docker compose up --build
```

- API: http://localhost:5001
- MLflow UI: http://localhost:5002

---

## Dataset

Synthetic data generated by `data/generate_data.py`. It mirrors the statistical
properties of the **Pima Indians Diabetes Dataset** (UCI ML Repository) — same
columns, similar class distribution (~34.8% positive), and realistic correlations
between glucose, BMI, age, and outcome.

| Feature                   | Description                                        |
|---------------------------|----------------------------------------------------|
| Pregnancies               | Number of times pregnant                           |
| Glucose                   | Plasma glucose concentration (mg/dL)               |
| BloodPressure             | Diastolic blood pressure (mm Hg)                   |
| SkinThickness             | Triceps skin fold thickness (mm)                   |
| Insulin                   | 2-hour serum insulin (mu U/ml)                     |
| BMI                       | Body mass index (kg/m²)                            |
| DiabetesPedigreeFunction  | Genetic diabetes risk score                        |
| Age                       | Age in years                                       |
| Outcome                   | 1 = diabetes risk, 0 = no diabetes risk            |

**Zero-coding**: In the original dataset, zero values in medical columns
(Glucose, BloodPressure, SkinThickness, Insulin, BMI) indicate missing
measurements, not true zeros. The preprocessing module replaces them with
NaN and imputes using the column median.

---

## Models Trained

| Model               | Notes                                                       |
|---------------------|-------------------------------------------------------------|
| Logistic Regression | Baseline; fast; highly interpretable                        |
| Random Forest       | Ensemble of decision trees; provides feature importance     |
| Gradient Boosting   | Often best accuracy; sequential boosted trees               |
| SVM (RBF)           | Effective in high-dimensional feature spaces                |

The model with the highest **ROC-AUC** on the test set is automatically
registered to the MLflow Model Registry and promoted to **Production**.

---

## MLflow Deep Dive

### How experiment tracking works

```python
with mlflow.start_run(run_name="Random Forest"):
    mlflow.log_params({"n_estimators": 150, "max_depth": 8})
    mlflow.log_metrics({"accuracy": 0.82, "roc_auc": 0.88})
    mlflow.log_artifact("docs/plots/roc_curve_random_forest.png")
    mlflow.sklearn.log_model(model, artifact_path="model")
```

Each `start_run` creates an isolated experiment entry with a unique `run_id`.
The full model, parameters, metrics, and plots are stored under `./mlruns/`.

### How the model registry works

```python
# Register the best run's model
mv = mlflow.register_model(f"runs:/{run_id}/model", "DiabetesPredictor")

# Promote to Production
client.transition_model_version_stage(
    name="DiabetesPredictor",
    version=mv.version,
    stage="Production",
    archive_existing_versions=True,
)
```

### How the API loads the Production model

```python
model = mlflow.sklearn.load_model("models:/DiabetesPredictor/Production")
```

The URI `models:/Name/Stage` always resolves to whichever version is
currently tagged as Production — enabling zero-downtime model updates
without changing application code.

---

## Project Structure

```
diabetes-tracker/
├── data/
│   ├── generate_data.py     # synthetic dataset generator
│   └── diabetes.csv         # generated on first pipeline run
├── src/
│   ├── utils.py             # shared constants and helpers
│   ├── data_validation.py   # validation checks + report
│   ├── preprocessing.py     # imputation, scaling, splitting
│   ├── train.py             # training + MLflow logging + registry
│   └── evaluate.py          # metrics + plot generation
├── api/
│   ├── app.py               # Flask REST API
│   └── templates/
│       └── index.html       # web frontend
├── docs/
│   ├── API.md               # endpoint reference
│   └── plots/               # generated evaluation plots
├── models/                  # scaler.joblib, imputer.joblib
├── logs/                    # predictions.db (SQLite)
├── mlruns/                  # MLflow tracking data (auto-created)
├── Dockerfile
├── docker-compose.yml
├── run_pipeline.py          # full pipeline orchestrator
├── requirements.txt
└── .env.example
```

---

## Environment Variables

Copy `.env.example` to `.env` and adjust as needed.

| Variable               | Default            | Description                          |
|------------------------|--------------------|--------------------------------------|
| `MLFLOW_TRACKING_URI`  | `./mlruns`         | MLflow backend store URI             |
| `MODEL_NAME`           | `DiabetesPredictor`| Registered model name                |
| `MODEL_STAGE`          | `Production`       | Model stage to load at API startup   |
| `FLASK_PORT`           | `5001`             | Port for the Flask API               |
| `FLASK_ENV`            | `development`      | `development` enables Flask debugger |
| `LOG_LEVEL`            | `INFO`             | Python logging level                 |

---

## API Reference

See [`docs/API.md`](docs/API.md) for the full endpoint reference.

Quick example:

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

---

## Important Notice

This project uses a synthetic dataset and is designed **solely for educational purposes** to demonstrate MLOps concepts. The predictions it produces are not validated for clinical use and should never influence medical decisions. Always consult a qualified healthcare professional for any health-related concerns.
