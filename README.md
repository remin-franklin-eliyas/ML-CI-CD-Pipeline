# ML CI/CD Pipeline

> A production-ready, end-to-end Machine Learning CI/CD pipeline that
> automatically tests, trains, containerises, and deploys an Iris species
> classifier as a REST API on every push to `main`.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Tech Stack](#tech-stack)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [API Reference](#api-reference)
6. [Training](#training)
7. [Data Validation](#data-validation)
8. [Experiment Tracking (MLflow)](#experiment-tracking-mlflow)
9. [Testing](#testing)
10. [Docker](#docker)
11. [CI/CD Pipeline](#cicd-pipeline)
12. [Deployment](#deployment)
13. [Blue-Green Deployment](#blue-green-deployment)
14. [Configuration Reference](#configuration-reference)

---

## Architecture Overview

```
GitHub push
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  GitHub Actions                                         │
│                                                         │
│  1. validate-and-train                                  │
│     ├─ pip install -r requirements.txt                  │
│     ├─ Great Expectations → validate data/sample.csv    │
│     ├─ python training/train.py  (+ MLflow logging)     │
│     └─ upload model.pkl artefact                        │
│                                                         │
│  2. test  (needs: validate-and-train)                   │
│     ├─ download model.pkl artefact                      │
│     └─ pytest tests/ -v                                 │
│                                                         │
│  3. build-and-push  (main only, needs: test)            │
│     ├─ docker build  (copies model.pkl into image)      │
│     └─ docker push → Docker Hub                         │
│                                                         │
│  4. deploy  (main only, needs: build-and-push)          │
│     └─ ./deploy/deploy.sh --dry-run                     │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Docker Hub  →  Production server
               docker run -p 8000:8000 <image>
               GET  /health
               POST /predict
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| API framework | FastAPI + Uvicorn |
| ML library | scikit-learn (RandomForestClassifier) |
| Data validation | Great Expectations 0.18 |
| Experiment tracking | MLflow |
| Serialisation | joblib |
| Containerisation | Docker (multi-stage, python:3.11-slim) |
| CI/CD | GitHub Actions |
| Testing | pytest + httpx |
| Schema validation | Pydantic v2 |

---

## Project Structure

```
ml-ci-cd-pipeline/
│
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app (lifespan, /health, /predict)
│   ├── model.py         # Model loading & prediction logic
│   └── schemas.py       # Pydantic request / response schemas
│
├── training/
│   ├── __init__.py
│   ├── preprocess.py    # Data loading, GE validation, feature extraction
│   └── train.py         # Training script (MLflow tracking, model saving)
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py      # Session-scoped model fixture
│   ├── test_api.py      # FastAPI endpoint tests
│   └── test_model.py    # Model unit tests
│
├── data/
│   └── sample.csv       # 150-row Iris dataset
│
├── deploy/
│   ├── deploy.sh              # Standard deployment script
│   └── blue_green_deploy.sh   # Blue-green deployment simulation
│
├── mlflow_tracking/     # MLflow run data (git-ignored, directory tracked)
│
├── Dockerfile           # Multi-stage production image
├── requirements.txt
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci-cd.yml    # Full CI/CD pipeline
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional, for containerised runs)

### 1 – Clone and install

```bash
git clone https://github.com/<your-org>/ML-CI-CD-Pipeline.git
cd ML-CI-CD-Pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2 – Train the model

```bash
python training/train.py
# Creates model.pkl in the project root
# Logs metrics to mlflow_tracking/
```

### 3 – Run the API locally

```bash
uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000/docs> for the interactive Swagger UI.

### 4 – Make a prediction

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

Example response:

```json
{
  "species": "setosa",
  "species_id": 0,
  "probability": 1.0,
  "model_version": "1.0.0"
}
```

---

## API Reference

### `GET /health`

Liveness and readiness probe.

| Field | Type | Description |
|---|---|---|
| `status` | string | `healthy` or `degraded` |
| `model_loaded` | boolean | Whether the model is ready |

### `POST /predict`

Classify a single Iris flower sample.

**Request body**

| Field | Type | Constraints |
|---|---|---|
| `sepal_length` | float | > 0 |
| `sepal_width` | float | > 0 |
| `petal_length` | float | > 0 |
| `petal_width` | float | > 0 |

**Response**

| Field | Type | Description |
|---|---|---|
| `species` | string | `setosa`, `versicolor`, or `virginica` |
| `species_id` | int | 0, 1, or 2 |
| `probability` | float | Model confidence [0, 1] |
| `model_version` | string | Version tag of the deployed model |

---

## Training

```bash
# All parameters can be overridden via environment variables
DATA_PATH=data/sample.csv \
MODEL_PATH=model.pkl \
N_ESTIMATORS=100 \
MAX_DEPTH=10 \
ACCURACY_THRESHOLD=0.85 \
python training/train.py
```

The training script:
1. Loads and validates the dataset with Great Expectations.
2. Performs a stratified 80/20 train/test split.
3. Fits a `StandardScaler → RandomForestClassifier` pipeline on the training set.
4. Evaluates accuracy, F1, precision, and recall on the test set.
5. Refits the pipeline on the full dataset for deployment.
6. Saves `model.pkl` and logs all parameters, metrics, and artefacts to MLflow.
7. Exits with code 1 if accuracy falls below `ACCURACY_THRESHOLD`.

---

## Data Validation

Data validation runs automatically before training and is also a standalone
CI step.  The following checks are enforced:

| Check | Detail |
|---|---|
| Column presence | All five columns must exist |
| No nulls | Every cell must be filled |
| Value ranges | Numeric features must be in (0, 30] cm |
| Domain values | `species` must be one of the three valid classes |
| Row count | Dataset must have ≥ 10 rows |

A validation failure raises a `ValueError` and halts the pipeline immediately.

---

## Experiment Tracking (MLflow)

```bash
# View the MLflow UI
mlflow ui --backend-store-uri mlflow_tracking --port 5000
```

Open <http://localhost:5000> to explore runs, compare metrics, and download
model artefacts.

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run only API tests
pytest tests/test_api.py -v

# Run only model unit tests
pytest tests/test_model.py -v
```

The `conftest.py` session fixture creates a minimal `model.pkl` automatically
if one does not already exist, so tests can run without prior training.

---

## Docker

### Build

```bash
# 1. Train model first (model.pkl must exist for the COPY step)
python training/train.py

# 2. Build the image
docker build -t ml-iris-classifier:latest .
```

### Run

```bash
docker run -d \
    --name ml-api \
    -p 8000:8000 \
    -e MODEL_VERSION=1.0.0 \
    ml-iris-classifier:latest
```

The container includes a `HEALTHCHECK` directive; Docker will mark it
unhealthy if `/health` does not return HTTP 200 within 10 seconds.

---

## CI/CD Pipeline

### Pipeline stages

```
push to main
     │
     ├─► validate-and-train  ──► test  ──► build-and-push  ──► deploy
     │                                                           (dry-run)
     └─► pull_request check  ──► test   (no Docker push)
```

### Required GitHub Secrets

| Secret | Description |
|---|---|
| `DOCKER_HUB_USERNAME` | Docker Hub account username |
| `DOCKER_HUB_TOKEN` | Docker Hub access token (Settings → Security) |

Go to **Settings → Secrets and variables → Actions → New repository secret**
to add them.

---

## Deployment

### Standard deployment

```bash
# Pull and restart the container
./deploy/deploy.sh myorg/ml-iris-classifier latest

# Dry-run (prints commands without executing)
./deploy/deploy.sh myorg/ml-iris-classifier abc1234 --dry-run
```

---

## Blue-Green Deployment

```bash
# Deploy a new version with zero downtime
./deploy/blue_green_deploy.sh myorg/ml-iris-classifier abc1234
```

The script:
1. Determines which slot (blue / green) is currently active.
2. Starts the new image in the **idle** slot on a secondary port.
3. Runs a health-check loop against the new slot.
4. Prints a message to switch the load balancer to the new port.
5. Stops the old slot.

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `model.pkl` | Path to the serialised model pipeline |
| `MODEL_VERSION` | `1.0.0` | Version tag included in API responses |
| `DATA_PATH` | `data/sample.csv` | Path to training CSV |
| `MLFLOW_TRACKING_URI` | `mlflow_tracking` | MLflow tracking backend |
| `N_ESTIMATORS` | `100` | Number of RF trees |
| `MAX_DEPTH` | `10` | Maximum RF tree depth |
| `MIN_SAMPLES_SPLIT` | `2` | Minimum samples to split an RF node |
| `RANDOM_STATE` | `42` | Global random seed |
| `ACCURACY_THRESHOLD` | `0.85` | Minimum required accuracy (training gate) |