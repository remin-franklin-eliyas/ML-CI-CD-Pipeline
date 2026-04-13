"""
Model training script.

Usage
-----
    python training/train.py

Environment variables (all optional)
--------------------------------------
DATA_PATH             Path to training CSV.          Default: data/sample.csv
MODEL_PATH            Where to save model.pkl.       Default: model.pkl
MLFLOW_TRACKING_URI   MLflow tracking server / dir.  Default: mlflow_tracking
N_ESTIMATORS          RF hyper-parameter.            Default: 100
MAX_DEPTH             RF hyper-parameter.            Default: 10
MIN_SAMPLES_SPLIT     RF hyper-parameter.            Default: 2
RANDOM_STATE          Global seed.                   Default: 42
"""

import logging
import os
import sys

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Make sure the project root is on sys.path when the script is executed directly
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from training.preprocess import (  # noqa: E402
    extract_features_and_labels,
    load_data,
    split_data,
    validate_data,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (from environment variables with sensible defaults)
# ---------------------------------------------------------------------------
DATA_PATH = os.getenv(
    "DATA_PATH",
    os.path.join(_PROJECT_ROOT, "data", "sample.csv"),
)
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(_PROJECT_ROOT, "model.pkl"))
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    os.path.join(_PROJECT_ROOT, "mlflow_tracking"),
)
EXPERIMENT_NAME = "iris-classifier"

# Hyper-parameters
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", "100"))
MAX_DEPTH = int(os.getenv("MAX_DEPTH", "10"))
MIN_SAMPLES_SPLIT = int(os.getenv("MIN_SAMPLES_SPLIT", "2"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train() -> tuple[Pipeline, dict[str, float]]:
    """
    Full training pipeline: load → validate → preprocess → train → evaluate → save.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        The fitted model pipeline (scaler + classifier).
    metrics : dict
        Evaluation metrics computed on the held-out test split.
    """
    # ── MLflow setup ──────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        logger.info("MLflow run ID: %s", run.info.run_id)

        # ── 1. Load & validate data ────────────────────────────────────────
        df = load_data(DATA_PATH)
        validate_data(df)

        # ── 2. Feature extraction & split ─────────────────────────────────
        X, y = extract_features_and_labels(df)
        X_train, X_test, y_train, y_test = split_data(X, y, random_state=RANDOM_STATE)
        logger.info("Train / test split: %d / %d samples.", len(X_train), len(X_test))

        # ── 3. Log hyper-parameters ────────────────────────────────────────
        params = {
            "model_type": "RandomForestClassifier",
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "min_samples_split": MIN_SAMPLES_SPLIT,
            "random_state": RANDOM_STATE,
        }
        mlflow.log_params(params)
        logger.info("Hyper-parameters: %s", params)

        # ── 4. Build & fit the pipeline ────────────────────────────────────
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=N_ESTIMATORS,
                        max_depth=MAX_DEPTH,
                        min_samples_split=MIN_SAMPLES_SPLIT,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        logger.info("Training model …")
        pipeline.fit(X_train, y_train)

        # ── 5. Evaluate on held-out test set ──────────────────────────────
        y_pred = pipeline.predict(X_test)
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
            "f1_score": round(float(f1_score(y_test, y_pred, average="weighted")), 6),
            "precision": round(float(precision_score(y_test, y_pred, average="weighted")), 6),
            "recall": round(float(recall_score(y_test, y_pred, average="weighted")), 6),
        }
        for name, value in metrics.items():
            logger.info("  %-12s %.6f", name, value)
        mlflow.log_metrics(metrics)

        # ── 6. Refit on full dataset for the deployment artefact ──────────
        logger.info("Refitting on full dataset for deployment …")
        pipeline.fit(X, y)

        # ── 7. Save model artefact ────────────────────────────────────────
        joblib.dump(pipeline, MODEL_PATH)
        logger.info("Model saved → '%s'", MODEL_PATH)

        # ── 8. Log artefacts in MLflow ────────────────────────────────────
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        mlflow.log_artifact(MODEL_PATH, artifact_path="artefacts")

        logger.info("Training complete!  Run ID: %s", run.info.run_id)

    return pipeline, metrics


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _, metrics = train()
    # Non-zero exit if accuracy is below threshold – catches regressions in CI
    accuracy_threshold = float(os.getenv("ACCURACY_THRESHOLD", "0.85"))
    if metrics["accuracy"] < accuracy_threshold:
        logger.error(
            "Accuracy %.4f is below the required threshold %.4f!",
            metrics["accuracy"],
            accuracy_threshold,
        )
        sys.exit(1)
    logger.info("Training succeeded with accuracy=%.4f ✓", metrics["accuracy"])
