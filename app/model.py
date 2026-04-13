"""
Model loading and prediction logic.

The model artefact is a scikit-learn Pipeline (StandardScaler → RandomForestClassifier)
serialised with joblib.  The path is resolved from the MODEL_PATH environment variable
so that it can be overridden at runtime (e.g. different paths in Docker vs. tests).
"""

import logging
import os

import joblib
import numpy as np

from app.schemas import PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)

# Human-readable mapping from numeric class label to species name
SPECIES_MAP: dict[int, str] = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}


def get_model_path() -> str:
    """Return the model file path, preferring the MODEL_PATH env variable."""
    return os.getenv("MODEL_PATH", "model.pkl")


def load_model():
    """
    Load the trained scikit-learn pipeline from disk.

    Returns
    -------
    sklearn.pipeline.Pipeline
        The loaded model pipeline ready for inference.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist at the resolved path.
    """
    model_path = get_model_path()
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artefact not found at '{model_path}'. "
            "Run 'python training/train.py' to create it."
        )
    logger.info("Loading model from '%s' …", model_path)
    model = joblib.load(model_path)
    logger.info("Model loaded successfully.")
    return model


def predict(model, request: PredictionRequest) -> PredictionResponse:
    """
    Run inference for a single flower sample.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        A fitted pipeline that includes preprocessing and a classifier.
    request : PredictionRequest
        Validated input features from the API caller.

    Returns
    -------
    PredictionResponse
        Predicted species name, numeric label, confidence, and model version.
    """
    features = np.array(
        [[request.sepal_length, request.sepal_width, request.petal_length, request.petal_width]]
    )

    predicted_label: int = int(model.predict(features)[0])
    probabilities: np.ndarray = model.predict_proba(features)[0]
    confidence: float = round(float(probabilities[predicted_label]), 6)

    logger.debug(
        "Prediction: label=%d species=%s confidence=%.4f",
        predicted_label,
        SPECIES_MAP[predicted_label],
        confidence,
    )

    return PredictionResponse(
        species=SPECIES_MAP[predicted_label],
        species_id=predicted_label,
        probability=confidence,
        model_version=os.getenv("MODEL_VERSION", "1.0.0"),
    )
