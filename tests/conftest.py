"""
pytest session-scoped fixtures shared by all test modules.

The fixture ``ensure_model`` guarantees that a valid ``model.pkl`` exists at
the path pointed to by the MODEL_PATH environment variable before any test
runs.  This allows the full test suite to execute in CI environments that
run tests *after* the training step (which creates model.pkl), but it also
creates a minimal in-memory model when the artefact is absent (e.g. during
local development without running training first).
"""

import os

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Determine a stable model path relative to the repository root
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_MODEL_PATH = os.path.join(_REPO_ROOT, "model.pkl")

# Small but representative training set (5 samples per class)
_TRAIN_X = np.array(
    [
        # setosa (class 0)
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [4.7, 3.2, 1.3, 0.2],
        [4.6, 3.1, 1.5, 0.2],
        [5.0, 3.6, 1.4, 0.2],
        # versicolor (class 1)
        [7.0, 3.2, 4.7, 1.4],
        [6.4, 3.2, 4.5, 1.5],
        [6.9, 3.1, 4.9, 1.5],
        [5.5, 2.3, 4.0, 1.3],
        [6.5, 2.8, 4.6, 1.5],
        # virginica (class 2)
        [6.3, 3.3, 6.0, 2.5],
        [5.8, 2.7, 5.1, 1.9],
        [7.1, 3.0, 5.9, 2.1],
        [6.3, 2.9, 5.6, 1.8],
        [6.5, 3.0, 5.8, 2.2],
    ],
    dtype=np.float64,
)
_TRAIN_Y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2], dtype=np.int64)


@pytest.fixture(scope="session", autouse=True)
def ensure_model():
    """
    Ensure a model.pkl artefact exists for the duration of the test session.

    * If MODEL_PATH env var is set, it is used as-is.
    * Otherwise the default repository-root model.pkl is used.
    * If the file does not exist, a minimal pipeline is trained and saved.
    """
    model_path = os.getenv("MODEL_PATH", _DEFAULT_MODEL_PATH)

    if not os.path.exists(model_path):
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
            ]
        )
        pipeline.fit(_TRAIN_X, _TRAIN_Y)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(pipeline, model_path)

    # Expose the path via env var so app/model.py picks it up at runtime
    os.environ["MODEL_PATH"] = model_path
    yield model_path
