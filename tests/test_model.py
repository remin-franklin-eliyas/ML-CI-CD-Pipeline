"""
Unit tests for the model loading and prediction functions in app/model.py.
"""

import os

import numpy as np
import pytest

from app.model import SPECIES_MAP, load_model, predict
from app.schemas import PredictionRequest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def loaded_model(ensure_model):  # noqa: ARG001
    """Return the loaded sklearn pipeline for the module."""
    return load_model()


# ---------------------------------------------------------------------------
# SPECIES_MAP
# ---------------------------------------------------------------------------
class TestSpeciesMap:
    def test_species_map_has_three_entries(self):
        assert len(SPECIES_MAP) == 3

    def test_species_map_labels(self):
        assert SPECIES_MAP[0] == "setosa"
        assert SPECIES_MAP[1] == "versicolor"
        assert SPECIES_MAP[2] == "virginica"


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------
class TestLoadModel:
    def test_load_model_returns_object(self, loaded_model):
        assert loaded_model is not None

    def test_load_model_has_predict(self, loaded_model):
        assert hasattr(loaded_model, "predict")

    def test_load_model_has_predict_proba(self, loaded_model):
        assert hasattr(loaded_model, "predict_proba")

    def test_load_model_raises_for_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MODEL_PATH", str(tmp_path / "nonexistent.pkl"))
        with pytest.raises(FileNotFoundError, match="not found"):
            load_model()


# ---------------------------------------------------------------------------
# predict – output contract
# ---------------------------------------------------------------------------
class TestPredict:
    _SETOSA = PredictionRequest(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2)
    _VERSICOLOR = PredictionRequest(sepal_length=6.0, sepal_width=2.9, petal_length=4.5, petal_width=1.5)
    _VIRGINICA = PredictionRequest(sepal_length=6.3, sepal_width=3.3, petal_length=6.0, petal_width=2.5)

    def test_predict_returns_response(self, loaded_model):
        from app.schemas import PredictionResponse

        result = predict(loaded_model, self._SETOSA)
        assert isinstance(result, PredictionResponse)

    def test_predict_species_is_string(self, loaded_model):
        result = predict(loaded_model, self._SETOSA)
        assert isinstance(result.species, str)

    def test_predict_species_in_valid_set(self, loaded_model):
        result = predict(loaded_model, self._SETOSA)
        assert result.species in {"setosa", "versicolor", "virginica"}

    def test_predict_species_id_in_valid_range(self, loaded_model):
        result = predict(loaded_model, self._SETOSA)
        assert result.species_id in {0, 1, 2}

    def test_predict_probability_between_0_and_1(self, loaded_model):
        result = predict(loaded_model, self._SETOSA)
        assert 0.0 <= result.probability <= 1.0

    def test_predict_model_version_is_string(self, loaded_model):
        result = predict(loaded_model, self._SETOSA)
        assert isinstance(result.model_version, str)

    def test_predict_setosa_correct(self, loaded_model):
        result = predict(loaded_model, self._SETOSA)
        assert result.species == "setosa"

    def test_predict_virginica_correct(self, loaded_model):
        result = predict(loaded_model, self._VIRGINICA)
        assert result.species == "virginica"

    def test_predict_all_classes_reachable(self, loaded_model):
        """Each class should be predicted for at least one representative sample."""
        samples = [self._SETOSA, self._VERSICOLOR, self._VIRGINICA]
        predicted = {predict(loaded_model, s).species for s in samples}
        # With a proper model all three classes should be reachable
        assert len(predicted) == 3

    def test_predict_probabilities_sum_to_one(self, loaded_model):
        """predict_proba should sum to 1 for any input."""
        import joblib

        model_path = os.getenv("MODEL_PATH", "model.pkl")
        model = joblib.load(model_path)
        features = np.array([[5.1, 3.5, 1.4, 0.2]])
        probs = model.predict_proba(features)[0]
        assert abs(probs.sum() - 1.0) < 1e-6
