"""
Unit tests for the FastAPI application endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client(ensure_model):  # noqa: ARG001  – ensure_model is the session fixture
    """Yield a synchronous TestClient that exercises the full ASGI lifespan."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client: TestClient):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data

    def test_health_model_is_loaded(self, client: TestClient):
        data = client.get("/health").json()
        assert data["model_loaded"] is True
        assert data["status"] == "healthy"


# ---------------------------------------------------------------------------
# /predict – happy path
# ---------------------------------------------------------------------------
class TestPredictEndpoint:
    _SETOSA_INPUT = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    _VIRGINICA_INPUT = {
        "sepal_length": 6.3,
        "sepal_width": 3.3,
        "petal_length": 6.0,
        "petal_width": 2.5,
    }

    def test_predict_returns_200(self, client: TestClient):
        response = client.post("/predict", json=self._SETOSA_INPUT)
        assert response.status_code == 200

    def test_predict_response_schema(self, client: TestClient):
        data = client.post("/predict", json=self._SETOSA_INPUT).json()
        assert "species" in data
        assert "species_id" in data
        assert "probability" in data
        assert "model_version" in data

    def test_predict_species_is_valid(self, client: TestClient):
        data = client.post("/predict", json=self._SETOSA_INPUT).json()
        assert data["species"] in {"setosa", "versicolor", "virginica"}

    def test_predict_species_id_is_valid(self, client: TestClient):
        data = client.post("/predict", json=self._SETOSA_INPUT).json()
        assert data["species_id"] in {0, 1, 2}

    def test_predict_probability_range(self, client: TestClient):
        data = client.post("/predict", json=self._SETOSA_INPUT).json()
        assert 0.0 <= data["probability"] <= 1.0

    def test_predict_setosa_classified_correctly(self, client: TestClient):
        """A clear setosa sample should be predicted as setosa."""
        data = client.post("/predict", json=self._SETOSA_INPUT).json()
        assert data["species"] == "setosa"

    def test_predict_virginica_classified_correctly(self, client: TestClient):
        """A clear virginica sample should be predicted as virginica."""
        data = client.post("/predict", json=self._VIRGINICA_INPUT).json()
        assert data["species"] == "virginica"

    def test_predict_model_version_present(self, client: TestClient):
        data = client.post("/predict", json=self._SETOSA_INPUT).json()
        assert len(data["model_version"]) > 0


# ---------------------------------------------------------------------------
# /predict – validation errors
# ---------------------------------------------------------------------------
class TestPredictValidation:
    def test_predict_rejects_negative_sepal_length(self, client: TestClient):
        response = client.post(
            "/predict",
            json={"sepal_length": -1.0, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
        )
        assert response.status_code == 422

    def test_predict_rejects_zero_petal_width(self, client: TestClient):
        response = client.post(
            "/predict",
            json={"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.0},
        )
        assert response.status_code == 422

    def test_predict_rejects_missing_field(self, client: TestClient):
        response = client.post(
            "/predict",
            json={"sepal_length": 5.1, "sepal_width": 3.5},
        )
        assert response.status_code == 422

    def test_predict_rejects_string_input(self, client: TestClient):
        response = client.post(
            "/predict",
            json={"sepal_length": "abc", "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
        )
        assert response.status_code == 422

    def test_predict_rejects_empty_body(self, client: TestClient):
        response = client.post("/predict", json={})
        assert response.status_code == 422
