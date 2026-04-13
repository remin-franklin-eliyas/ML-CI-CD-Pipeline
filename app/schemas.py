"""
Pydantic schemas for API request and response payloads.
"""

from pydantic import BaseModel, Field, ConfigDict


class PredictionRequest(BaseModel):
    """Input features for a single Iris flower prediction."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            }
        }
    )

    sepal_length: float = Field(..., gt=0, description="Sepal length in centimetres")
    sepal_width: float = Field(..., gt=0, description="Sepal width in centimetres")
    petal_length: float = Field(..., gt=0, description="Petal length in centimetres")
    petal_width: float = Field(..., gt=0, description="Petal width in centimetres")


class PredictionResponse(BaseModel):
    """Prediction result returned by the /predict endpoint."""

    model_config = ConfigDict(protected_namespaces=())

    species: str = Field(..., description="Predicted Iris species name")
    species_id: int = Field(..., description="Numeric class label (0=setosa, 1=versicolor, 2=virginica)")
    probability: float = Field(..., description="Model confidence for the predicted class")
    model_version: str = Field(..., description="Version tag of the deployed model")


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""

    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(..., description="Service status (healthy / degraded)")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded and ready")
