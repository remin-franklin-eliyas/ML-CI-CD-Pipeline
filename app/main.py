"""
FastAPI application entry-point.

Endpoints
---------
GET  /health   – liveness / readiness probe
POST /predict  – single-sample Iris species prediction
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.model import load_model, predict
from app.schemas import HealthResponse, PredictionRequest, PredictionResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# Module-level model reference – populated during application startup
_model = None


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Load the ML model once at startup; release resources on shutdown."""
    global _model
    try:
        _model = load_model()
        logger.info("Model is ready for inference.")
    except FileNotFoundError as exc:
        logger.error("Model could not be loaded: %s", exc)
        _model = None
    yield
    logger.info("Application shutting down.")
    _model = None


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Iris Species Classifier API",
    description=(
        "A production-ready REST API that classifies Iris flowers into three "
        "species (setosa, versicolor, virginica) using a RandomForest model "
        "trained with scikit-learn and tracked with MLflow."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Exception handler
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:  # noqa: ARG001
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check() -> HealthResponse:
    """
    Liveness and readiness probe.

    Returns HTTP 200 with ``status='healthy'`` when the model is loaded,
    or ``status='degraded'`` when the model failed to load.
    """
    return HealthResponse(
        status="healthy" if _model is not None else "degraded",
        model_loaded=_model is not None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_species(request: PredictionRequest) -> PredictionResponse:
    """
    Classify a single Iris flower sample.

    Accepts four continuous measurements (sepal / petal length and width in cm)
    and returns the predicted species together with the model's confidence score.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. The service is temporarily unavailable.",
        )
    try:
        result = predict(_model, request)
        logger.info(
            "Predicted '%s' (id=%d, confidence=%.4f)",
            result.species,
            result.species_id,
            result.probability,
        )
        return result
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
