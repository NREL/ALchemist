"""
Pydantic response models for API endpoints.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
from datetime import datetime


# ============================================================
# Session Models
# ============================================================

class SessionCreateResponse(BaseModel):
    """Response when creating a new session."""
    session_id: str = Field(..., description="Unique session identifier")
    created_at: str = Field(..., description="Session creation timestamp")
    expires_at: str = Field(..., description="Session expiration timestamp")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "created_at": "2025-10-31T14:30:00",
                "expires_at": "2025-11-01T14:30:00"
            }
        }
    )


class VariableInfo(BaseModel):
    """Information about a variable."""
    name: str
    type: str
    bounds: Optional[List[float]] = None
    categories: Optional[List[str]] = None


class DataSummary(BaseModel):
    """Summary of experimental data."""
    n_experiments: int
    has_data: bool
    has_noise: bool = False
    target_stats: Optional[Dict[str, float]] = None
    feature_names: Optional[List[str]] = None


class ModelSummary(BaseModel):
    """Summary of trained model."""
    backend: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    is_trained: bool


class SessionInfoResponse(BaseModel):
    """Full session information."""
    session_id: str
    created_at: str
    last_accessed: str
    expires_at: str
    search_space: Dict[str, Any]
    data: DataSummary
    model: Optional[ModelSummary]
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "created_at": "2025-10-31T14:30:00",
                "last_accessed": "2025-10-31T14:35:00",
                "expires_at": "2025-11-01T14:30:00",
                "search_space": {
                    "n_variables": 2,
                    "variables": [
                        {"name": "temp", "type": "real", "bounds": [300, 500]},
                        {"name": "catalyst", "type": "categorical", "categories": ["A", "B"]}
                    ]
                },
                "data": {
                    "n_experiments": 10,
                    "has_data": True,
                    "has_noise": False
                },
                "model": None
            }
        }
    )


# ============================================================
# Variable Models
# ============================================================

class VariableResponse(BaseModel):
    """Response when adding a variable."""
    message: str = "Variable added successfully"
    variable: Dict[str, Any]


class VariablesListResponse(BaseModel):
    """List of all variables in search space."""
    variables: List[Dict[str, Any]]
    n_variables: int


# ============================================================
# Experiment Models
# ============================================================

class ExperimentResponse(BaseModel):
    """Response when adding an experiment."""
    message: str = "Experiment added successfully"
    n_experiments: int


class ExperimentsListResponse(BaseModel):
    """List of all experiments."""
    experiments: List[Dict[str, Any]]
    n_experiments: int


class ExperimentsSummaryResponse(BaseModel):
    """Statistical summary of experimental data."""
    n_experiments: int
    has_data: bool
    has_noise: Optional[bool] = None
    target_stats: Optional[Dict[str, float]] = None
    feature_names: Optional[List[str]] = None
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "n_experiments": 10,
                "has_data": True,
                "has_noise": False,
                "target_stats": {
                    "min": 0.5,
                    "max": 0.95,
                    "mean": 0.75,
                    "std": 0.12
                },
                "feature_names": ["temperature", "pressure"]
            }
        }
    )


# ============================================================
# Model Training Models
# ============================================================

class TrainModelResponse(BaseModel):
    """Response from model training."""
    success: bool
    backend: str
    kernel: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    message: str = "Model trained successfully"
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "backend": "sklearn",
                "kernel": "Matern",
                "hyperparameters": {
                    "lengthscales": [50.2, 0.8],
                    "noise_variance": 0.01
                },
                "metrics": {
                    "rmse": 0.05,
                    "mae": 0.03,
                    "r2": 0.95
                },
                "message": "Model trained successfully"
            }
        }
    )


class ModelInfoResponse(BaseModel):
    """Model information response."""
    backend: Optional[str]
    hyperparameters: Optional[Dict[str, Any]]
    metrics: Optional[Dict[str, float]]
    is_trained: bool
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "backend": "sklearn",
                "hyperparameters": {"lengthscales": [50.2, 0.8]},
                "metrics": {"rmse": 0.05, "r2": 0.95},
                "is_trained": True
            }
        }
    )


# ============================================================
# Acquisition Models
# ============================================================

class AcquisitionResponse(BaseModel):
    """Response from acquisition function."""
    suggestions: List[Dict[str, Any]]
    n_suggestions: int
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "suggestions": [
                    {"temperature": 375.2, "catalyst": "A"}
                ],
                "n_suggestions": 1
            }
        }
    )


class FindOptimumResponse(BaseModel):
    """Response from find model optimum."""
    optimum: Dict[str, Any] = Field(..., description="Optimal point found by model")
    predicted_value: float = Field(..., description="Predicted value at optimum")
    predicted_std: Optional[float] = Field(None, description="Standard deviation at optimum")
    goal: str = Field(..., description="Optimization goal (maximize/minimize)")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "optimum": {"temperature": 425.7, "catalyst": "B"},
                "predicted_value": 0.956,
                "predicted_std": 0.023,
                "goal": "maximize"
            }
        }
    )


# ============================================================
# Prediction Models
# ============================================================

class PredictionResult(BaseModel):
    """Single prediction result."""
    inputs: Dict[str, Any]
    prediction: float
    uncertainty: float


class PredictionResponse(BaseModel):
    """Response from prediction endpoint."""
    predictions: List[PredictionResult]
    n_predictions: int
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predictions": [
                    {
                        "inputs": {"temperature": 375, "catalyst": "A"},
                        "prediction": 0.87,
                        "uncertainty": 0.04
                    }
                ],
                "n_predictions": 1
            }
        }
    )


# ============================================================
# Error Models
# ============================================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
    error_type: str
    status_code: int
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": "Session not found",
                "error_type": "SessionNotFoundError",
                "status_code": 404
            }
        }
    )
