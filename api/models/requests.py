"""
Pydantic request models for API endpoints.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional, Literal, Union


# ============================================================
# Variable Models
# ============================================================

class AddRealVariableRequest(BaseModel):
    """Request to add a real-valued variable."""
    name: str = Field(..., description="Variable name")
    type: Literal["real"] = Field(default="real", description="Variable type")
    min: float = Field(..., description="Minimum value")
    max: float = Field(..., description="Maximum value")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    description: Optional[str] = Field(None, description="Variable description")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "temperature",
                "type": "real",
                "min": 300,
                "max": 500,
                "unit": "°C",
                "description": "Reaction temperature"
            }
        }
    )


class AddIntegerVariableRequest(BaseModel):
    """Request to add an integer variable."""
    name: str = Field(..., description="Variable name")
    type: Literal["integer"] = Field(default="integer", description="Variable type")
    min: int = Field(..., description="Minimum value")
    max: int = Field(..., description="Maximum value")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    description: Optional[str] = Field(None, description="Variable description")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "batch_size",
                "type": "integer",
                "min": 1,
                "max": 10,
                "unit": "batches",
                "description": "Number of batches"
            }
        }
    )


class AddCategoricalVariableRequest(BaseModel):
    """Request to add a categorical variable."""
    name: str = Field(..., description="Variable name")
    type: Literal["categorical"] = Field(default="categorical", description="Variable type")
    categories: List[str] = Field(..., description="List of category values")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    description: Optional[str] = Field(None, description="Variable description")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "catalyst",
                "type": "categorical",
                "categories": ["A", "B", "C"],
                "description": "Catalyst type"
            }
        }
    )


# Union type for any variable request
AddVariableRequest = Union[
    AddRealVariableRequest,
    AddIntegerVariableRequest,
    AddCategoricalVariableRequest
]


# ============================================================
# Experiment Models
# ============================================================

class AddExperimentRequest(BaseModel):
    """Request to add a single experiment."""
    inputs: Dict[str, Union[float, int, str]] = Field(..., description="Variable values")
    output: Optional[float] = Field(None, description="Target/output value")
    noise: Optional[float] = Field(None, description="Measurement uncertainty")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "inputs": {"temperature": 350, "catalyst": "A"},
                "output": 0.85,
                "noise": 0.02
            }
        }
    )


class AddExperimentsBatchRequest(BaseModel):
    """Request to add multiple experiments."""
    experiments: List[AddExperimentRequest] = Field(..., description="List of experiments")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "experiments": [
                    {"inputs": {"temperature": 350, "catalyst": "A"}, "output": 0.85},
                    {"inputs": {"temperature": 400, "catalyst": "B"}, "output": 0.92}
                ]
            }
        }
    )


# ============================================================
# Model Training Models
# ============================================================

class TrainModelRequest(BaseModel):
    """Request to train a surrogate model."""
    backend: Literal["sklearn", "botorch"] = Field(default="sklearn", description="Modeling backend")
    kernel: str = Field(default="Matern", description="Kernel type (RBF, Matern, RationalQuadratic)")
    kernel_params: Optional[Dict[str, Any]] = Field(None, description="Kernel-specific parameters")
    input_transform: Optional[str] = Field(None, description="Input transformation (Normalize, Standardize, etc.)")
    output_transform: Optional[str] = Field(None, description="Output transformation (Standardize, etc.)")
    calibration_enabled: bool = Field(default=False, description="Enable uncertainty calibration")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "backend": "sklearn",
                "kernel": "Matern",
                "kernel_params": {"nu": 2.5}
            }
        }
    )


# ============================================================
# Acquisition Models
# ============================================================

class AcquisitionRequest(BaseModel):
    """Request to suggest next experiments."""
    strategy: str = Field(default="EI", description="Acquisition strategy (EI, PI, UCB, qEI, qUCB, qNIPV)")
    goal: Literal["maximize", "minimize"] = Field(default="maximize", description="Optimization goal")
    n_suggestions: int = Field(default=1, ge=1, le=10, description="Number of suggestions (batch size)")
    xi: Optional[float] = Field(default=0.01, description="Exploration parameter for EI/PI")
    kappa: Optional[float] = Field(default=2.0, description="Exploration parameter for UCB")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "strategy": "EI",
                "goal": "maximize",
                "n_suggestions": 1,
                "xi": 0.01
            }
        }
    )


class FindOptimumRequest(BaseModel):
    """Request to find model's predicted optimum."""
    goal: Literal["maximize", "minimize"] = Field(default="maximize", description="Optimization goal")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "goal": "maximize"
            }
        }
    )


# ============================================================
# Prediction Models
# ============================================================

class PredictionRequest(BaseModel):
    """Request to make predictions at new points."""
    inputs: List[Dict[str, Union[float, int, str]]] = Field(..., description="Input points for prediction")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "inputs": [
                    {"temperature": 375, "catalyst": "A"},
                    {"temperature": 425, "catalyst": "B"}
                ]
            }
        }
    )
