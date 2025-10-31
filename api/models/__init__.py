"""API models initialization."""

from .requests import (
    AddVariableRequest,
    AddRealVariableRequest,
    AddIntegerVariableRequest,
    AddCategoricalVariableRequest,
    AddExperimentRequest,
    AddExperimentsBatchRequest,
    TrainModelRequest,
    AcquisitionRequest,
    PredictionRequest,
)

from .responses import (
    SessionCreateResponse,
    SessionInfoResponse,
    VariableResponse,
    VariablesListResponse,
    ExperimentResponse,
    ExperimentsListResponse,
    TrainModelResponse,
    ModelInfoResponse,
    AcquisitionResponse,
    PredictionResponse,
    ErrorResponse,
)

__all__ = [
    # Requests
    "AddVariableRequest",
    "AddRealVariableRequest",
    "AddIntegerVariableRequest",
    "AddCategoricalVariableRequest",
    "AddExperimentRequest",
    "AddExperimentsBatchRequest",
    "TrainModelRequest",
    "AcquisitionRequest",
    "PredictionRequest",
    # Responses
    "SessionCreateResponse",
    "SessionInfoResponse",
    "VariableResponse",
    "VariablesListResponse",
    "ExperimentResponse",
    "ExperimentsListResponse",
    "TrainModelResponse",
    "ModelInfoResponse",
    "AcquisitionResponse",
    "PredictionResponse",
    "ErrorResponse",
]
