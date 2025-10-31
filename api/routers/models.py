"""
Models router - Surrogate model training and prediction.
"""

from fastapi import APIRouter, Depends
from ..models.requests import TrainModelRequest, PredictionRequest
from ..models.responses import TrainModelResponse, ModelInfoResponse, PredictionResponse, PredictionResult
from ..dependencies import get_session
from ..middleware.error_handlers import NoDataError, NoModelError
from alchemist_core.session import OptimizationSession
import logging
import pandas as pd

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/{session_id}/model/train", response_model=TrainModelResponse)
async def train_model(
    session_id: str,
    request: TrainModelRequest,
    session: OptimizationSession = Depends(get_session)
):
    """
    Train a surrogate model on experimental data.
    
    Requires at least 5 experiments with output values.
    Training is synchronous and typically takes a few seconds.
    """
    # Check if data exists
    if session.experiment_manager.df.empty:
        raise NoDataError("No experimental data available. Add experiments first.")
    
    # Train model - map transform parameter names to what the models expect
    results = session.train_model(
        backend=request.backend,
        kernel=request.kernel,
        kernel_params=request.kernel_params,
        input_transform_type=request.input_transform,  # SklearnModel expects _type suffix
        output_transform_type=request.output_transform,  # SklearnModel expects _type suffix
        calibration_enabled=request.calibration_enabled
    )
    
    logger.info(f"Trained {request.backend} model for session {session_id}")
    
    return TrainModelResponse(
        success=results["success"],
        backend=results["backend"],
        kernel=results["kernel"],
        hyperparameters=results["hyperparameters"],
        metrics=results["metrics"],
        message="Model trained successfully"
    )


@router.get("/{session_id}/model", response_model=ModelInfoResponse)
async def get_model_info(
    session_id: str,
    session: OptimizationSession = Depends(get_session)
):
    """
    Get information about the trained model.
    
    Returns model type, hyperparameters, and performance metrics.
    """
    model_summary = session.get_model_summary()
    
    if model_summary is None:
        return ModelInfoResponse(
            backend=None,
            hyperparameters=None,
            metrics=None,
            is_trained=False
        )
    
    return ModelInfoResponse(**model_summary)


@router.post("/{session_id}/model/predict", response_model=PredictionResponse)
async def predict(
    session_id: str,
    request: PredictionRequest,
    session: OptimizationSession = Depends(get_session)
):
    """
    Make predictions at new input points.
    
    Returns predicted mean and uncertainty (standard deviation) for each input.
    """
    # Check if model exists
    if session.model is None:
        raise NoModelError("No trained model available. Train a model first.")
    
    # Convert inputs to DataFrame
    df_inputs = pd.DataFrame(request.inputs)
    
    # Make predictions
    predictions, uncertainties = session.predict(df_inputs)
    
    # Format response
    results = [
        PredictionResult(
            inputs=inputs,
            prediction=float(pred),
            uncertainty=float(uncert)
        )
        for inputs, pred, uncert in zip(request.inputs, predictions, uncertainties)
    ]
    
    logger.info(f"Made {len(results)} predictions for session {session_id}")
    
    return PredictionResponse(
        predictions=results,
        n_predictions=len(results)
    )
