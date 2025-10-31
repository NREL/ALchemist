"""
Experiments router - Experimental data management.
"""

from fastapi import APIRouter, Depends, UploadFile, File
from ..models.requests import AddExperimentRequest, AddExperimentsBatchRequest
from ..models.responses import ExperimentResponse, ExperimentsListResponse, ExperimentsSummaryResponse
from ..dependencies import get_session
from ..middleware.error_handlers import NoVariablesError
from alchemist_core.session import OptimizationSession
import logging
import pandas as pd
import tempfile
import os

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/{session_id}/experiments", response_model=ExperimentResponse)
async def add_experiment(
    session_id: str,
    experiment: AddExperimentRequest,
    session: OptimizationSession = Depends(get_session)
):
    """
    Add a single experiment to the dataset.
    
    The experiment must include values for all defined variables.
    Output value is optional for candidate experiments.
    """
    # Check if variables are defined
    if len(session.search_space.variables) == 0:
        raise NoVariablesError("No variables defined. Add variables to search space first.")
    
    session.add_experiment(
        inputs=experiment.inputs,
        output=experiment.output,
        noise=experiment.noise
    )
    
    n_experiments = len(session.experiment_manager.df)
    logger.info(f"Added experiment to session {session_id}. Total: {n_experiments}")
    
    return ExperimentResponse(
        message="Experiment added successfully",
        n_experiments=n_experiments
    )


@router.post("/{session_id}/experiments/batch", response_model=ExperimentResponse)
async def add_experiments_batch(
    session_id: str,
    batch: AddExperimentsBatchRequest,
    session: OptimizationSession = Depends(get_session)
):
    """
    Add multiple experiments at once.
    
    Useful for bulk data import or initialization.
    """
    # Check if variables are defined
    if len(session.search_space.variables) == 0:
        raise NoVariablesError("No variables defined. Add variables to search space first.")
    
    for exp in batch.experiments:
        session.add_experiment(
            inputs=exp.inputs,
            output=exp.output,
            noise=exp.noise
        )
    
    n_experiments = len(session.experiment_manager.df)
    logger.info(f"Added {len(batch.experiments)} experiments to session {session_id}. Total: {n_experiments}")
    
    return ExperimentResponse(
        message=f"Added {len(batch.experiments)} experiments successfully",
        n_experiments=n_experiments
    )


@router.get("/{session_id}/experiments", response_model=ExperimentsListResponse)
async def list_experiments(
    session_id: str,
    session: OptimizationSession = Depends(get_session)
):
    """
    Get all experiments in the dataset.
    
    Returns complete experimental data including inputs, outputs, and noise values.
    """
    df = session.experiment_manager.get_data()
    experiments = df.to_dict('records')
    
    return ExperimentsListResponse(
        experiments=experiments,
        n_experiments=len(experiments)
    )


@router.post("/{session_id}/experiments/upload")
async def upload_experiments(
    session_id: str,
    file: UploadFile = File(...),
    target_column: str = "Output",
    session: OptimizationSession = Depends(get_session)
):
    """
    Upload experimental data from CSV file.
    
    The CSV should have columns matching the variable names,
    plus an optional output column (default: "Output") and
    optional noise column ("Noise").
    """
    # Check if variables are defined
    if len(session.search_space.variables) == 0:
        raise NoVariablesError("No variables defined. Add variables to search space first.")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Load data using session's load_data method
        session.load_data(tmp_path, target_column=target_column)
        
        n_experiments = len(session.experiment_manager.df)
        logger.info(f"Loaded {n_experiments} experiments from CSV for session {session_id}")
        
        return {
            "message": f"Loaded {n_experiments} experiments successfully",
            "n_experiments": n_experiments
        }
        
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.get("/{session_id}/experiments/summary", response_model=ExperimentsSummaryResponse)
async def get_experiments_summary(
    session_id: str,
    session: OptimizationSession = Depends(get_session)
):
    """
    Get statistical summary of experimental data.
    
    Returns sample size, target variable statistics, and feature information.
    """
    return session.get_data_summary()
