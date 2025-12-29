"""
Experiments router - Experimental data management.
"""

from fastapi import APIRouter, Depends, UploadFile, File, Query
from ..models.requests import AddExperimentRequest, AddExperimentsBatchRequest, InitialDesignRequest
from ..models.responses import (
    ExperimentResponse, 
    ExperimentsListResponse, 
    ExperimentsSummaryResponse,
    InitialDesignResponse
)
from ..dependencies import get_session
from ..middleware.error_handlers import NoVariablesError
from alchemist_core.session import OptimizationSession
import logging
import pandas as pd
import tempfile
import os
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/{session_id}/experiments", response_model=ExperimentResponse)
async def add_experiment(
    session_id: str,
    experiment: AddExperimentRequest,
    auto_train: bool = Query(False, description="Auto-train model after adding data"),
    training_backend: Optional[str] = Query(None, description="Model backend (sklearn/botorch)"),
    training_kernel: Optional[str] = Query(None, description="Kernel type (rbf/matern)"),
    session: OptimizationSession = Depends(get_session)
):
    """
    Add a single experiment to the dataset.
    
    The experiment must include values for all defined variables.
    Output value is optional for candidate experiments.
    
    Args:
        auto_train: If True, retrain model after adding data
        training_backend: Model backend (uses last if None)
        training_kernel: Kernel type (uses last or 'rbf' if None)
    """
    # Check if variables are defined
    if len(session.search_space.variables) == 0:
        raise NoVariablesError("No variables defined. Add variables to search space first.")
    
    session.add_experiment(
        inputs=experiment.inputs,
        output=experiment.output,
        noise=experiment.noise,
        iteration=experiment.iteration,
        reason=experiment.reason
    )
    
    n_experiments = len(session.experiment_manager.df)
    logger.info(f"Added experiment to session {session_id}. Total: {n_experiments}")
    
    # Auto-train if requested (need at least 5 points to train)
    model_trained = False
    training_metrics = None
    
    if auto_train and n_experiments >= 5:
        try:
            # Use previous config or provided config
            backend = training_backend or (session.model_backend if session.model else "sklearn")
            kernel = training_kernel or "rbf"
            
            # Note: Input/output transforms are now automatically applied by core Session.train_model()
            # for BoTorch models. No need to specify them here unless overriding defaults.
            result = session.train_model(backend=backend, kernel=kernel)
            model_trained = True
            metrics = result.get("metrics", {})
            hyperparameters = result.get("hyperparameters", {})
            training_metrics = {
                "rmse": metrics.get("rmse"),
                "r2": metrics.get("r2"),
                "backend": backend
            }
            logger.info(f"Auto-trained model for session {session_id}: {training_metrics}")
            
            # Record in audit log if this is an optimization iteration
            if experiment.iteration is not None and experiment.iteration > 0:
                session.audit_log.lock_model(
                    backend=backend,
                    kernel=kernel,
                    hyperparameters=hyperparameters,
                    cv_metrics=metrics,
                    iteration=experiment.iteration,
                    notes=f"Auto-trained after iteration {experiment.iteration}"
                )
        except Exception as e:
            logger.error(f"Auto-train failed for session {session_id}: {e}")
            # Don't fail the whole request, just log it
    
    return ExperimentResponse(
        message="Experiment added successfully",
        n_experiments=n_experiments,
        model_trained=model_trained,
        training_metrics=training_metrics
    )


@router.post("/{session_id}/experiments/batch", response_model=ExperimentResponse)
async def add_experiments_batch(
    session_id: str,
    batch: AddExperimentsBatchRequest,
    auto_train: bool = Query(False, description="Auto-train model after adding data"),
    training_backend: Optional[str] = Query(None, description="Model backend (sklearn/botorch)"),
    training_kernel: Optional[str] = Query(None, description="Kernel type (rbf/matern)"),
    session: OptimizationSession = Depends(get_session)
):
    """
    Add multiple experiments at once.
    
    Useful for bulk data import or initialization.
    
    Args:
        auto_train: If True, retrain model after adding data
        training_backend: Model backend (uses last if None)
        training_kernel: Kernel type (uses last or 'rbf' if None)
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
    
    # Auto-train if requested
    model_trained = False
    training_metrics = None
    
    if auto_train and n_experiments >= 5:  # Minimum data for training
        try:
            backend = training_backend or (session.model_backend if session.model else "sklearn")
            kernel = training_kernel or "rbf"
            
            result = session.train_model(backend=backend, kernel=kernel)
            model_trained = True
            metrics = result.get("metrics", {})
            training_metrics = {
                "rmse": metrics.get("rmse"),
                "r2": metrics.get("r2"),
                "backend": backend
            }
            logger.info(f"Auto-trained model for session {session_id}: {training_metrics}")
        except Exception as e:
            logger.error(f"Auto-train failed for session {session_id}: {e}")
    
    return ExperimentResponse(
        message=f"Added {len(batch.experiments)} experiments successfully",
        n_experiments=n_experiments,
        model_trained=model_trained,
        training_metrics=training_metrics
    )


@router.post("/{session_id}/initial-design", response_model=InitialDesignResponse)
async def generate_initial_design(
    session_id: str,
    request: InitialDesignRequest,
    session: OptimizationSession = Depends(get_session)
):
    """
    Generate initial experimental design (DoE) for autonomous operation.
    
    Generates space-filling experimental designs before Bayesian optimization begins.
    Useful for autonomous controllers to get initial points to evaluate.
    
    Methods:
    - random: Random sampling
    - lhs: Latin Hypercube Sampling (space-filling)
    - sobol: Sobol sequence (quasi-random)
    - halton: Halton sequence (quasi-random)
    - hammersly: Hammersly sequence (quasi-random)
    
    Returns list of experiments (input combinations) to evaluate.
    """
    # Check if variables are defined
    if len(session.search_space.variables) == 0:
        raise NoVariablesError("No variables defined. Add variables to search space first.")
    
    # Generate design
    design_points = session.generate_initial_design(
        method=request.method,
        n_points=request.n_points,
        random_seed=request.random_seed,
        lhs_criterion=request.lhs_criterion
    )
    
    logger.info(f"Generated {len(design_points)} initial design points using {request.method} for session {session_id}")
    
    return InitialDesignResponse(
        points=design_points,
        method=request.method,
        n_points=len(design_points)
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


@router.post("/{session_id}/experiments/preview")
async def preview_csv_columns(
    session_id: str,
    file: UploadFile = File(...),
    session: OptimizationSession = Depends(get_session)
):
    """
    Preview CSV file columns before uploading to check for target columns.
    
    Returns:
        - available_columns: List of all columns in CSV
        - has_output: Whether 'Output' column exists
        - recommended_target: Suggested target column if 'Output' missing
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Read CSV to get column names
        df = pd.read_csv(tmp_path)
        columns = df.columns.tolist()
        
        # Check for 'Output' column
        has_output = 'Output' in columns
        
        # Filter out metadata columns
        metadata_cols = {'Iteration', 'Reason', 'Noise'}
        available_targets = [col for col in columns if col not in metadata_cols]
        
        # Recommend target column
        recommended = None
        if not has_output:
            # Look for common target column names
            common_names = ['output', 'y', 'target', 'yield', 'response']
            for name in common_names:
                if name in [col.lower() for col in available_targets]:
                    recommended = [col for col in available_targets if col.lower() == name][0]
                    break
            
            # If no common name found, use first numeric column
            if not recommended and available_targets:
                # Check if first available column is numeric
                if pd.api.types.is_numeric_dtype(df[available_targets[0]]):
                    recommended = available_targets[0]
        
        return {
            "columns": columns,
            "available_targets": available_targets,
            "has_output": has_output,
            "recommended_target": recommended,
            "n_rows": len(df)
        }
        
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post("/{session_id}/experiments/upload")
async def upload_experiments(
    session_id: str,
    file: UploadFile = File(...),
    target_columns: str = "Output",  # Note: API accepts string, will be normalized by Session API
    session: OptimizationSession = Depends(get_session)
):
    """
    Upload experimental data from CSV file.
    
    The CSV should have columns matching the variable names,
    plus target column(s) (default: "Output") and optional noise column ("Noise").
    
    Args:
        target_columns: Target column name (single-objective) or comma-separated names (multi-objective).
                       Examples: "Output", "yield", "yield,selectivity"
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
        # Parse target_columns (handle comma-separated for future multi-objective support)
        target_cols_parsed = target_columns.split(',') if ',' in target_columns else target_columns
        
        # Load data using session's load_data method
        session.load_data(tmp_path, target_columns=target_cols_parsed)
        
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
