"""
Variables router - Search space management.
"""

from fastapi import APIRouter, Depends, UploadFile, File, Form
from typing import Union
from ..models.requests import (
    AddRealVariableRequest,
    AddIntegerVariableRequest,
    AddCategoricalVariableRequest,
)
from ..models.responses import VariableResponse, VariablesListResponse
from ..dependencies import get_session
from ..middleware.error_handlers import NoVariablesError
from alchemist_core.session import OptimizationSession
import logging
import json
import tempfile
import os

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/{session_id}/variables", response_model=VariableResponse)
async def add_variable(
    session_id: str,
    variable: Union[AddRealVariableRequest, AddIntegerVariableRequest, AddCategoricalVariableRequest],
    session: OptimizationSession = Depends(get_session)
):
    """
    Add a variable to the search space.
    
    Supports three types of variables:
    - real: Continuous floating-point values
    - integer: Discrete integer values
    - categorical: Discrete categorical values
    """
    # Extract variable data
    var_dict = variable.model_dump()
    var_type = var_dict.pop("type")
    name = var_dict.pop("name")
    
    # Handle categories → values conversion for categorical
    if "categories" in var_dict:
        var_dict["values"] = var_dict.pop("categories")
    
    # Add variable to session
    session.add_variable(name, var_type, **var_dict)
    
    logger.info(f"Added variable '{name}' ({var_type}) to session {session_id}")
    
    return VariableResponse(
        message="Variable added successfully",
        variable={
            "name": name,
            "type": var_type,
            **var_dict
        }
    )


@router.get("/{session_id}/variables", response_model=VariablesListResponse)
async def list_variables(
    session_id: str,
    session: OptimizationSession = Depends(get_session)
):
    """
    Get all variables in the search space.
    
    Returns list of variables with their types and parameters.
    """
    summary = session.get_search_space_summary()
    
    return VariablesListResponse(
        variables=summary["variables"],
        n_variables=summary["n_variables"]
    )


@router.post("/{session_id}/variables/load")
async def load_variables_from_file(
    session_id: str,
    file: UploadFile = File(...),
    session: OptimizationSession = Depends(get_session)
):
    """
    Load search space definition from JSON file.
    
    Expected JSON format:
    [
        {"name": "temp", "type": "real", "min": 300, "max": 500},
        {"name": "catalyst", "type": "categorical", "categories": ["A", "B", "C"]}
    ]
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.json') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Load and parse JSON
        with open(tmp_path, 'r') as f:
            variables_data = json.load(f)
        
        # Add each variable
        for var in variables_data:
            var_type = var.pop("type")
            name = var.pop("name")
            
            # Handle categories for categorical variables
            if "categories" in var:
                var["values"] = var.pop("categories")
            
            session.add_variable(name, var_type, **var)
        
        logger.info(f"Loaded {len(variables_data)} variables from file for session {session_id}")
        
        return {
            "message": f"Loaded {len(variables_data)} variables successfully",
            "n_variables": len(variables_data)
        }
        
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
