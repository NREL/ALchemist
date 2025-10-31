"""
Acquisition router - Next experiment suggestions.
"""

from fastapi import APIRouter, Depends
from ..models.requests import AcquisitionRequest
from ..models.responses import AcquisitionResponse
from ..dependencies import get_session
from ..middleware.error_handlers import NoModelError
from alchemist_core.session import OptimizationSession
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/{session_id}/acquisition/suggest", response_model=AcquisitionResponse)
async def suggest_next_experiments(
    session_id: str,
    request: AcquisitionRequest,
    session: OptimizationSession = Depends(get_session)
):
    """
    Suggest next experiments using acquisition function.
    
    Requires a trained model. Returns one or more suggested experiments
    based on the acquisition strategy and batch size.
    
    Common strategies:
    - EI (Expected Improvement): Balances exploration and exploitation
    - PI (Probability of Improvement): More conservative than EI
    - UCB (Upper Confidence Bound): More exploratory
    - qEI, qUCB: Batch versions for parallel experiments
    - qNIPV: Pure exploration for model improvement
    """
    # Check if model exists
    if session.model is None:
        raise NoModelError("No trained model available. Train a model first.")
    
    # Build kwargs for acquisition function
    acq_kwargs = {}
    if request.xi is not None:
        acq_kwargs['xi'] = request.xi
    if request.kappa is not None:
        acq_kwargs['kappa'] = request.kappa
    
    # Get suggestions
    suggestions_df = session.suggest_next(
        strategy=request.strategy,
        goal=request.goal,
        n_suggestions=request.n_suggestions,
        **acq_kwargs
    )
    
    # Convert to list of dicts
    suggestions = suggestions_df.to_dict('records')
    
    logger.info(f"Generated {len(suggestions)} suggestions for session {session_id} using {request.strategy}")
    
    return AcquisitionResponse(
        suggestions=suggestions,
        n_suggestions=len(suggestions)
    )
