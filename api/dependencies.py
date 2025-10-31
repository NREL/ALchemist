"""
Shared dependencies for API endpoints.
"""

from fastapi import Depends, HTTPException, status
from alchemist_core.session import OptimizationSession
from .services import session_store
from .middleware.error_handlers import SessionNotFoundError
import logging

logger = logging.getLogger(__name__)


def get_session(session_id: str) -> OptimizationSession:
    """
    Dependency to get an optimization session by ID.
    
    Args:
        session_id: Session identifier
        
    Returns:
        OptimizationSession instance
        
    Raises:
        SessionNotFoundError: If session not found or expired
    """
    session = session_store.get(session_id)
    if session is None:
        raise SessionNotFoundError(f"Session {session_id} not found or expired")
    return session


def get_session_or_none(session_id: str) -> OptimizationSession | None:
    """
    Get session without raising error if not found.
    
    Args:
        session_id: Session identifier
        
    Returns:
        OptimizationSession or None
    """
    return session_store.get(session_id)
