"""
Sessions router - Session lifecycle management.
"""

from fastapi import APIRouter, HTTPException, status, UploadFile, File
from fastapi.responses import Response
from ..models.responses import SessionCreateResponse, SessionInfoResponse
from ..services import session_store
from ..dependencies import get_session
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/sessions", response_model=SessionCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_session():
    """
    Create a new optimization session.
    
    Returns a unique session ID that should be used in subsequent requests.
    Sessions expire after 24 hours of creation.
    """
    session_id = session_store.create()
    session_info = session_store.get_info(session_id)
    
    return SessionCreateResponse(
        session_id=session_id,
        created_at=session_info["created_at"],
        expires_at=session_info["expires_at"]
    )


@router.get("/sessions/{session_id}", response_model=SessionInfoResponse)
async def get_session_info(session_id: str):
    """
    Get information about an optimization session.
    
    Returns session metadata, search space, data summary, and model status.
    """
    info = session_store.get_info(session_id)
    if info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found or expired"
        )
    
    return SessionInfoResponse(**info)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str):
    """
    Delete an optimization session.
    
    Permanently removes the session and all associated data.
    """
    deleted = session_store.delete(session_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    return None


@router.post("/sessions/{session_id}/extend", status_code=status.HTTP_200_OK)
async def extend_session(session_id: str, hours: int = 24):
    """
    Extend session TTL.
    
    Args:
        session_id: Session identifier
        hours: Number of hours to extend (default: 24)
    """
    extended = session_store.extend_ttl(session_id, hours)
    if not extended:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    
    info = session_store.get_info(session_id)
    return {
        "message": "Session TTL extended",
        "expires_at": info["expires_at"]
    }


@router.get("/sessions/{session_id}/export")
async def export_session(session_id: str):
    """
    Export a session for download.
    
    Downloads the complete session state as a .pkl file that can be
    reimported later.
    """
    session_data = session_store.export_session(session_id)
    if session_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    
    return Response(
        content=session_data,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename=session_{session_id}.pkl"
        }
    )


@router.post("/sessions/import", response_model=SessionCreateResponse, status_code=status.HTTP_201_CREATED)
async def import_session(file: UploadFile = File(...)):
    """
    Import a previously exported session.
    
    Uploads a .pkl session file and creates a new session with the imported data.
    A new session ID will be generated.
    """
    try:
        session_data = await file.read()
        session_id = session_store.import_session(session_data)
        
        if session_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid session file or corrupted data"
            )
        
        session_info = session_store.get_info(session_id)
        return SessionCreateResponse(
            session_id=session_id,
            created_at=session_info["created_at"],
            expires_at=session_info["expires_at"]
        )
        
    except Exception as e:
        logger.error(f"Failed to import session: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to import session: {str(e)}"
        )
