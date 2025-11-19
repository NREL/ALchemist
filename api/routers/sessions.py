"""
Sessions router - Session lifecycle management.
"""

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Depends
from fastapi.responses import Response, FileResponse, JSONResponse
from ..models.requests import UpdateMetadataRequest, LockDecisionRequest
from ..models.responses import (
    SessionCreateResponse, SessionInfoResponse, SessionStateResponse,
    SessionMetadataResponse, AuditLogResponse, AuditEntryResponse, LockDecisionResponse
)
from ..services import session_store
from ..dependencies import get_session
from alchemist_core.session import OptimizationSession
from datetime import datetime
import logging
import json
from pathlib import Path
import tempfile

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


@router.get("/sessions/{session_id}/state", response_model=SessionStateResponse)
async def get_session_state(
    session_id: str,
    session: OptimizationSession = Depends(get_session)
):
    """
    Get current session state for monitoring autonomous optimization.
    
    Returns key metrics for dashboard displays or autonomous controllers
    to monitor optimization progress without retrieving full session data.
    """
    # Get session metrics
    n_variables = len(session.search_space.variables)
    n_experiments = len(session.experiment_manager.df)
    model_trained = session.model is not None
    
    # Get last suggestion if available
    last_suggestion = None
    if hasattr(session, '_last_suggestion') and session._last_suggestion:
        last_suggestion = session._last_suggestion
    
    return SessionStateResponse(
        session_id=session_id,
        n_variables=n_variables,
        n_experiments=n_experiments,
        model_trained=model_trained,
        last_suggestion=last_suggestion
    )


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


# ============================================================
# Metadata Management Endpoints
# ============================================================

@router.get("/sessions/{session_id}/metadata", response_model=SessionMetadataResponse)
async def get_metadata(
    session_id: str,
    session: OptimizationSession = Depends(get_session)
):
    """
    Get session metadata.
    
    Returns the session's user-friendly name, description, tags, and timestamps.
    """
    return SessionMetadataResponse(
        session_id=session.metadata.session_id,
        name=session.metadata.name,
        created_at=session.metadata.created_at,
        last_modified=session.metadata.last_modified,
        description=session.metadata.description,
        tags=session.metadata.tags
    )


@router.patch("/sessions/{session_id}/metadata", response_model=SessionMetadataResponse)
async def update_metadata(
    session_id: str,
    request: UpdateMetadataRequest,
    session: OptimizationSession = Depends(get_session)
):
    """
    Update session metadata.
    
    Update the session's name, description, and/or tags. Only provided fields
    will be updated; omitted fields remain unchanged.
    """
    session.update_metadata(
        name=request.name,
        description=request.description,
        tags=request.tags
    )
    
    return SessionMetadataResponse(
        session_id=session.metadata.session_id,
        name=session.metadata.name,
        created_at=session.metadata.created_at,
        last_modified=session.metadata.last_modified,
        description=session.metadata.description,
        tags=session.metadata.tags
    )


# ============================================================
# Audit Log Endpoints
# ============================================================

@router.get("/sessions/{session_id}/audit", response_model=AuditLogResponse)
async def get_audit_log(
    session_id: str,
    entry_type: str = None,
    session: OptimizationSession = Depends(get_session)
):
    """
    Get audit log entries.
    
    Retrieves the complete audit trail or filters by entry type.
    
    Args:
        session_id: Session identifier
        entry_type: Optional filter ('data_locked', 'model_locked', 'acquisition_locked')
    """
    if entry_type:
        entries = session.audit_log.get_entries(entry_type)
    else:
        entries = session.audit_log.get_entries()
    
    return AuditLogResponse(
        entries=[AuditEntryResponse(**e.to_dict()) for e in entries],
        n_entries=len(entries)
    )


@router.post("/sessions/{session_id}/audit/lock", response_model=LockDecisionResponse)
async def lock_decision(
    session_id: str,
    request: LockDecisionRequest,
    session: OptimizationSession = Depends(get_session)
):
    """
    Lock in a decision to the audit log.
    
    Creates an immutable audit entry for data, model, or acquisition decisions.
    This should be called when the user is satisfied with their configuration
    and ready to commit the decision to the audit trail.
    
    Args:
        session_id: Session identifier
        request: Lock decision request
    """
    try:
        if request.lock_type == "data":
            entry = session.lock_data(notes=request.notes or "")
            message = "Data decision locked successfully"
            
        elif request.lock_type == "model":
            entry = session.lock_model(notes=request.notes or "")
            message = "Model decision locked successfully"
            
        elif request.lock_type == "acquisition":
            if not request.strategy or not request.parameters or not request.suggestions:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Acquisition lock requires strategy, parameters, and suggestions"
                )
            entry = session.lock_acquisition(
                strategy=request.strategy,
                parameters=request.parameters,
                suggestions=request.suggestions,
                notes=request.notes or ""
            )
            message = "Acquisition decision locked successfully"
            
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid lock_type: {request.lock_type}"
            )
        
        return LockDecisionResponse(
            success=True,
            entry=AuditEntryResponse(**entry.to_dict()),
            message=message
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/sessions/{session_id}/audit/export")
async def export_audit_markdown(
    session_id: str,
    session: OptimizationSession = Depends(get_session)
):
    """
    Export audit log as markdown.
    
    Returns the audit trail formatted as markdown for publication methods sections.
    """
    markdown = session.export_audit_markdown()
    
    return Response(
        content=markdown,
        media_type="text/markdown",
        headers={
            "Content-Disposition": f"attachment; filename=audit_log_{session_id}.md"
        }
    )


# ============================================================
# Session File Management (JSON Format)
# ============================================================

@router.get("/sessions/{session_id}/download")
async def download_session(
    session_id: str,
    session: OptimizationSession = Depends(get_session)
):
    """
    Download session as JSON file.
    
    Downloads the complete session state as a .json file with user-friendly
    naming support. The file includes metadata, audit log, search space,
    experiments, and configuration.
    """
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
        
    try:
        # Save session to temp file
        session.save_session(temp_path)
        
        # Use session name for filename (sanitized)
        filename = session.metadata.name.replace(" ", "_").replace("/", "_")
        filename = f"{filename}.json"
        
        return FileResponse(
            path=temp_path,
            media_type="application/json",
            filename=filename,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        # Clean up temp file on error
        Path(temp_path).unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export session: {str(e)}"
        )


@router.post("/sessions/upload", response_model=SessionCreateResponse, status_code=status.HTTP_201_CREATED)
async def upload_session(file: UploadFile = File(...)):
    """
    Upload and restore a session from JSON file.
    
    Uploads a .json session file and creates a new session with the restored data.
    A new session ID will be generated for API use, but the original session ID
    is preserved in the metadata.
    """
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.json', delete=False) as f:
            content = await file.read()
            f.write(content)
            temp_path = f.name
        
        try:
            # Load session from file
            loaded_session = OptimizationSession.load_session(temp_path)
            
            # Create new session in store
            new_session_id = session_store.create()
            
            # Replace the session object with loaded one
            session_store._sessions[new_session_id]["session"] = loaded_session
            
            # Update last accessed
            session_store._sessions[new_session_id]["last_accessed"] = datetime.now()
            
            # Persist to disk
            session_store._save_to_disk(new_session_id)
            
            session_info = session_store.get_info(new_session_id)
            
            return SessionCreateResponse(
                session_id=new_session_id,
                created_at=session_info["created_at"],
                expires_at=session_info["expires_at"]
            )
            
        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)
            
    except Exception as e:
        logger.error(f"Failed to upload session: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to upload session: {str(e)}"
        )
