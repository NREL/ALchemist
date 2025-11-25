"""
Session Store - Session management with disk persistence.

Stores OptimizationSession instances with TTL and automatic cleanup.
Sessions are persisted to disk as JSON to survive server restarts.
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
import uuid
from alchemist_core.session import OptimizationSession
import logging
import json
import tempfile
from pathlib import Path
import threading

# TODO: Consider migrating per-session `threading.Lock()` to an async-compatible
# `anyio.Lock()` (or `asyncio.Lock`) for cleaner async endpoint integration.
#
# Rationale / next steps:
# - Many API endpoints are `async def` and blocking the event loop with
#   `threading.Lock().acquire()` is undesirable.
# - A migration plan is in `memory/SESSION_LOCKING_ASYNC_PLAN.md` describing
#   how to transition to `anyio.Lock()` and update handlers to use `async with`.

logger = logging.getLogger(__name__)


class SessionStore:
    """Session store with disk persistence."""
    
    def __init__(self, default_ttl_hours: int = 24, persist_dir: Optional[str] = None):
        """
        Initialize session store.
        
        Args:
            default_ttl_hours: Default time-to-live for sessions in hours
            persist_dir: Directory to persist sessions (None = memory only)
        """
        self._sessions: Dict[str, Dict] = {}
        self.default_ttl = timedelta(hours=default_ttl_hours)
        self.persist_dir = Path(persist_dir) if persist_dir else Path("cache/sessions")
        
        # Create persistence directory
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            # Load existing sessions from disk
            self._load_from_disk()
        
        logger.info(f"SessionStore initialized with TTL={default_ttl_hours}h, persist_dir={self.persist_dir}")
    
    def _get_session_file(self, session_id: str) -> Path:
        """Get path to session file."""
        return self.persist_dir / f"{session_id}.json"
    
    def _save_to_disk(self, session_id: str):
        """Save session to disk as JSON."""
        if not self.persist_dir:
            return
        
        try:
            session_file = self._get_session_file(session_id)
            session_data = self._sessions[session_id]
            
            # Create a temporary file for the session
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                session_data["session"].save_session(tmp.name)
                temp_path = tmp.name
            
            # Store metadata alongside session
            metadata = {
                "created_at": session_data["created_at"].isoformat(),
                "last_accessed": session_data["last_accessed"].isoformat(),
                "expires_at": session_data["expires_at"].isoformat()
            }
            
            # Load session JSON and add metadata
            with open(temp_path, 'r') as f:
                session_json = json.load(f)
            
            session_json["_session_store_metadata"] = metadata
            
            # Write combined data
            with open(session_file, 'w') as f:
                json.dump(session_json, f, indent=2)
            
            # Clean up temp file
            Path(temp_path).unlink()
            
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
    
    def _load_from_disk(self):
        """Load all sessions from disk."""
        if not self.persist_dir or not self.persist_dir.exists():
            return
        
        loaded_count = 0
        for session_file in self.persist_dir.glob("*.json"):
            try:
                with open(session_file, 'r') as f:
                    session_json = json.load(f)
                
                # Extract metadata
                metadata = session_json.pop("_session_store_metadata", {})
                
                # Check if expired
                if metadata:
                    expires_at = datetime.fromisoformat(metadata["expires_at"])
                    if datetime.now() > expires_at:
                        session_file.unlink()  # Delete expired session file
                        continue
                
                # Write session data to temp file and load
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                    json.dump(session_json, tmp, indent=2)
                    temp_path = tmp.name
                
                # Load without retraining by default during startup
                session = OptimizationSession.load_session(temp_path, retrain_on_load=False)
                Path(temp_path).unlink()
                session_id = session_file.stem
                self._sessions[session_id] = {
                    "session": session,
                    "created_at": datetime.fromisoformat(metadata.get("created_at", datetime.now().isoformat())),
                    "last_accessed": datetime.fromisoformat(metadata.get("last_accessed", datetime.now().isoformat())),
                    "expires_at": datetime.fromisoformat(metadata.get("expires_at", (datetime.now() + self.default_ttl).isoformat())),
                    "lock": threading.Lock()
                }
                loaded_count += 1
                
            except Exception as e:
                logger.error(f"Failed to load session from {session_file}: {e}")
        
        if loaded_count > 0:
            logger.info(f"Loaded {loaded_count} sessions from disk")
    
    def _delete_from_disk(self, session_id: str):
        """Delete session file from disk."""
        if not self.persist_dir:
            return
        
        try:
            session_file = self._get_session_file(session_id)
            if session_file.exists():
                session_file.unlink()
        except Exception as e:
            logger.error(f"Failed to delete session file {session_id}: {e}")
    
    def create(self, name: Optional[str] = None, description: Optional[str] = None, tags: Optional[list] = None) -> str:
        """
        Create a new session.
        
        Returns:
            session_id: Unique session identifier
        """
        session_id = str(uuid.uuid4())
        session = OptimizationSession()
        # Ensure session metadata matches store id
        try:
            session.metadata.session_id = session_id
        except Exception:
            pass
        # Populate optional metadata
        if name:
            session.metadata.name = name
        if description:
            session.metadata.description = description
        if tags:
            try:
                session.metadata.tags = tags
            except Exception:
                pass

        self._sessions[session_id] = {
            "session": session,
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
            "expires_at": datetime.now() + self.default_ttl,
            "lock": threading.Lock()
        }
        
        # Persist to disk
        self._save_to_disk(session_id)
        
        logger.info(f"Created session {session_id}")
        return session_id
    
    def get(self, session_id: str) -> Optional[OptimizationSession]:
        """
        Get session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            OptimizationSession or None if not found/expired
        """
        # Clean up expired sessions first
        self._cleanup_expired()
        
        if session_id not in self._sessions:
            logger.warning(f"Session {session_id} not found")
            return None
        
        session_data = self._sessions[session_id]
        lock = session_data.get("lock")
        if lock:
            with lock:
                # Check if expired
                if datetime.now() > session_data["expires_at"]:
                    logger.info(f"Session {session_id} expired, removing")
                    del self._sessions[session_id]
                    return None

                # Update last accessed time
                session_data["last_accessed"] = datetime.now()

                # Save updated access time to disk
                self._save_to_disk(session_id)

                return session_data["session"]
        else:
            # Fallback (no lock present)
            if datetime.now() > session_data["expires_at"]:
                logger.info(f"Session {session_id} expired, removing")
                del self._sessions[session_id]
                return None
            session_data["last_accessed"] = datetime.now()
            self._save_to_disk(session_id)
            return session_data["session"]
    
    def delete(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted, False if not found
        """
        if session_id in self._sessions:
            lock = self._sessions[session_id].get("lock")
            if lock:
                with lock:
                    del self._sessions[session_id]
                    self._delete_from_disk(session_id)
            else:
                del self._sessions[session_id]
                self._delete_from_disk(session_id)
            logger.info(f"Deleted session {session_id}")
            return True
        return False
    
    def get_info(self, session_id: str) -> Optional[Dict]:
        """
        Get session metadata.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with session info or None
        """
        if session_id not in self._sessions:
            return None
        
        session_data = self._sessions[session_id]
        session = session_data["session"]
        
        return {
            "session_id": session_id,
            "created_at": session_data["created_at"].isoformat(),
            "last_accessed": session_data["last_accessed"].isoformat(),
            "expires_at": session_data["expires_at"].isoformat(),
            "search_space": session.get_search_space_summary(),
            "data": session.get_data_summary(),
            "model": session.get_model_summary()
        }
    
    def extend_ttl(self, session_id: str, hours: int = None) -> bool:
        """
        Extend session TTL.
        
        Args:
            session_id: Session identifier
            hours: Hours to extend (uses default if None)
            
        Returns:
            True if extended, False if session not found
        """
        if session_id not in self._sessions:
            return False
        lock = self._sessions[session_id].get("lock")
        if lock:
            with lock:
                extension = timedelta(hours=hours) if hours else self.default_ttl
                self._sessions[session_id]["expires_at"] = datetime.now() + extension
                self._save_to_disk(session_id)
        else:
            extension = timedelta(hours=hours) if hours else self.default_ttl
            self._sessions[session_id]["expires_at"] = datetime.now() + extension
            self._save_to_disk(session_id)

        logger.info(f"Extended TTL for session {session_id}")
        return True
    
    def _cleanup_expired(self):
        """Remove expired sessions."""
        now = datetime.now()
        expired = [
            sid for sid, data in self._sessions.items()
            if now > data["expires_at"]
        ]
        
        for sid in expired:
            del self._sessions[sid]
            self._delete_from_disk(sid)
            logger.info(f"Cleaned up expired session {sid}")
    
    def count(self) -> int:
        """Get count of active sessions."""
        self._cleanup_expired()
        return len(self._sessions)
    
    def list_all(self) -> list:
        """Get list of all active session IDs."""
        self._cleanup_expired()
        return list(self._sessions.keys())
    
    def export_session(self, session_id: str) -> Optional[str]:
        """
        Export a session as JSON string for download.
        
        Args:
            session_id: Session identifier
            
        Returns:
            JSON string of session data or None if not found
        """
        if session_id not in self._sessions:
            return None

        try:
            lock = self._sessions[session_id].get("lock")
            if lock:
                with lock:
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                        self._sessions[session_id]["session"].save_session(tmp.name)
                        temp_path = tmp.name

                    # Read the JSON content
                    with open(temp_path, 'r') as f:
                        json_content = f.read()

                    # Clean up temp file
                    Path(temp_path).unlink()
                    return json_content
            else:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                    self._sessions[session_id]["session"].save_session(tmp.name)
                    temp_path = tmp.name

                with open(temp_path, 'r') as f:
                    json_content = f.read()

                Path(temp_path).unlink()
                return json_content
        except Exception as e:
            logger.error(f"Failed to export session {session_id}: {e}")
            return None

    def persist_session_to_disk(self, session_id: str) -> bool:
        """
        Persist the in-memory session to disk (overwrite existing persisted file).

        Returns True on success, False otherwise.
        """
        if session_id not in self._sessions:
            return False
        try:
            lock = self._sessions[session_id].get('lock')
            if lock:
                with lock:
                    self._save_to_disk(session_id)
            else:
                self._save_to_disk(session_id)
            return True
        except Exception as e:
            logger.error(f"Failed to persist session {session_id}: {e}")
            return False
    
    def import_session(self, session_data: str, session_id: Optional[str] = None) -> Optional[str]:
        """
        Import a session from JSON string.
        
        Args:
            session_data: JSON string of session data
            session_id: Optional custom session ID (generates new one if None)
            
        Returns:
            Session ID or None if import failed
        """
        try:
            # Write JSON to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                tmp.write(session_data)
                temp_path = tmp.name
            
            # Load session without automatic retrain
            session = OptimizationSession.load_session(temp_path, retrain_on_load=False)
            Path(temp_path).unlink()
            
            # Generate new session ID if not provided
            if not session_id:
                session_id = str(uuid.uuid4())

            # Ensure session metadata session_id matches store id
            try:
                session.metadata.session_id = session_id
            except Exception:
                pass

            # Store session with metadata and lock
            self._sessions[session_id] = {
                "session": session,
                "created_at": datetime.now(),
                "last_accessed": datetime.now(),
                "expires_at": datetime.now() + self.default_ttl,
                "lock": threading.Lock()
            }
            
            self._save_to_disk(session_id)
            
            logger.info(f"Imported session {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to import session: {e}")
            return None


# Global session store instance
session_store = SessionStore(default_ttl_hours=24)
