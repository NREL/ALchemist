"""
Session Store - Session management with disk persistence.

Stores OptimizationSession instances with TTL and automatic cleanup.
Sessions are persisted to disk to survive server restarts.
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
import uuid
from alchemist_core.session import OptimizationSession
import logging
import pickle
from pathlib import Path

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
        return self.persist_dir / f"{session_id}.pkl"
    
    def _save_to_disk(self, session_id: str):
        """Save session to disk."""
        if not self.persist_dir:
            return
        
        try:
            session_file = self._get_session_file(session_id)
            with open(session_file, 'wb') as f:
                pickle.dump(self._sessions[session_id], f)
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
    
    def _load_from_disk(self):
        """Load all sessions from disk."""
        if not self.persist_dir or not self.persist_dir.exists():
            return
        
        loaded_count = 0
        for session_file in self.persist_dir.glob("*.pkl"):
            try:
                with open(session_file, 'rb') as f:
                    session_data = pickle.load(f)
                    session_id = session_file.stem
                    
                    # Check if expired
                    if datetime.now() > session_data["expires_at"]:
                        session_file.unlink()  # Delete expired session file
                        continue
                    
                    self._sessions[session_id] = session_data
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
    
    def create(self) -> str:
        """
        Create a new session.
        
        Returns:
            session_id: Unique session identifier
        """
        session_id = str(uuid.uuid4())
        session = OptimizationSession()
        
        self._sessions[session_id] = {
            "session": session,
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
            "expires_at": datetime.now() + self.default_ttl
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
    
    def delete(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted, False if not found
        """
        if session_id in self._sessions:
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
    
    def export_session(self, session_id: str) -> Optional[bytes]:
        """
        Export a session as bytes for download.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Pickled session data or None if not found
        """
        if session_id not in self._sessions:
            return None
        
        try:
            return pickle.dumps(self._sessions[session_id])
        except Exception as e:
            logger.error(f"Failed to export session {session_id}: {e}")
            return None
    
    def import_session(self, session_data: bytes, session_id: Optional[str] = None) -> Optional[str]:
        """
        Import a session from bytes.
        
        Args:
            session_data: Pickled session data
            session_id: Optional custom session ID (generates new one if None)
            
        Returns:
            Session ID or None if import failed
        """
        try:
            imported_data = pickle.loads(session_data)
            
            # Generate new session ID if not provided
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # Update timestamps
            imported_data["last_accessed"] = datetime.now()
            imported_data["expires_at"] = datetime.now() + self.default_ttl
            
            # Store session
            self._sessions[session_id] = imported_data
            self._save_to_disk(session_id)
            
            logger.info(f"Imported session {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to import session: {e}")
            return None


# Global session store instance
session_store = SessionStore(default_ttl_hours=24)
