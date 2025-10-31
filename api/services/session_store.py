"""
Session Store - In-memory session management.

Stores OptimizationSession instances with TTL and automatic cleanup.
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
import uuid
from alchemist_core.session import OptimizationSession
import logging

logger = logging.getLogger(__name__)


class SessionStore:
    """In-memory store for optimization sessions."""
    
    def __init__(self, default_ttl_hours: int = 24):
        """
        Initialize session store.
        
        Args:
            default_ttl_hours: Default time-to-live for sessions in hours
        """
        self._sessions: Dict[str, Dict] = {}
        self.default_ttl = timedelta(hours=default_ttl_hours)
        logger.info(f"SessionStore initialized with TTL={default_ttl_hours}h")
    
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
            logger.info(f"Cleaned up expired session {sid}")
    
    def count(self) -> int:
        """Get count of active sessions."""
        self._cleanup_expired()
        return len(self._sessions)
    
    def list_all(self) -> list:
        """Get list of all active session IDs."""
        self._cleanup_expired()
        return list(self._sessions.keys())


# Global session store instance
session_store = SessionStore(default_ttl_hours=24)
