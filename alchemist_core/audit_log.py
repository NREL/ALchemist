"""
Audit Log - Append-only logging for reproducible optimization workflows.

This module provides structured logging of optimization decisions to ensure
research reproducibility and traceability. The audit log captures:
- Experimental data lock-ins
- Model training decisions
- Acquisition function choices

Users can explore freely without spamming the log; only explicit "lock-in"
actions create audit entries.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict, field
import hashlib
import json
import uuid


@dataclass
class SessionMetadata:
    """
    Session metadata for user-friendly session management.
    
    Attributes:
        session_id: Unique session identifier (UUID)
        name: User-friendly session name
        created_at: ISO timestamp of session creation
        last_modified: ISO timestamp of last modification
        description: Optional detailed description
        tags: Optional list of tags for organization
    """
    session_id: str
    name: str
    created_at: str
    last_modified: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    @staticmethod
    def create(name: str = "Untitled Session", description: str = "", 
               tags: Optional[List[str]] = None) -> 'SessionMetadata':
        """
        Create new session metadata.
        
        Args:
            name: User-friendly session name
            description: Optional description
            tags: Optional tags for organization
            
        Returns:
            SessionMetadata instance
        """
        now = datetime.now().isoformat()
        return SessionMetadata(
            session_id=str(uuid.uuid4()),
            name=name,
            created_at=now,
            last_modified=now,
            description=description,
            tags=tags or []
        )
    
    def update_modified(self):
        """Update last_modified timestamp to now."""
        self.last_modified = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        return asdict(self)
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'SessionMetadata':
        """Import from dictionary."""
        return SessionMetadata(**data)


@dataclass
class AuditEntry:
    """
    Single audit log entry.
    
    Attributes:
        timestamp: ISO timestamp of entry creation
        entry_type: Type of decision ('data_locked', 'model_locked', 'acquisition_locked')
        parameters: Complete snapshot of decision parameters
        hash: Reproducibility checksum (SHA256 of parameters)
        notes: Optional user notes
    """
    timestamp: str
    entry_type: str
    parameters: Dict[str, Any]
    hash: str
    notes: str = ""
    
    @staticmethod
    def create(entry_type: str, parameters: Dict[str, Any], 
               notes: str = "") -> 'AuditEntry':
        """
        Create new audit entry with auto-generated timestamp and hash.
        
        Args:
            entry_type: Type of entry ('data_locked', 'model_locked', 'acquisition_locked')
            parameters: Parameters to log
            notes: Optional user notes
            
        Returns:
            AuditEntry instance
        """
        timestamp = datetime.now().isoformat()
        
        # Create reproducibility hash
        # Sort keys for deterministic hashing
        param_str = json.dumps(parameters, sort_keys=True, default=str)
        hash_val = hashlib.sha256(param_str.encode()).hexdigest()[:16]
        
        return AuditEntry(
            timestamp=timestamp,
            entry_type=entry_type,
            parameters=parameters,
            hash=hash_val,
            notes=notes
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        return asdict(self)
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'AuditEntry':
        """Import from dictionary."""
        return AuditEntry(**data)


class AuditLog:
    """
    Append-only audit log for optimization decisions.
    
    This class maintains a complete, immutable history of optimization decisions
    to ensure reproducibility and traceability. Only explicit "lock-in" actions
    add entries, preventing log spam from exploration activities.
    
    Example:
        >>> audit_log = AuditLog()
        >>> 
        >>> # Lock in experimental data
        >>> entry = audit_log.lock_data(
        ...     n_experiments=10,
        ...     variables=[{'name': 'temp', 'type': 'real', 'min': 100, 'max': 300}],
        ...     data_hash='abc123'
        ... )
        >>> 
        >>> # Lock in trained model
        >>> entry = audit_log.lock_model(
        ...     backend='sklearn',
        ...     kernel='matern',
        ...     hyperparameters={'length_scale': 0.5},
        ...     cv_metrics={'rmse': 0.15, 'r2': 0.92}
        ... )
    """
    
    def __init__(self):
        """Initialize empty audit log."""
        self.entries: List[AuditEntry] = []
    
    def lock_data(self, n_experiments: int, variables: List[Dict[str, Any]], 
                  data_hash: str, notes: str = "") -> AuditEntry:
        """
        Lock in experimental data configuration.
        
        Args:
            n_experiments: Number of experiments
            variables: List of variable definitions
            data_hash: Hash of experimental data for verification
            notes: Optional user notes
            
        Returns:
            Created AuditEntry
        """
        entry = AuditEntry.create(
            entry_type='data_locked',
            parameters={
                'n_experiments': n_experiments,
                'variables': variables,
                'data_hash': data_hash
            },
            notes=notes
        )
        self.entries.append(entry)
        return entry
    
    def lock_model(self, backend: str, kernel: str, 
                   hyperparameters: Dict[str, Any], 
                   cv_metrics: Optional[Dict[str, float]] = None,
                   notes: str = "") -> AuditEntry:
        """
        Lock in trained model configuration.
        
        Args:
            backend: Model backend ('sklearn', 'botorch')
            kernel: Kernel type
            hyperparameters: Learned hyperparameters
            cv_metrics: Cross-validation metrics (optional)
            notes: Optional user notes
            
        Returns:
            Created AuditEntry
        """
        params = {
            'backend': backend,
            'kernel': kernel,
            'hyperparameters': hyperparameters
        }
        if cv_metrics is not None:
            params['cv_metrics'] = cv_metrics
        
        entry = AuditEntry.create(
            entry_type='model_locked',
            parameters=params,
            notes=notes
        )
        self.entries.append(entry)
        return entry
    
    def lock_acquisition(self, strategy: str, parameters: Dict[str, Any],
                        suggestions: List[Dict[str, Any]], 
                        notes: str = "") -> AuditEntry:
        """
        Lock in acquisition function decision.
        
        Args:
            strategy: Acquisition strategy name
            parameters: Acquisition function parameters
            suggestions: Suggested next experiments
            notes: Optional user notes
            
        Returns:
            Created AuditEntry
        """
        entry = AuditEntry.create(
            entry_type='acquisition_locked',
            parameters={
                'strategy': strategy,
                'parameters': parameters,
                'suggestions': suggestions
            },
            notes=notes
        )
        self.entries.append(entry)
        return entry
    
    def get_entries(self, entry_type: Optional[str] = None) -> List[AuditEntry]:
        """
        Get audit entries, optionally filtered by type.
        
        Args:
            entry_type: Optional filter ('data_locked', 'model_locked', 'acquisition_locked')
            
        Returns:
            List of AuditEntry objects
        """
        if entry_type is None:
            return self.entries.copy()
        return [e for e in self.entries if e.entry_type == entry_type]
    
    def get_latest(self, entry_type: str) -> Optional[AuditEntry]:
        """
        Get most recent entry of specified type.
        
        Args:
            entry_type: Entry type to find
            
        Returns:
            Latest AuditEntry or None if not found
        """
        entries = self.get_entries(entry_type)
        return entries[-1] if entries else None
    
    def clear(self):
        """
        Clear all entries (use with caution - breaks immutability contract).
        
        This should only be used when starting a completely new optimization
        campaign within the same session.
        """
        self.entries = []
    
    def to_dict(self) -> List[Dict[str, Any]]:
        """
        Export audit log to dictionary format.
        
        Returns:
            List of entry dictionaries
        """
        return [entry.to_dict() for entry in self.entries]
    
    def from_dict(self, data: List[Dict[str, Any]]):
        """
        Import audit log from dictionary format.
        
        Args:
            data: List of entry dictionaries
        """
        self.entries = [AuditEntry.from_dict(entry) for entry in data]
    
    def to_markdown(self) -> str:
        """
        Export audit log to markdown format for publications.
        
        Returns:
            Markdown-formatted audit trail
        """
        lines = ["# Optimization Audit Trail\n"]
        
        for i, entry in enumerate(self.entries, 1):
            lines.append(f"## Entry {i}: {entry.entry_type.replace('_', ' ').title()}")
            lines.append(f"**Timestamp**: {entry.timestamp}")
            lines.append(f"**Hash**: `{entry.hash}`\n")
            
            if entry.notes:
                lines.append(f"**Notes**: {entry.notes}\n")
            
            lines.append("**Parameters**:")
            lines.append("```json")
            lines.append(json.dumps(entry.parameters, indent=2))
            lines.append("```\n")
        
        return "\n".join(lines)
    
    def __len__(self) -> int:
        """Return number of entries."""
        return len(self.entries)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"AuditLog({len(self.entries)} entries)"
