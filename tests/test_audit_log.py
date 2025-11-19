"""
Test suite for audit log and session management functionality.
"""

import pytest
import json
from pathlib import Path
from alchemist_core.session import OptimizationSession
from alchemist_core.audit_log import AuditLog, SessionMetadata, AuditEntry


def test_session_metadata_creation():
    """Test creating session metadata."""
    metadata = SessionMetadata.create(
        name="Test Session",
        description="A test session",
        tags=["test", "demo"]
    )
    
    assert metadata.name == "Test Session"
    assert metadata.description == "A test session"
    assert metadata.tags == ["test", "demo"]
    assert metadata.session_id is not None
    assert metadata.created_at is not None


def test_audit_log_basic():
    """Test basic audit log functionality."""
    audit_log = AuditLog()
    assert len(audit_log) == 0
    
    # Lock data
    entry = audit_log.lock_data(
        n_experiments=10,
        variables=[{"name": "temp", "type": "real"}],
        data_hash="abc123"
    )
    
    assert len(audit_log) == 1
    assert entry.entry_type == "data_locked"
    assert entry.parameters["n_experiments"] == 10
    
    # Lock model
    entry = audit_log.lock_model(
        backend="sklearn",
        kernel="matern",
        hyperparameters={"length_scale": 0.5},
        cv_metrics={"rmse": 0.15, "r2": 0.92}
    )
    
    assert len(audit_log) == 2
    assert entry.entry_type == "model_locked"


def test_audit_log_export():
    """Test audit log export to dict and markdown."""
    audit_log = AuditLog()
    
    audit_log.lock_data(
        n_experiments=5,
        variables=[],
        data_hash="test123"
    )
    
    # Export to dict
    data = audit_log.to_dict()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["entry_type"] == "data_locked"
    
    # Export to markdown
    md = audit_log.to_markdown()
    assert "# Optimization Audit Trail" in md
    assert "Data Locked" in md or "data_locked" in md.lower()


def test_session_with_audit_log():
    """Test OptimizationSession with audit log integration."""
    session = OptimizationSession()
    
    # Update metadata
    session.update_metadata(
        name="Test Optimization",
        description="Testing audit log",
        tags=["test"]
    )
    
    assert session.metadata.name == "Test Optimization"
    assert len(session.audit_log) == 0
    
    # Add variables and data
    session.add_variable("temperature", "real", min=100, max=300)
    session.add_experiment({"temperature": 200}, output=85.0)
    
    # Lock data
    entry = session.lock_data(notes="Initial dataset")
    assert len(session.audit_log) == 1
    assert entry.notes == "Initial dataset"
    
    # Train model
    session.train_model(backend="sklearn", kernel="rbf")
    
    # Lock model
    entry = session.lock_model(notes="Good CV performance")
    assert len(session.audit_log) == 2
    assert entry.notes == "Good CV performance"


def test_session_save_load(tmp_path):
    """Test session save and load functionality."""
    # Create session with data
    session = OptimizationSession()
    session.update_metadata(name="Save Test", description="Testing save/load")
    session.add_variable("x", "real", min=0, max=10)
    session.add_variable("y", "integer", min=1, max=5)
    session.add_experiment({"x": 5.0, "y": 3}, output=42.0)
    session.lock_data(notes="Test data")
    
    # Save session
    filepath = tmp_path / "test_session.json"
    session.save_session(str(filepath))
    
    assert filepath.exists()
    
    # Verify JSON structure
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    assert data["version"] == "1.0.0"
    assert data["metadata"]["name"] == "Save Test"
    assert len(data["audit_log"]) == 1
    assert len(data["search_space"]["variables"]) == 2
    assert data["experiments"]["n_total"] == 1
    
    # Load session
    loaded_session = OptimizationSession.load_session(str(filepath))
    
    assert loaded_session.metadata.name == "Save Test"
    assert loaded_session.metadata.description == "Testing save/load"
    assert len(loaded_session.search_space.variables) == 2
    assert len(loaded_session.experiment_manager.df) == 1
    assert len(loaded_session.audit_log) == 1


def test_lock_acquisition():
    """Test locking acquisition decisions."""
    session = OptimizationSession()
    session.add_variable("x", "real", min=0, max=1)
    session.add_experiment({"x": 0.5}, output=1.0)
    
    suggestions = [
        {"x": 0.75, "acquisition_value": 0.5}
    ]
    
    entry = session.lock_acquisition(
        strategy="EI",
        parameters={"xi": 0.01, "goal": "maximize"},
        suggestions=suggestions,
        notes="Top candidate"
    )
    
    assert entry.entry_type == "acquisition_locked"
    assert entry.parameters["strategy"] == "EI"
    assert len(entry.parameters["suggestions"]) == 1


def test_audit_log_get_methods():
    """Test audit log filtering and retrieval."""
    audit_log = AuditLog()
    
    # Add multiple entries
    audit_log.lock_data(5, [], "hash1")
    audit_log.lock_model("sklearn", "rbf", {})
    audit_log.lock_data(10, [], "hash2")
    
    # Get all entries
    all_entries = audit_log.get_entries()
    assert len(all_entries) == 3
    
    # Get by type
    data_entries = audit_log.get_entries("data_locked")
    assert len(data_entries) == 2
    
    model_entries = audit_log.get_entries("model_locked")
    assert len(model_entries) == 1
    
    # Get latest
    latest_data = audit_log.get_latest("data_locked")
    assert latest_data.parameters["n_experiments"] == 10
    
    latest_acq = audit_log.get_latest("acquisition_locked")
    assert latest_acq is None


def test_export_audit_markdown():
    """Test markdown export from session."""
    session = OptimizationSession()
    session.add_variable("temp", "real", min=100, max=300)
    session.add_experiment({"temp": 200}, output=85.0)
    
    session.lock_data()
    session.train_model(backend="sklearn", kernel="rbf")
    session.lock_model()
    
    md = session.export_audit_markdown()
    
    assert "# Optimization Audit Trail" in md
    assert ("data locked" in md.lower() or "Data Locked" in md)
    assert ("model locked" in md.lower() or "Model Locked" in md)
    assert "Entry 1" in md
    assert "Entry 2" in md


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
