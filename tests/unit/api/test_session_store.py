"""
Unit tests for the SessionStore service.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Generator
from unittest.mock import MagicMock

import pytest

import api.services.session_store as session_store_module
import alchemist_core.session as core_session
from api.services.session_store import SessionStore
@pytest.fixture
def store(tmp_path) -> Generator[SessionStore, None, None]:
    """Create an isolated SessionStore that persists into a temp directory."""
    session_store = SessionStore(default_ttl_hours=1, persist_dir=str(tmp_path))
    yield session_store
    for session_id in list(session_store.list_all()):
        session_store.delete(session_id)
    for session_id in list(session_store.list_all()):
        session_store.delete(session_id)


def test_create_and_get_session(store: SessionStore):
    session_id = store.create(name="Coverage Test")
    session = store.get(session_id)
    assert session is not None
    assert session.metadata.name == "Coverage Test"
    assert store.count() == 1


def test_session_expiration_removes_entry(store: SessionStore):
    session_id = store.create()
    session_data = store._sessions[session_id]
    lock = session_data["lock"]
    with lock:
        session_data["expires_at"] = datetime.now() - timedelta(seconds=1)
    assert store.get(session_id) is None
    assert store.count() == 0


def test_persist_and_reload_session(tmp_path):
    store = SessionStore(default_ttl_hours=1, persist_dir=str(tmp_path))
    session_id = store.create()
    session = store.get(session_id)
    session.add_variable("temperature", "real", min=200.0, max=400.0)
    assert store.persist_session_to_disk(session_id) is True

    reloaded_store = SessionStore(default_ttl_hours=1, persist_dir=str(tmp_path))
    reloaded_session = reloaded_store.get(session_id)
    assert reloaded_session is not None
    names = {var["name"] for var in reloaded_session.search_space.variables}
    assert "temperature" in names

    store.delete(session_id)
    for sid in list(reloaded_store.list_all()):
        reloaded_store.delete(sid)


def test_export_and_import_session(store: SessionStore):
    session_id = store.create()
    session = store.get(session_id)
    session.add_variable("pressure", "real", min=1.0, max=10.0)

    exported = store.export_session(session_id)
    assert exported is not None

    imported_session_id = store.import_session(exported)
    assert imported_session_id is not None

    imported_session = store.get(imported_session_id)
    assert imported_session is not None
    names = {var["name"] for var in imported_session.search_space.variables}
    assert "pressure" in names

    store.delete(session_id)
    store.delete(imported_session_id)


def test_lock_and_unlock_session(store: SessionStore):
    session_id = store.create()

    lock_info = store.lock_session(session_id, locked_by="test-suite")
    assert lock_info["locked"] is True
    assert lock_info["lock_token"]

    status = store.get_lock_status(session_id)
    assert status["locked"] is True
    assert status["lock_token"] is None

    with pytest.raises(ValueError):
        store.unlock_session(session_id, lock_token="invalid-token")

    unlock_info = store.unlock_session(session_id, lock_token=lock_info["lock_token"])
    assert unlock_info["locked"] is False
    assert store.get_lock_status(session_id)["locked"] is False


def test_extend_ttl_updates_expiration(store: SessionStore):
    session_id = store.create()
    before = store._sessions[session_id]["expires_at"]
    assert store.extend_ttl(session_id, hours=2) is True
    after = store._sessions[session_id]["expires_at"]
    assert after > before


def test_extend_ttl_missing_session_returns_false(store: SessionStore):
    assert store.extend_ttl("missing") is False


def test_export_session_missing_returns_none(store: SessionStore):
    assert store.export_session("missing") is None


def test_export_session_handles_save_error(store: SessionStore):
    session_id = store.create()
    session = store.get(session_id)
    session.save_session = MagicMock(side_effect=RuntimeError("boom"))
    assert store.export_session(session_id) is None


def test_persist_session_to_disk_missing_session(store: SessionStore):
    assert store.persist_session_to_disk("missing") is False


def test_persist_session_to_disk_handles_error(store: SessionStore, monkeypatch: pytest.MonkeyPatch):
    session_id = store.create()

    def boom(_session_id: str) -> None:
        raise RuntimeError("io error")

    monkeypatch.setattr(store, "_save_to_disk", boom)
    assert store.persist_session_to_disk(session_id) is False


def test_import_session_invalid_data_returns_none(store: SessionStore, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        core_session.OptimizationSession,
        "load_session",
        MagicMock(side_effect=RuntimeError("bad import"))
    )
    assert store.import_session("{}") is None
    assert len(store.list_all()) == 0


def test_delete_missing_session_returns_false(store: SessionStore):
    assert store.delete("missing") is False


def test_get_info_missing_returns_none(store: SessionStore):
    assert store.get_info("missing") is None


def test_list_all_removes_expired(store: SessionStore):
    session_id = store.create()
    with store._sessions[session_id]["lock"]:
        store._sessions[session_id]["expires_at"] = datetime.now() - timedelta(seconds=1)
    assert session_id not in store.list_all()


def test_lock_session_missing_raises_keyerror(store: SessionStore):
    with pytest.raises(KeyError):
        store.lock_session("missing", locked_by="tester")


def test_unlock_session_missing_raises_keyerror(store: SessionStore):
    with pytest.raises(KeyError):
        store.unlock_session("missing")


def test_get_lock_status_missing_raises_keyerror(store: SessionStore):
    with pytest.raises(KeyError):
        store.get_lock_status("missing")


def test_get_lock_status_defaults_to_unlocked(store: SessionStore):
    session_id = store.create()
    status = store.get_lock_status(session_id)
    assert status["locked"] is False
    assert status["lock_token"] is None


def test_unlock_session_without_token(store: SessionStore):
    session_id = store.create()
    store.lock_session(session_id, locked_by="tester")
    result = store.unlock_session(session_id)
    assert result["locked"] is False
    assert store.get_lock_status(session_id)["locked"] is False
