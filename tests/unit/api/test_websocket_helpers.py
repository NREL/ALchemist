"""Unit tests for helper utilities in ``api.routers.websocket``."""

import asyncio
from typing import Any, Dict, List

import pytest

from api.routers import websocket as ws


@pytest.fixture(autouse=True)
def reset_connections():
    """Ensure the global connection registry starts empty for each test."""

    ws.active_connections.clear()
    yield
    ws.active_connections.clear()


class DummyWebSocket:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.sent: List[Dict[str, Any]] = []

    async def send_json(self, payload: Dict[str, Any]):
        if self.fail:
            raise RuntimeError("connection closed")
        self.sent.append(payload)


def test_broadcast_no_listeners_capitalizes_on_debug():
    asyncio.run(ws.broadcast_to_session("missing", {"event": "anything"}))
    assert ws.active_connections == {}


def test_broadcast_delivers_to_all_connections():
    conn = DummyWebSocket()
    ws.active_connections["session"] = {conn}

    payload = {"event": "lock-status", "status": "locked"}
    asyncio.run(ws.broadcast_to_session("session", payload))

    assert conn.sent == [payload]
    assert ws.get_connection_count("session") == 1


def test_broadcast_cleans_dead_connections():
    good = DummyWebSocket()
    bad = DummyWebSocket(fail=True)
    ws.active_connections["session"] = {good, bad}

    payload = {"event": "update"}
    asyncio.run(ws.broadcast_to_session("session", payload))

    assert good.sent == [payload]
    assert "session" in ws.active_connections
    assert ws.active_connections["session"] == {good}


def test_connection_count_helpers():
    conn = DummyWebSocket()
    ws.active_connections["s1"] = {conn}
    ws.active_connections["s2"] = set()

    assert ws.get_connection_count("s1") == 1
    assert ws.get_connection_count("s2") == 0
    assert ws.get_connection_count("missing") == 0

    counts = ws.get_all_connection_counts()
    assert counts == {"s1": 1, "s2": 0}
