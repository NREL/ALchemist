"""
Tests for the websocket router utilities.
"""

import asyncio

from fastapi.testclient import TestClient

from api.main import app
from api.routers import websocket as ws_router

client = TestClient(app)


def test_websocket_connection_lifecycle():
    session_id = "ws-session-lifecycle"
    with client.websocket_connect(f"/api/v1/ws/sessions/{session_id}") as websocket:
        initial = websocket.receive_json()
        assert initial["event"] == "connected"
        assert initial["session_id"] == session_id
        assert ws_router.get_connection_count(session_id) == 1
        assert ws_router.get_all_connection_counts()[session_id] == 1
    assert ws_router.get_connection_count(session_id) == 0
    assert session_id not in ws_router.active_connections


class DummyWebSocket:
    def __init__(self):
        self.sent = []

    async def send_json(self, data):
        self.sent.append(data)

    def __hash__(self):
        return id(self)


class FailingWebSocket(DummyWebSocket):
    async def send_json(self, data):
        raise RuntimeError("socket closed")


def test_broadcast_to_session_cleans_dead_connections():
    session_id = "ws-broadcast"
    good_socket = DummyWebSocket()
    failing_socket = FailingWebSocket()
    ws_router.active_connections[session_id] = {good_socket, failing_socket}

    event = {"event": "lock", "status": "locked"}
    asyncio.run(ws_router.broadcast_to_session(session_id, event))

    assert good_socket.sent == [event]
    assert session_id in ws_router.active_connections
    assert ws_router.active_connections[session_id] == {good_socket}

    event_two = {"event": "unlock", "status": "unlocked"}
    asyncio.run(ws_router.broadcast_to_session(session_id, event_two))
    assert good_socket.sent == [event, event_two]

    ws_router.active_connections.pop(session_id, None)
