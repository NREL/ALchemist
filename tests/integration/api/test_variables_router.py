"""
Integration tests for the variables router endpoints.
"""

import io
import json

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


@pytest.fixture
def session_id():
    """Create a fresh optimization session for each test and clean it up afterwards."""
    response = client.post("/api/v1/sessions", json={"ttl_hours": 1})
    assert response.status_code == 201
    session_id = response.json()["session_id"]
    yield session_id
    client.delete(f"/api/v1/sessions/{session_id}")


def test_add_variable_rejects_duplicates(session_id):
    payload = {
        "name": "temperature",
        "type": "real",
        "min": 250.0,
        "max": 500.0,
        "unit": "degC",
    }

    first_response = client.post(f"/api/v1/sessions/{session_id}/variables", json=payload)
    assert first_response.status_code == 200

    duplicate_response = client.post(f"/api/v1/sessions/{session_id}/variables", json=payload)
    assert duplicate_response.status_code == 400
    assert "already exists" in duplicate_response.json()["detail"].lower()


def test_update_variable_success_and_mismatched_name(session_id):
    payload = {
        "name": "pressure",
        "type": "real",
        "min": 1.0,
        "max": 10.0,
        "unit": "bar",
    }
    add_response = client.post(f"/api/v1/sessions/{session_id}/variables", json=payload)
    assert add_response.status_code == 200

    update_payload = {
        "name": "pressure",
        "type": "real",
        "min": 2.0,
        "max": 12.0,
        "unit": "bar",
    }
    update_response = client.put(
        f"/api/v1/sessions/{session_id}/variables/pressure",
        json=update_payload,
    )
    assert update_response.status_code == 200
    updated_variable = update_response.json()["variable"]
    assert updated_variable["min"] == pytest.approx(2.0)
    assert updated_variable["max"] == pytest.approx(12.0)

    mismatch_payload = {
        "name": "pressure-renamed",
        "type": "real",
        "min": 2.0,
        "max": 12.0,
    }
    mismatch_response = client.put(
        f"/api/v1/sessions/{session_id}/variables/pressure",
        json=mismatch_payload,
    )
    assert mismatch_response.status_code == 400
    assert "must match" in mismatch_response.json()["detail"].lower()

    missing_response = client.put(
        f"/api/v1/sessions/{session_id}/variables/does-not-exist",
        json={
            "name": "does-not-exist",
            "type": "real",
            "min": 0.0,
            "max": 1.0,
        },
    )
    assert missing_response.status_code == 404


def test_load_and_export_variables(session_id):
    variables_payload = [
        {
            "name": "temperature",
            "type": "real",
            "min": 100.0,
            "max": 200.0,
        },
        {
            "name": "catalyst",
            "type": "categorical",
            "categories": ["A", "B", "C"],
        },
    ]

    json_buffer = io.BytesIO(json.dumps(variables_payload).encode("utf-8"))
    load_response = client.post(
        f"/api/v1/sessions/{session_id}/variables/load",
        files={"file": ("variables.json", json_buffer, "application/json")},
    )
    assert load_response.status_code == 200
    assert load_response.json()["n_variables"] == 2

    list_response = client.get(f"/api/v1/sessions/{session_id}/variables")
    assert list_response.status_code == 200
    listed = list_response.json()
    assert listed["n_variables"] == 2
    names = {var["name"] for var in listed["variables"]}
    assert names == {"temperature", "catalyst"}

    export_response = client.get(f"/api/v1/sessions/{session_id}/variables/export")
    assert export_response.status_code == 200
    exported = export_response.json()
    assert len(exported) == 2
    assert any(var["name"] == "catalyst" for var in exported)
    assert export_response.headers["Content-Disposition"].startswith("attachment; filename=")


def test_delete_variable(session_id):
    payload = {
        "name": "cycles",
        "type": "integer",
        "min": 1,
        "max": 5,
    }
    add_response = client.post(f"/api/v1/sessions/{session_id}/variables", json=payload)
    assert add_response.status_code == 200

    delete_response = client.delete(f"/api/v1/sessions/{session_id}/variables/cycles")
    assert delete_response.status_code == 200
    body = delete_response.json()
    assert body["n_variables"] == 0

    get_response = client.get(f"/api/v1/sessions/{session_id}/variables")
    assert get_response.status_code == 200
    assert get_response.json()["n_variables"] == 0
