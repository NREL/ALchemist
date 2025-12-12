"""
Integration tests for experiments router error handling and edge cases.
"""

import io

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


@pytest.fixture
def session_id():
    response = client.post("/api/v1/sessions", json={"ttl_hours": 1})
    response.raise_for_status()
    sid = response.json()["session_id"]
    yield sid
    client.delete(f"/api/v1/sessions/{sid}")


def _add_variables(sid: str) -> None:
    variables = [
        {"name": "temperature", "type": "real", "min": 100.0, "max": 500.0},
        {"name": "pressure", "type": "real", "min": 1.0, "max": 10.0},
    ]
    for payload in variables:
        response = client.post(f"/api/v1/sessions/{sid}/variables", json=payload)
        response.raise_for_status()


def test_batch_requires_variables(session_id):
    response = client.post(
        f"/api/v1/sessions/{session_id}/experiments/batch",
        json={"experiments": []},
    )
    assert response.status_code == 400
    assert "no variables" in response.json()["detail"].lower()


def test_upload_csv_requires_variables(session_id):
    csv_content = """temperature,pressure,Output\n200,3,0.5\n"""
    files = {"file": ("data.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    response = client.post(
        f"/api/v1/sessions/{session_id}/experiments/upload",
        files=files,
    )
    assert response.status_code == 400
    assert "no variables" in response.json()["detail"].lower()


def test_add_experiment_requires_variables(session_id):
    response = client.post(
        f"/api/v1/sessions/{session_id}/experiments",
        json={"inputs": {"temperature": 200}, "output": 0.5},
    )
    assert response.status_code == 400
    assert "no variables" in response.json()["detail"].lower()


def test_auto_train_handles_failure(session_id):
    _add_variables(session_id)

    exp_payload = {
        "inputs": {"temperature": 200, "pressure": 3},
        "output": 0.5,
    }
    response = client.post(
        f"/api/v1/sessions/{session_id}/experiments",
        params={"auto_train": "true", "training_backend": "unknown_backend"},
        json=exp_payload,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model_trained"] is False


def test_csv_upload_success(session_id):
    _add_variables(session_id)

    csv_content = """temperature,pressure,Output\n200,3,0.5\n210,4,0.6\n"""
    files = {"file": ("good.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    response = client.post(
        f"/api/v1/sessions/{session_id}/experiments/upload",
        files=files,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["n_experiments"] == 2

    summary = client.get(f"/api/v1/sessions/{session_id}/experiments").json()
    assert summary["n_experiments"] == 2
