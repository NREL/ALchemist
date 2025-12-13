"""
Integration tests for experiments router covering error handling and
auto-training/initial-design success paths.
"""

import io

import pytest
from fastapi.testclient import TestClient

from alchemist_core.session import OptimizationSession
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


def test_add_experiment_auto_trains_and_logs(session_id, monkeypatch):
    _add_variables(session_id)

    # Seed session with enough experiments to meet the auto-train threshold.
    for idx in range(4):
        payload = {
            "inputs": {"temperature": 200 + idx * 5, "pressure": 3 + idx * 0.5},
            "output": 0.5 + idx * 0.01,
        }
        resp = client.post(f"/api/v1/sessions/{session_id}/experiments", json=payload)
        resp.raise_for_status()

    train_call = {}

    def fake_train_model(self, backend="sklearn", kernel="rbf"):
        train_call["backend"] = backend
        train_call["kernel"] = kernel
        return {
            "metrics": {"rmse": 0.12, "r2": 0.9},
            "hyperparameters": {"kernel": kernel},
        }

    monkeypatch.setattr(OptimizationSession, "train_model", fake_train_model)

    locked = {}

    def fake_lock_model(self, backend, kernel, hyperparameters, cv_metrics, iteration, notes):
        locked.update(
            {
                "backend": backend,
                "kernel": kernel,
                "iteration": iteration,
                "notes": notes,
            }
        )

    monkeypatch.setattr("alchemist_core.audit_log.AuditLog.lock_model", fake_lock_model)

    final_payload = {
        "inputs": {"temperature": 225, "pressure": 4.5},
        "output": 0.61,
        "iteration": 2,
    }
    response = client.post(
        f"/api/v1/sessions/{session_id}/experiments",
        params={"auto_train": "true", "training_backend": "sklearn", "training_kernel": "matern"},
        json=final_payload,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model_trained"] is True
    assert body["training_metrics"] == {"rmse": 0.12, "r2": 0.9, "backend": "sklearn"}
    assert train_call == {"backend": "sklearn", "kernel": "matern"}
    assert locked["iteration"] == 2
    assert locked["backend"] == "sklearn"


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

    stats = client.get(f"/api/v1/sessions/{session_id}/experiments/summary")
    stats.raise_for_status()
    summary_body = stats.json()
    assert summary_body["n_experiments"] == 2


def test_batch_auto_train_returns_metrics(session_id, monkeypatch):
    _add_variables(session_id)

    train_call = {}

    def fake_train_model(self, backend="sklearn", kernel="rbf"):
        train_call["backend"] = backend
        train_call["kernel"] = kernel
        return {
            "metrics": {"rmse": 0.2, "r2": 0.85},
        }

    monkeypatch.setattr(OptimizationSession, "train_model", fake_train_model)

    experiments = [
        {"inputs": {"temperature": 200 + i * 10, "pressure": 3 + i}, "output": 0.5 + i * 0.05}
        for i in range(5)
    ]

    response = client.post(
        f"/api/v1/sessions/{session_id}/experiments/batch",
        params={"auto_train": "true", "training_backend": "sklearn", "training_kernel": "rbf"},
        json={"experiments": experiments},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model_trained"] is True
    assert body["training_metrics"] == {"rmse": 0.2, "r2": 0.85, "backend": "sklearn"}
    assert train_call == {"backend": "sklearn", "kernel": "rbf"}


def test_initial_design_requires_variables(session_id):
    response = client.post(
        f"/api/v1/sessions/{session_id}/initial-design",
        json={"method": "lhs", "n_points": 3, "lhs_criterion": "maximin"},
    )
    assert response.status_code == 400
    assert "no variables" in response.json()["detail"].lower()


def test_initial_design_generates_points(session_id):
    _add_variables(session_id)

    response = client.post(
        f"/api/v1/sessions/{session_id}/initial-design",
        json={
            "method": "lhs",
            "n_points": 4,
            "lhs_criterion": "maximin",
            "random_seed": 123,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["n_points"] == 4
    assert body["method"] == "lhs"
    assert len(body["points"]) == 4
