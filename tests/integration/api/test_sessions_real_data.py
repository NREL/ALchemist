
import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.services.session_store import session_store
import json
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import os

client = TestClient(app)

class TestRealDataWorkflow:
    @pytest.fixture(autouse=True)
    def setup_session_store(self):
        # Create temp dir for persistence
        temp_dir = tempfile.mkdtemp()
        original_persist_dir = session_store.persist_dir
        session_store.persist_dir = Path(temp_dir)
        session_store._sessions = {}  # Clear in-memory sessions
        
        yield
        
        # Cleanup
        session_store.persist_dir = original_persist_dir
        shutil.rmtree(temp_dir)

    def test_full_workflow_catalyst_data(self):
        # 1. Create Session
        response = client.post("/api/v1/sessions")
        assert response.status_code == 201
        session_id = response.json()["session_id"]
        
        # 2. Load Search Space
        search_space_path = Path("tests/catalyst_search_space.json")
        with open(search_space_path, "r") as f:
            search_space_data = json.load(f)
            
        for var in search_space_data:
            payload = {
                "name": var["name"],
                "type": var["type"].lower(),
                "min": var.get("min"),
                "max": var.get("max"),
                "categories": var.get("values")  # Map values to categories for API
            }
            # Filter None values
            payload = {k: v for k, v in payload.items() if v is not None}
            
            resp = client.post(f"/api/v1/sessions/{session_id}/variables", json=payload)
            assert resp.status_code == 200, f"Failed to add variable {var['name']}: {resp.text}"

        # 3. Upload Experiments
        csv_path = Path("tests/catalyst_experiments.csv")
        with open(csv_path, "rb") as f:
            files = {"file": ("catalyst_experiments.csv", f, "text/csv")}
            # Default target_column is "Output", which matches the CSV
            resp = client.post(f"/api/v1/sessions/{session_id}/experiments/upload", files=files)
            assert resp.status_code == 200, f"Failed to upload CSV: {resp.text}"
            data = resp.json()
            assert data["n_experiments"] > 0
            
        # 4. Train Model (Sklearn)
        train_payload = {
            "backend": "sklearn",
            "kernel": "rbf"
        }
        resp = client.post(f"/api/v1/sessions/{session_id}/model/train", json=train_payload)
        assert resp.status_code == 200, f"Failed to train model: {resp.text}"
        model_info = resp.json()
        assert model_info["success"] is True
        assert "metrics" in model_info
        
        # 5. Get Suggestions
        suggest_payload = {
            "strategy": "ei",
            "n_suggestions": 1,
            "maximize": True
        }
        resp = client.post(f"/api/v1/sessions/{session_id}/acquisition/suggest", json=suggest_payload)
        assert resp.status_code == 200, f"Failed to get suggestions: {resp.text}"
        suggestions = resp.json()["suggestions"]
        assert len(suggestions) == 1
        assert "Temperature" in suggestions[0]
        
        # 6. Train Model (BoTorch)
        train_payload = {
            "backend": "botorch",
            "kernel": "rbf"
        }
        resp = client.post(f"/api/v1/sessions/{session_id}/model/train", json=train_payload)
        assert resp.status_code == 200, f"Failed to train BoTorch model: {resp.text}"
        
        # 7. Get Suggestions (BoTorch qEI)
        suggest_payload = {
            "strategy": "qei",
            "n_suggestions": 2,
            "maximize": True
        }
        resp = client.post(f"/api/v1/sessions/{session_id}/acquisition/suggest", json=suggest_payload)
        assert resp.status_code == 200, f"Failed to get qEI suggestions: {resp.text}"
        suggestions = resp.json()["suggestions"]
        assert len(suggestions) == 2
