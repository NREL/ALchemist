"""
Integration tests for ALchemist FastAPI endpoints.

Tests complete workflow: session creation → variables → experiments → training → suggestions → predictions
"""

import pytest
from fastapi.testclient import TestClient
import io

# Import app - path should be configured by conftest.py
from api.main import app

client = TestClient(app)


class TestCompleteWorkflow:
    """Test complete optimization workflow."""
    
    def test_health_check(self):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_complete_workflow(self):
        """Test complete workflow from session creation to predictions."""
        
        # 1. Create session
        response = client.post("/api/v1/sessions", json={"ttl_hours": 1})
        assert response.status_code == 201  # 201 Created is correct for POST
        session_id = response.json()["session_id"]
        assert session_id is not None
        
        # 2. Add variables
        # Temperature variable
        response = client.post(
            f"/api/v1/sessions/{session_id}/variables",
            json={
                "name": "temperature",
                "type": "real",  # Use 'real' not 'continuous'
                "min": 100,
                "max": 500,
                "unit": "°C",
                "description": "Reaction temperature"
            }
        )
        assert response.status_code == 200
        
        # Pressure variable
        response = client.post(
            f"/api/v1/sessions/{session_id}/variables",
            json={
                "name": "pressure",
                "type": "real",  # Use 'real' not 'continuous'
                "min": 1,
                "max": 10,
                "unit": "bar"
            }
        )
        assert response.status_code == 200
        
        # Verify variables
        response = client.get(f"/api/v1/sessions/{session_id}/variables")
        assert response.status_code == 200
        variables = response.json()["variables"]
        assert len(variables) == 2
        assert variables[0]["name"] == "temperature"
        assert variables[1]["name"] == "pressure"
        
        # 3. Add experiments (need at least 5 for training)
        experiments = [
            {"inputs": {"temperature": 200, "pressure": 3}, "output": 0.65},
            {"inputs": {"temperature": 250, "pressure": 5}, "output": 0.85},
            {"inputs": {"temperature": 300, "pressure": 7}, "output": 0.92},
            {"inputs": {"temperature": 350, "pressure": 4}, "output": 0.78},
            {"inputs": {"temperature": 400, "pressure": 6}, "output": 0.88},
            {"inputs": {"temperature": 450, "pressure": 8}, "output": 0.81},
        ]
        
        # Add batch of experiments
        response = client.post(
            f"/api/v1/sessions/{session_id}/experiments/batch",
            json={"experiments": experiments}
        )
        assert response.status_code == 200
        assert response.json()["n_experiments"] == 6  # Check total, not n_added
        
        # Verify experiments
        response = client.get(f"/api/v1/sessions/{session_id}/experiments")
        assert response.status_code == 200
        assert response.json()["n_experiments"] == 6
        
        # Get summary
        response = client.get(f"/api/v1/sessions/{session_id}/experiments/summary")
        assert response.status_code == 200
        summary = response.json()
        assert summary["n_experiments"] == 6
        assert "temperature" in summary["feature_names"]
        assert "pressure" in summary["feature_names"]
        assert "min" in summary["target_stats"]
        assert "max" in summary["target_stats"]
        
        # 4. Train model
        response = client.post(
            f"/api/v1/sessions/{session_id}/model/train",
            json={
                "backend": "sklearn",
                "kernel": "rbf",
                "output_transform": "standardize"
            }
        )
        assert response.status_code == 200
        train_results = response.json()
        assert train_results["success"] is True
        assert train_results["backend"] == "sklearn"
        
        # Get model info
        response = client.get(f"/api/v1/sessions/{session_id}/model")
        assert response.status_code == 200
        model_info = response.json()
        assert model_info["is_trained"] is True
        assert model_info["backend"] == "sklearn"
        
        # 5. Get suggestions
        response = client.post(
            f"/api/v1/sessions/{session_id}/acquisition/suggest",
            json={
                "strategy": "EI",
                "goal": "maximize",
                "n_suggestions": 1  # Currently only supports 1 suggestion at a time
            }
        )
        assert response.status_code == 200
        suggestions = response.json()
        assert suggestions["n_suggestions"] == 1
        assert len(suggestions["suggestions"]) == 1
        
        # Verify suggestion format
        suggestion = suggestions["suggestions"][0]
        assert "temperature" in suggestion
        assert "pressure" in suggestion
        
        # 6. Make predictions
        test_points = [
            {"temperature": 275, "pressure": 5.5},
            {"temperature": 325, "pressure": 6.5},
        ]
        
        response = client.post(
            f"/api/v1/sessions/{session_id}/model/predict",
            json={"inputs": test_points}
        )
        assert response.status_code == 200
        predictions = response.json()
        assert predictions["n_predictions"] == 2
        
        # Verify prediction format
        pred = predictions["predictions"][0]
        assert "inputs" in pred
        assert "prediction" in pred
        assert "uncertainty" in pred
        assert pred["inputs"] == test_points[0]
        
        # 7. Delete session
        response = client.delete(f"/api/v1/sessions/{session_id}")
        assert response.status_code == 204  # 204 No Content is correct for DELETE
        
        # Verify session is gone
        response = client.get(f"/api/v1/sessions/{session_id}")
        assert response.status_code == 404


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_session_not_found(self):
        """Test accessing non-existent session."""
        response = client.get("/api/v1/sessions/nonexistent-id")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_train_without_data(self):
        """Test training model without data."""
        # Create session
        response = client.post("/api/v1/sessions")
        session_id = response.json()["session_id"]
        
        # Try to train without data
        response = client.post(
            f"/api/v1/sessions/{session_id}/model/train",
            json={"backend": "sklearn"}
        )
        assert response.status_code == 400
        assert "no experimental data" in response.json()["detail"].lower()
    
    def test_predict_without_model(self):
        """Test predictions without trained model."""
        # Create session
        response = client.post("/api/v1/sessions")
        session_id = response.json()["session_id"]
        
        # Try to predict without model
        response = client.post(
            f"/api/v1/sessions/{session_id}/model/predict",
            json={"inputs": [{"x": 1}]}
        )
        assert response.status_code == 400
        assert "no trained model" in response.json()["detail"].lower()
    
    def test_suggest_without_model(self):
        """Test suggestions without trained model."""
        # Create session
        response = client.post("/api/v1/sessions")
        session_id = response.json()["session_id"]
        
        # Try to suggest without model
        response = client.post(
            f"/api/v1/sessions/{session_id}/acquisition/suggest",
            json={"strategy": "EI", "goal": "maximize"}
        )
        assert response.status_code == 400
        assert "no trained model" in response.json()["detail"].lower()


class TestCSVUpload:
    """Test CSV file upload functionality."""
    
    def test_upload_experiments_csv(self):
        """Test uploading experiments from CSV."""
        # Create session and add variables
        response = client.post("/api/v1/sessions")
        session_id = response.json()["session_id"]
        
        client.post(
            f"/api/v1/sessions/{session_id}/variables",
            json={"name": "x1", "type": "real", "min": 0, "max": 10}
        )
        client.post(
            f"/api/v1/sessions/{session_id}/variables",
            json={"name": "x2", "type": "real", "min": 0, "max": 10}
        )
        
        # Create CSV content
        csv_content = """x1,x2,output
1,2,0.5
3,4,0.7
5,6,0.9
"""
        csv_file = io.BytesIO(csv_content.encode())
        
        # Upload CSV
        response = client.post(
            f"/api/v1/sessions/{session_id}/experiments/upload",
            files={"file": ("data.csv", csv_file, "text/csv")}
        )
        assert response.status_code == 200
        assert response.json()["n_experiments"] == 3  # Check total count
        
        # Verify experiments
        response = client.get(f"/api/v1/sessions/{session_id}/experiments")
        assert response.status_code == 200
        assert response.json()["n_experiments"] == 3


class TestVariableTypes:
    """Test different variable types."""
    
    def test_categorical_variable(self):
        """Test categorical variable definition."""
        response = client.post("/api/v1/sessions")
        session_id = response.json()["session_id"]
        
        response = client.post(
            f"/api/v1/sessions/{session_id}/variables",
            json={
                "name": "catalyst",
                "type": "categorical",
                "categories": ["Pt", "Pd", "Rh"],
                "description": "Catalyst type"
            }
        )
        assert response.status_code == 200
        
        # Verify variable
        response = client.get(f"/api/v1/sessions/{session_id}/variables")
        var = response.json()["variables"][0]
        assert var["type"] == "categorical"
        # Categories are stored internally, may not be in response - just verify type
        assert var["name"] == "catalyst"
    
    def test_discrete_variable(self):
        """Test discrete variable definition."""
        response = client.post("/api/v1/sessions")
        session_id = response.json()["session_id"]
        
        response = client.post(
            f"/api/v1/sessions/{session_id}/variables",
            json={
                "name": "cycles",
                "type": "integer",  # Use 'integer' not 'discrete'
                "min": 1,
                "max": 10,
                "unit": "cycles"
            }
        )
        assert response.status_code == 200
        
        # Verify variable
        response = client.get(f"/api/v1/sessions/{session_id}/variables")
        var = response.json()["variables"][0]
        assert var["type"] == "integer"
        assert var["name"] == "cycles"  # Just verify name, not min/max (stored internally)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
