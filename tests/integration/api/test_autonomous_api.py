"""
Test autonomous optimization API workflow.

Tests the complete autonomous workflow:
1. Create session
2. Define search space
3. Generate initial design (DoE)
4. Add experiments with auto-train
5. Get suggestions
6. Monitor via state endpoint

NOTE: These tests require a running API server on localhost:8000.
They will be skipped if the server is not available.
"""

import requests
import pytest
import json
import os
from typing import Dict, Any, Generator

BASE_URL = "http://localhost:8000/api/v1"

# Check if API server is running
def is_server_running() -> bool:
    """Check if the API server is available."""
    try:
        response = requests.get(f"{BASE_URL.rsplit('/api/v1', 1)[0]}/health", timeout=1)
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False

# Skip all tests in this module if server isn't running
pytestmark = pytest.mark.skipif(
    not is_server_running(),
    reason="API server not running on localhost:8000. Start with: python run_api.py"
)

# Get test data paths
# Go up 3 levels: api -> integration -> tests
TEST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CATALYST_SPACE_PATH = os.path.join(TEST_DIR, "catalyst_search_space.json")
CATALYST_DATA_PATH = os.path.join(TEST_DIR, "catalyst_experiments.csv")


@pytest.fixture
def session_id() -> Generator[str, None, None]:
    """Create a test session."""
    response = requests.post(f"{BASE_URL}/sessions")
    assert response.status_code == 201
    data = response.json()
    session_id = data["session_id"]
    
    yield session_id
    
    # Cleanup
    requests.delete(f"{BASE_URL}/sessions/{session_id}")


def test_complete_autonomous_workflow(session_id: str):
    """Test complete autonomous optimization workflow with real catalyst data."""
    
    # 1. Define search space from JSON spec (matching catalyst_search_space.json)
    variables = [
        {"name": "Temperature", "type": "real", "min": 350.0, "max": 450.0},
        {"name": "Catalyst", "type": "categorical", "categories": ["High SAR", "Low SAR"]},
        {"name": "Metal Loading", "type": "real", "min": 0.0, "max": 5.0},
        {"name": "Zinc Fraction", "type": "real", "min": 0.0, "max": 1.0}
    ]
    
    for var in variables:
        response = requests.post(
            f"{BASE_URL}/sessions/{session_id}/variables",
            json=var
        )
        assert response.status_code == 200, f"Failed to add variable {var['name']}: {response.json()}"
    
    print(f"✓ Added {len(variables)} variables to search space")
    
    # 2. Generate initial design (LHS, 10 points)
    response = requests.post(
        f"{BASE_URL}/sessions/{session_id}/initial-design",
        json={
            "method": "lhs",
            "n_points": 10,
            "random_seed": 42
        }
    )
    assert response.status_code == 200
    design_data = response.json()
    assert design_data["method"] == "lhs"
    assert design_data["n_points"] == 10
    assert len(design_data["points"]) == 10
    
    # Verify each point has correct structure
    for point in design_data["points"]:
        assert "Temperature" in point
        assert "Catalyst" in point
        assert "Metal Loading" in point
        assert "Zinc Fraction" in point
        assert 350.0 <= point["Temperature"] <= 450.0
        assert point["Catalyst"] in ["High SAR", "Low SAR"]
        assert 0.0 <= point["Metal Loading"] <= 5.0
        assert 0.0 <= point["Zinc Fraction"] <= 1.0
    
    print(f"✓ Generated {len(design_data['points'])} initial design points")
    
    # 3. Add initial experiments (simulate evaluation with simple objective)
    for i, point in enumerate(design_data["points"]):
        # Simulate a catalyst objective: higher temp + low SAR catalyst is better
        output = (point["Temperature"] - 350) / 100  # 0 to 1 range
        if point["Catalyst"] == "Low SAR":
            output += 0.2
        output += point["Metal Loading"] * 0.05
        output += point["Zinc Fraction"] * 0.1
        
        response = requests.post(
            f"{BASE_URL}/sessions/{session_id}/experiments",
            json={
                "inputs": point,
                "output": float(output),
                "noise": 0.01
            }
        )
        assert response.status_code == 200
    
    print(f"✓ Added {len(design_data['points'])} initial experiments")
    
    # 4. Check session state before training
    response = requests.get(f"{BASE_URL}/sessions/{session_id}/state")
    assert response.status_code == 200
    state = response.json()
    assert state["n_variables"] == 4
    assert state["n_experiments"] == 10
    assert state["model_trained"] == False
    
    print(f"✓ Session state: {state['n_experiments']} experiments, model_trained={state['model_trained']}")
    
    # 5. Add one more experiment with auto-train
    test_point = {
        "Temperature": 400.0,
        "Catalyst": "Low SAR",
        "Metal Loading": 2.5,
        "Zinc Fraction": 0.5
    }
    output = (400 - 350) / 100 + 0.2 + 2.5 * 0.05 + 0.5 * 0.1
    
    response = requests.post(
        f"{BASE_URL}/sessions/{session_id}/experiments?auto_train=true&training_backend=sklearn&training_kernel=rbf",
        json={
            "inputs": test_point,
            "output": float(output),
            "noise": 0.01
        }
    )
    assert response.status_code == 200
    exp_data = response.json()
    assert exp_data["n_experiments"] == 11
    assert exp_data["model_trained"] == True, f"Expected model_trained=True, got: {exp_data}"
    assert exp_data["training_metrics"] is not None
    assert "rmse" in exp_data["training_metrics"]
    assert "r2" in exp_data["training_metrics"]
    
    print(f"✓ Auto-trained model: RMSE={exp_data['training_metrics']['rmse']:.4f}, R²={exp_data['training_metrics']['r2']:.4f}")
    
    # 6. Check session state after training
    response = requests.get(f"{BASE_URL}/sessions/{session_id}/state")
    assert response.status_code == 200
    state = response.json()
    assert state["n_variables"] == 4
    assert state["n_experiments"] == 11
    assert state["model_trained"] == True
    
    print(f"✓ Session state: {state['n_experiments']} experiments, model_trained={state['model_trained']}")
    
    # 7. Get suggestion
    response = requests.post(
        f"{BASE_URL}/sessions/{session_id}/acquisition/suggest",
        json={"method": "ei"}
    )
    assert response.status_code == 200
    suggestion_data = response.json()
    assert "suggestions" in suggestion_data
    assert len(suggestion_data["suggestions"]) > 0
    
    suggestion = suggestion_data["suggestions"][0]
    assert "Temperature" in suggestion
    assert "Catalyst" in suggestion
    assert "Metal Loading" in suggestion
    assert "Zinc Fraction" in suggestion
    
    print(f"✓ Got suggestion: {suggestion}")
    
    print("\n✅ Complete autonomous workflow test passed!")


def test_batch_experiments_with_auto_train(session_id: str):
    """Test batch experiment upload with auto-train."""
    
    # Define simple 2D search space
    variables = [
        {"name": "x1", "type": "real", "min": 0.0, "max": 10.0},
        {"name": "x2", "type": "real", "min": 0.0, "max": 10.0}
    ]
    
    for var in variables:
        response = requests.post(
            f"{BASE_URL}/sessions/{session_id}/variables",
            json=var
        )
        assert response.status_code == 200
    
    # Add batch of 5 experiments with auto-train
    batch_experiments = [
        {"inputs": {"x1": 1.0, "x2": 2.0}, "output": 5.0, "noise": 0.1},
        {"inputs": {"x1": 3.0, "x2": 4.0}, "output": 25.0, "noise": 0.1},
        {"inputs": {"x1": 5.0, "x2": 6.0}, "output": 61.0, "noise": 0.1},
        {"inputs": {"x1": 7.0, "x2": 8.0}, "output": 113.0, "noise": 0.1},
        {"inputs": {"x1": 9.0, "x2": 1.0}, "output": 82.0, "noise": 0.1}
    ]
    
    response = requests.post(
        f"{BASE_URL}/sessions/{session_id}/experiments/batch?auto_train=true",
        json={"experiments": batch_experiments}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["n_experiments"] == 5
    assert data["model_trained"] == True
    assert data["training_metrics"] is not None
    
    print(f"✓ Batch upload with auto-train: {data['n_experiments']} experiments, RMSE={data['training_metrics']['rmse']:.3f}")
    
    # Verify session state
    response = requests.get(f"{BASE_URL}/sessions/{session_id}/state")
    assert response.status_code == 200
    state = response.json()
    assert state["n_experiments"] == 5
    assert state["model_trained"] == True
    
    print("✅ Batch auto-train test passed!")


def test_initial_design_methods(session_id: str):
    """Test all DoE methods via API."""
    
    # Define search space
    variables = [
        {"name": "x", "type": "real", "min": 0.0, "max": 1.0},
        {"name": "y", "type": "real", "min": 0.0, "max": 1.0}
    ]
    
    for var in variables:
        response = requests.post(
            f"{BASE_URL}/sessions/{session_id}/variables",
            json=var
        )
        assert response.status_code == 200
    
    # Test each DoE method
    methods = ["random", "lhs", "sobol", "halton", "hammersly"]
    
    for method in methods:
        response = requests.post(
            f"{BASE_URL}/sessions/{session_id}/initial-design",
            json={
                "method": method,
                "n_points": 10,
                "random_seed": 42
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["method"] == method
        assert data["n_points"] == 10
        assert len(data["points"]) == 10
        
        print(f"✓ {method}: generated {len(data['points'])} points")
    
    print("✅ All DoE methods test passed!")


if __name__ == "__main__":
    print("Testing autonomous optimization API endpoints...")
    print(f"Base URL: {BASE_URL}")
    print("=" * 60)
    
    # Create session for manual testing
    response = requests.post(f"{BASE_URL}/sessions")
    session_id = response.json()["session_id"]
    print(f"Created test session: {session_id}\n")
    
    try:
        test_complete_autonomous_workflow(session_id)
        print("\n" + "=" * 60)
        
        # Create new session for batch test
        response = requests.post(f"{BASE_URL}/sessions")
        session_id_2 = response.json()["session_id"]
        test_batch_experiments_with_auto_train(session_id_2)
        print("\n" + "=" * 60)
        
        # Create new session for DoE methods test
        response = requests.post(f"{BASE_URL}/sessions")
        session_id_3 = response.json()["session_id"]
        test_initial_design_methods(session_id_3)
        
        # Cleanup
        requests.delete(f"{BASE_URL}/sessions/{session_id}")
        requests.delete(f"{BASE_URL}/sessions/{session_id_2}")
        requests.delete(f"{BASE_URL}/sessions/{session_id_3}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        requests.delete(f"{BASE_URL}/sessions/{session_id}")
        raise
