"""
Simple test script to verify the ALchemist Docker container is working.

Run this after starting the container:
    python test_docker.py

This will:
1. Check if the API is responding
2. Create a test session
3. Add variables and experiments
4. Train a model
5. Get suggestions

Note: These tests require a running Docker container and will be skipped in CI.
"""

import requests
import json
import time
import pytest
import socket

BASE_URL = "http://localhost:8000/api/v1"

def is_docker_running():
    """Check if Docker container is running on localhost:8000"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', 8000))
        sock.close()
        return result == 0
    except:
        return False

# Skip all tests in this module if Docker is not running
pytestmark = pytest.mark.skipif(
    not is_docker_running(),
    reason="Docker container not running (integration tests only)"
)

@pytest.fixture(scope="module")
def session_id():
    """Create a session for all tests in this module"""
    response = requests.post(f"{BASE_URL}/sessions", json={"ttl_hours": 24})
    if response.status_code != 201:
        pytest.skip(f"Could not create session: {response.text}")
    return response.json()["session_id"]

def test_health():
    """Test if the server is running"""
    print("1. Testing health endpoint...")
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"   ✓ Health check: {response.json()}")
        return True
    except Exception as e:
        print(f"   ✗ Error: {e}")
        print("   Make sure the container is running: docker-compose ps")
        return False

def test_create_session():
    """Create a new session"""
    print("\n2. Creating session...")
    response = requests.post(f"{BASE_URL}/sessions", json={"ttl_hours": 24})
    if response.status_code == 201:  # POST returns 201 Created
        session_id = response.json()["session_id"]
        print(f"   ✓ Session created: {session_id}")
        return session_id
    else:
        print(f"   ✗ Error (status {response.status_code}): {response.text}")
        return None

def test_add_variables(session_id):
    """Add variables to search space"""
    print("\n3. Adding variables...")
    
    # Add temperature variable
    response = requests.post(
        f"{BASE_URL}/sessions/{session_id}/variables",
        json={
            "name": "temperature",
            "type": "real",
            "min": 100,
            "max": 500
        }
    )
    if response.status_code == 200:
        print("   ✓ Added 'temperature' variable")
    else:
        print(f"   ✗ Error adding temperature (status {response.status_code}): {response.text}")
        return False
    
    # Add pressure variable
    response = requests.post(
        f"{BASE_URL}/sessions/{session_id}/variables",
        json={
            "name": "pressure",
            "type": "real",
            "min": 1,
            "max": 10
        }
    )
    if response.status_code == 200:
        print("   ✓ Added 'pressure' variable")
        return True
    else:
        print(f"   ✗ Error adding pressure (status {response.status_code}): {response.text}")
        return False

def test_add_experiments(session_id):
    """Add experimental data"""
    print("\n4. Adding experimental data...")
    
    experiments = [
        {"inputs": {"temperature": 250, "pressure": 5}, "output": 0.85},
        {"inputs": {"temperature": 300, "pressure": 7}, "output": 0.92},
        {"inputs": {"temperature": 200, "pressure": 3}, "output": 0.71},
        {"inputs": {"temperature": 400, "pressure": 8}, "output": 0.88},
        {"inputs": {"temperature": 350, "pressure": 4}, "output": 0.78},
    ]
    
    for i, exp in enumerate(experiments, 1):
        response = requests.post(
            f"{BASE_URL}/sessions/{session_id}/experiments",
            json=exp
        )
        if response.status_code != 200:
            print(f"   ✗ Error adding experiment {i}: {response.text}")
            return False
    
    print(f"   ✓ Added {len(experiments)} experiments")
    return True

def test_train_model(session_id):
    """Train a surrogate model"""
    print("\n5. Training model...")
    
    response = requests.post(
        f"{BASE_URL}/sessions/{session_id}/model/train",
        json={
            "backend": "botorch",
            "kernel": "rbf",
            "output_transform": "standardize"
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Model trained successfully")
        print(f"   - Backend: {result['backend']}")
        print(f"   - Kernel: {result['kernel']}")
        return True
    else:
        print(f"   ✗ Error training model: {response.text}")
        return False

def test_get_suggestions(session_id):
    """Get next experiment suggestions"""
    print("\n6. Getting next experiment suggestions...")
    
    response = requests.post(
        f"{BASE_URL}/sessions/{session_id}/acquisition/suggest",
        json={
            "strategy": "qEI",
            "goal": "maximize",
            "n_suggestions": 3
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Generated {result['n_suggestions']} suggestions:")
        for i, sugg in enumerate(result['suggestions'], 1):
            print(f"     {i}. Temperature: {sugg['temperature']:.1f}°C, Pressure: {sugg['pressure']:.1f} bar")
        return True
    else:
        print(f"   ✗ Error getting suggestions: {response.text}")
        return False

def test_predictions(session_id):
    """Make predictions at test points"""
    print("\n7. Making predictions...")
    
    response = requests.post(
        f"{BASE_URL}/sessions/{session_id}/model/predict",
        json={
            "inputs": [
                {"temperature": 275, "pressure": 6},
                {"temperature": 375, "pressure": 5}
            ]
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Made {result['n_predictions']} predictions:")
        for pred in result['predictions']:
            print(f"     T={pred['inputs']['temperature']}°C, P={pred['inputs']['pressure']} bar → "
                  f"Output: {pred['prediction']:.3f} ± {pred['uncertainty']:.3f}")
        return True
    else:
        print(f"   ✗ Error making predictions: {response.text}")
        return False

def main():
    print("=" * 60)
    print("ALchemist Docker Container Test")
    print("=" * 60)
    
    # Test health
    if not test_health():
        return
    
    # Create session
    session_id = test_create_session()
    if not session_id:
        return
    
    # Add variables
    if not test_add_variables(session_id):
        return
    
    # Add experiments
    if not test_add_experiments(session_id):
        return
    
    # Train model
    if not test_train_model(session_id):
        return
    
    # Get suggestions
    if not test_get_suggestions(session_id):
        return
    
    # Make predictions
    if not test_predictions(session_id):
        return
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! Docker container is working correctly.")
    print("=" * 60)
    print("\nNext steps:")
    print("- View API docs: http://localhost:8000/api/docs")
    print("- View logs: docker-compose logs -f")
    print("- Stop container: docker-compose down")

if __name__ == "__main__":
    main()
