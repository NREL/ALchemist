"""
Integration tests using real catalyst data with sklearn and botorch backends.

Tests the complete FastAPI workflow with actual experimental data:
- Catalyst search space with mixed variable types
- Real experimental results from CSV
- Both sklearn and botorch backends
- Matern 1.5 kernel
- EI acquisition function
"""
import sys
from pathlib import Path
import json

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

# Load test data files
TESTS_DIR = Path(__file__).parent.parent
SEARCH_SPACE_FILE = TESTS_DIR / "catalyst_search_space.json"
EXPERIMENTS_FILE = TESTS_DIR / "catalyst_experiments.csv"


class TestCatalystRealData:
    """Test real catalyst optimization with both backends."""

    def test_sklearn_backend_catalyst_data(self):
        """Test complete workflow with sklearn backend and real catalyst data."""
        
        # 1. Create session
        response = client.post("/api/v1/sessions", json={"ttl_hours": 2})
        assert response.status_code == 201
        session_id = response.json()["session_id"]
        print(f"\n✓ Created session: {session_id}")

        # 2. Load and add variables from JSON
        with open(SEARCH_SPACE_FILE, 'r') as f:
            search_space = json.load(f)
        
        for var in search_space:
            if var["type"] == "Real":
                payload = {
                    "name": var["name"],
                    "type": "real",
                    "min": var["min"],
                    "max": var["max"]
                }
            elif var["type"] == "Categorical":
                payload = {
                    "name": var["name"],
                    "type": "categorical",
                    "categories": var["values"]
                }
            else:
                raise ValueError(f"Unknown variable type: {var['type']}")
            
            response = client.post(
                f"/api/v1/sessions/{session_id}/variables",
                json=payload
            )
            assert response.status_code == 200
            print(f"✓ Added variable: {var['name']} ({var['type']})")

        # Verify variables
        response = client.get(f"/api/v1/sessions/{session_id}/variables")
        assert response.status_code == 200
        variables = response.json()["variables"]
        assert len(variables) == 4
        print(f"✓ Verified {len(variables)} variables")

        # 3. Upload experiments from CSV
        with open(EXPERIMENTS_FILE, 'rb') as f:
            files = {'file': ('catalyst_experiments.csv', f, 'text/csv')}
            response = client.post(
                f"/api/v1/sessions/{session_id}/experiments/upload",
                files=files
            )
        assert response.status_code == 200
        n_experiments = response.json()["n_experiments"]
        print(f"✓ Loaded {n_experiments} experiments from CSV")

        # Get data summary
        response = client.get(f"/api/v1/sessions/{session_id}/experiments/summary")
        assert response.status_code == 200
        summary = response.json()
        print(f"✓ Data summary:")
        print(f"  - Experiments: {summary['n_experiments']}")
        print(f"  - Features: {summary['feature_names']}")
        print(f"  - Target range: [{summary['target_stats']['min']:.4f}, {summary['target_stats']['max']:.4f}]")

        # 4. Train sklearn model with Matern 1.5 kernel
        response = client.post(
            f"/api/v1/sessions/{session_id}/model/train",
            json={
                "backend": "sklearn",
                "kernel": "matern",
                "kernel_params": {"nu": 1.5},
                "output_transform": "standardize"
            }
        )
        assert response.status_code == 200
        train_results = response.json()
        assert train_results["success"] is True
        print(f"✓ Trained sklearn model:")
        print(f"  - Kernel: {train_results['kernel']}")
        print(f"  - R²: {train_results['metrics']['r2']:.4f}")
        print(f"  - RMSE: {train_results['metrics']['rmse']:.4f}")

        # Get model info
        response = client.get(f"/api/v1/sessions/{session_id}/model")
        assert response.status_code == 200
        model_info = response.json()
        assert model_info["is_trained"] is True
        assert model_info["backend"] == "sklearn"
        print(f"✓ Model verified")

        # 5. Get acquisition suggestions using EI
        response = client.post(
            f"/api/v1/sessions/{session_id}/acquisition/suggest",
            json={
                "strategy": "EI",
                "goal": "maximize",
                "n_suggestions": 1
            }
        )
        assert response.status_code == 200
        suggestions = response.json()
        assert suggestions["n_suggestions"] == 1
        suggestion = suggestions["suggestions"][0]
        print(f"✓ Generated suggestion using EI:")
        for key, value in suggestion.items():
            print(f"  - {key}: {value}")

        # 6. Make predictions on suggested point
        response = client.post(
            f"/api/v1/sessions/{session_id}/model/predict",
            json={"inputs": [suggestion]}
        )
        assert response.status_code == 200
        predictions = response.json()
        pred = predictions["predictions"][0]
        print(f"✓ Prediction at suggested point:")
        print(f"  - Mean: {pred['prediction']:.4f}")
        print(f"  - Uncertainty: {pred['uncertainty']:.4f}")

        # 7. Clean up
        response = client.delete(f"/api/v1/sessions/{session_id}")
        assert response.status_code == 204
        print(f"✓ Session deleted\n")

    def test_botorch_backend_catalyst_data(self):
        """Test complete workflow with botorch backend and real catalyst data."""
        
        # 1. Create session
        response = client.post("/api/v1/sessions", json={"ttl_hours": 2})
        assert response.status_code == 201
        session_id = response.json()["session_id"]
        print(f"\n✓ Created session: {session_id}")

        # 2. Load and add variables from JSON
        with open(SEARCH_SPACE_FILE, 'r') as f:
            search_space = json.load(f)
        
        for var in search_space:
            if var["type"] == "Real":
                payload = {
                    "name": var["name"],
                    "type": "real",
                    "min": var["min"],
                    "max": var["max"]
                }
            elif var["type"] == "Categorical":
                payload = {
                    "name": var["name"],
                    "type": "categorical",
                    "categories": var["values"]
                }
            else:
                raise ValueError(f"Unknown variable type: {var['type']}")
            
            response = client.post(
                f"/api/v1/sessions/{session_id}/variables",
                json=payload
            )
            assert response.status_code == 200
            print(f"✓ Added variable: {var['name']} ({var['type']})")

        # 3. Upload experiments from CSV
        with open(EXPERIMENTS_FILE, 'rb') as f:
            files = {'file': ('catalyst_experiments.csv', f, 'text/csv')}
            response = client.post(
                f"/api/v1/sessions/{session_id}/experiments/upload",
                files=files
            )
        assert response.status_code == 200
        n_experiments = response.json()["n_experiments"]
        print(f"✓ Loaded {n_experiments} experiments from CSV")

        # Get data summary
        response = client.get(f"/api/v1/sessions/{session_id}/experiments/summary")
        assert response.status_code == 200
        summary = response.json()
        print(f"✓ Data summary:")
        print(f"  - Experiments: {summary['n_experiments']}")
        print(f"  - Features: {summary['feature_names']}")
        print(f"  - Target range: [{summary['target_stats']['min']:.4f}, {summary['target_stats']['max']:.4f}]")

        # 4. Train botorch model with Matern 1.5 kernel
        response = client.post(
            f"/api/v1/sessions/{session_id}/model/train",
            json={
                "backend": "botorch",
                "kernel": "matern",
                "kernel_params": {"nu": 1.5},
                "output_transform": "standardize"
            }
        )
        assert response.status_code == 200
        train_results = response.json()
        assert train_results["success"] is True
        print(f"✓ Trained botorch model:")
        print(f"  - Kernel: {train_results['kernel']}")
        print(f"  - R²: {train_results['metrics']['r2']:.4f}")
        print(f"  - RMSE: {train_results['metrics']['rmse']:.4f}")

        # Get model info
        response = client.get(f"/api/v1/sessions/{session_id}/model")
        assert response.status_code == 200
        model_info = response.json()
        assert model_info["is_trained"] is True
        assert model_info["backend"] == "botorch"
        print(f"✓ Model verified")

        # 5. Get acquisition suggestions using LogEI
        response = client.post(
            f"/api/v1/sessions/{session_id}/acquisition/suggest",
            json={
                "strategy": "LogEI",
                "goal": "maximize",
                "n_suggestions": 1
            }
        )
        assert response.status_code == 200
        suggestions = response.json()
        assert suggestions["n_suggestions"] == 1
        suggestion = suggestions["suggestions"][0]
        print(f"✓ Generated suggestion using LogEI:")
        for key, value in suggestion.items():
            print(f"  - {key}: {value}")

        # 6. Make predictions on suggested point
        response = client.post(
            f"/api/v1/sessions/{session_id}/model/predict",
            json={"inputs": [suggestion]}
        )
        assert response.status_code == 200
        predictions = response.json()
        pred = predictions["predictions"][0]
        print(f"✓ Prediction at suggested point:")
        print(f"  - Mean: {pred['prediction']:.4f}")
        print(f"  - Uncertainty: {pred['uncertainty']:.4f}")

        # 7. Clean up
        response = client.delete(f"/api/v1/sessions/{session_id}")
        assert response.status_code == 204
        print(f"✓ Session deleted\n")

    def test_compare_backends_catalyst_data(self):
        """Compare sklearn vs botorch predictions on same catalyst data."""
        
        print("\n" + "="*70)
        print("BACKEND COMPARISON TEST")
        print("="*70)
        
        results = {}
        
        for backend in ["sklearn", "botorch"]:
            print(f"\nTesting {backend.upper()} backend...")
            
            # Create session
            response = client.post("/api/v1/sessions", json={"ttl_hours": 2})
            session_id = response.json()["session_id"]
            
            # Add variables
            with open(SEARCH_SPACE_FILE, 'r') as f:
                search_space = json.load(f)
            
            for var in search_space:
                if var["type"] == "Real":
                    payload = {
                        "name": var["name"],
                        "type": "real",
                        "min": var["min"],
                        "max": var["max"]
                    }
                elif var["type"] == "Categorical":
                    payload = {
                        "name": var["name"],
                        "type": "categorical",
                        "categories": var["values"]
                    }
                
                response = client.post(
                    f"/api/v1/sessions/{session_id}/variables",
                    json=payload
                )
                assert response.status_code == 200
            
            # Upload data
            with open(EXPERIMENTS_FILE, 'rb') as f:
                files = {'file': ('catalyst_experiments.csv', f, 'text/csv')}
                response = client.post(
                    f"/api/v1/sessions/{session_id}/experiments/upload",
                    files=files
                )
            
            # Train model
            response = client.post(
                f"/api/v1/sessions/{session_id}/model/train",
                json={
                    "backend": backend,
                    "kernel": "matern",
                    "kernel_params": {"nu": 1.5},
                    "output_transform": "standardize"
                }
            )
            train_results = response.json()
            
            # Get suggestion
            strategy = "LogEI" if backend == "botorch" else "EI"
            response = client.post(
                f"/api/v1/sessions/{session_id}/acquisition/suggest",
                json={
                    "strategy": strategy,
                    "goal": "maximize",
                    "n_suggestions": 1
                }
            )
            suggestion = response.json()["suggestions"][0]
            
            # Make prediction
            response = client.post(
                f"/api/v1/sessions/{session_id}/model/predict",
                json={"inputs": [suggestion]}
            )
            prediction = response.json()["predictions"][0]
            
            # Store results
            results[backend] = {
                "r2": train_results["metrics"]["r2"],
                "rmse": train_results["metrics"]["rmse"],
                "suggestion": suggestion,
                "prediction": prediction["prediction"],
                "uncertainty": prediction["uncertainty"]
            }
            
            # Clean up
            client.delete(f"/api/v1/sessions/{session_id}")
        
        # Print comparison
        print("\n" + "="*70)
        print("RESULTS COMPARISON")
        print("="*70)
        
        print(f"\nModel Performance:")
        print(f"  sklearn R²:   {results['sklearn']['r2']:.4f}")
        print(f"  botorch R²:   {results['botorch']['r2']:.4f}")
        print(f"  sklearn RMSE: {results['sklearn']['rmse']:.4f}")
        print(f"  botorch RMSE: {results['botorch']['rmse']:.4f}")
        
        print(f"\nSuggested Next Experiments:")
        print(f"  sklearn:")
        for key, value in results['sklearn']['suggestion'].items():
            print(f"    - {key}: {value}")
        print(f"  botorch:")
        for key, value in results['botorch']['suggestion'].items():
            print(f"    - {key}: {value}")
        
        print(f"\nPredictions at Suggested Points:")
        print(f"  sklearn: {results['sklearn']['prediction']:.4f} ± {results['sklearn']['uncertainty']:.4f}")
        print(f"  botorch: {results['botorch']['prediction']:.4f} ± {results['botorch']['uncertainty']:.4f}")
        
        print("\n" + "="*70 + "\n")
