"""
Simple test runner for ALchemist Core data layer.
This verifies the basic functionality without requiring pytest.
"""

import sys
import os

# Add alchemist_core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def test_imports():
    """Test that we can import from alchemist_core"""
    print("Testing imports...")
    
    try:
        from alchemist_core import SearchSpace, ExperimentManager
        print("✓ Successfully imported SearchSpace and ExperimentManager from alchemist_core")
    except ImportError as e:
        print(f"✗ Failed to import from alchemist_core: {e}")
        return False
    
    try:
        from alchemist_core.data import SearchSpace as SS, ExperimentManager as EM
        print("✓ Successfully imported from alchemist_core.data submodule")
    except ImportError as e:
        print(f"✗ Failed to import from alchemist_core.data: {e}")
        return False
    
    return True


def test_search_space():
    """Test basic SearchSpace functionality"""
    print("\nTesting SearchSpace...")
    
    from alchemist_core import SearchSpace
    
    # Test 1: Create empty space
    space = SearchSpace()
    assert len(space) == 0, "Empty space should have length 0"
    print("✓ Created empty SearchSpace")
    
    # Test 2: Add real variable
    space.add_variable("temperature", "real", min=100, max=200)
    assert len(space) == 1, "Space should have 1 variable"
    assert "temperature" in space.get_variable_names()
    print("✓ Added real variable")
    
    # Test 3: Add integer variable
    space.add_variable("iterations", "integer", min=10, max=100)
    assert len(space) == 2
    assert "iterations" in space.get_integer_variables()
    print("✓ Added integer variable")
    
    # Test 4: Add categorical variable
    space.add_variable("catalyst", "categorical", values=["A", "B", "C"])
    assert len(space) == 3
    assert "catalyst" in space.get_categorical_variables()
    print("✓ Added categorical variable")
    
    # Test 5: to_dict and from_dict
    space_dict = space.to_dict()
    new_space = SearchSpace().from_dict(space_dict)
    assert len(new_space) == 3
    assert new_space.get_variable_names() == space.get_variable_names()
    print("✓ Serialization to/from dict works")
    
    return True


def test_experiment_manager():
    """Test basic ExperimentManager functionality"""
    print("\nTesting ExperimentManager...")
    
    from alchemist_core import ExperimentManager
    
    # Test 1: Create empty manager
    manager = ExperimentManager()
    assert len(manager) == 0
    print("✓ Created empty ExperimentManager")
    
    # Test 2: Add single experiment
    manager.add_experiment(
        {"temp": 150, "pressure": 5},
        output_value=42.0
    )
    assert len(manager) == 1
    assert "Output" in manager.df.columns
    print("✓ Added single experiment")
    
    # Test 3: Add experiment with noise
    manager.add_experiment(
        {"temp": 160, "pressure": 6},
        output_value=45.0,
        noise_value=0.1
    )
    assert manager.has_noise_data()
    print("✓ Added experiment with noise")
    
    # Test 4: Get features and target
    X, y = manager.get_features_and_target()
    assert len(X) == 2
    assert len(y) == 2
    assert "Output" not in X.columns
    print("✓ Extracted features and target")
    
    # Test 5: Clear
    manager.clear()
    assert len(manager) == 0
    print("✓ Cleared experiments")
    
    return True


def test_integration():
    """Test SearchSpace and ExperimentManager together"""
    print("\nTesting integration...")
    
    from alchemist_core import SearchSpace, ExperimentManager
    
    # Create search space
    space = SearchSpace()
    space.add_variable("temp", "real", min=100, max=200)
    space.add_variable("pressure", "integer", min=1, max=10)
    
    # Create manager with search space
    manager = ExperimentManager(search_space=space)
    manager.add_experiment({"temp": 150, "pressure": 5}, output_value=42.0)
    
    X, y = manager.get_features_and_target()
    assert list(X.columns) == space.get_variable_names()
    print("✓ SearchSpace and ExperimentManager work together")
    
    return True



def main():
    """Run all tests"""
    print("=" * 60)
    print("ALchemist Core - Data Layer Tests")
    print("=" * 60)
    
    all_passed = True
    
    tests = [
        ("Import Test", test_imports),
        ("SearchSpace Test", test_search_space),
        ("ExperimentManager Test", test_experiment_manager),
        ("Integration Test", test_integration),
    ]
    
    for test_name, test_func in tests:
        try:
            if not test_func():
                print(f"\n✗ {test_name} FAILED")
                all_passed = False
        except Exception as e:
            print(f"\n✗ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
