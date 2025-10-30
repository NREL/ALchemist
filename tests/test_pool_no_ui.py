"""
Test that logic/pool.py has no UI dependencies.
"""

def test_pool_imports():
    """Test that pool module can be imported without UI dependencies."""
    try:
        from logic import pool
        print("✓ logic.pool imports successfully without UI dependencies")
        return True
    except ImportError as e:
        print(f"✗ Failed to import logic.pool: {e}")
        return False


def test_no_ctk_in_pool():
    """Test that CTkMessagebox is not imported in pool module."""
    from logic import pool
    import inspect
    
    source = inspect.getsource(pool)
    assert 'CTkMessagebox' not in source, "CTkMessagebox should not be in pool.py"
    assert 'customtkinter' not in source.lower(), "customtkinter should not be in pool.py"
    print("✓ No UI dependencies found in pool.py source code")
    return True


def test_generate_pool_works():
    """Test that generate_pool function still works."""
    from logic.pool import generate_pool
    from skopt.space import Real, Integer, Categorical
    import pandas as pd
    
    # Create a simple search space
    search_space = [
        Real(0.0, 1.0, name="x1"),
        Real(0.0, 1.0, name="x2"),
        Categorical(["A", "B"], name="cat")
    ]
    
    # Generate pool
    pool = generate_pool(search_space, pool_size=100, lhs_iterations=5)
    
    assert isinstance(pool, pd.DataFrame), "Pool should be a DataFrame"
    assert len(pool) == 100, "Pool should have 100 points"
    assert set(pool.columns) == {"x1", "x2", "cat"}, "Pool should have correct columns"
    print("✓ generate_pool() works correctly")
    return True


def test_load_search_space_error_handling():
    """Test that load_search_space_from_file raises proper exceptions."""
    from logic.pool import load_search_space_from_file
    import pytest
    
    try:
        # Try to load non-existent file
        load_search_space_from_file("nonexistent_file.json")
        print("✗ Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"✓ Correctly raises ValueError: {e}")
        return True
    except Exception as e:
        print(f"✗ Unexpected exception type: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("Testing pool.py UI independence")
    print("="*60)
    
    tests = [
        test_pool_imports,
        test_no_ctk_in_pool,
        test_generate_pool_works,
        test_load_search_space_error_handling,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ {test.__name__} failed with exception: {e}")
            results.append(False)
        print()
    
    print("="*60)
    if all(results):
        print("All tests passed!")
    else:
        print(f"Some tests failed: {sum(results)}/{len(results)} passed")
    print("="*60)
