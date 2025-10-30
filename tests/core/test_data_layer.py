"""
Tests for the data layer of ALchemist Core.

These tests verify that SearchSpace and ExperimentManager work correctly
and can be imported from the core package.
"""

import pytest
import pandas as pd
import numpy as np
from alchemist_core.data import SearchSpace, ExperimentManager


class TestSearchSpace:
    """Test cases for SearchSpace class"""
    
    def test_create_empty_search_space(self):
        """Test creating an empty search space"""
        space = SearchSpace()
        assert len(space) == 0
        assert space.get_variable_names() == []
    
    def test_add_real_variable(self):
        """Test adding a real-valued variable"""
        space = SearchSpace()
        space.add_variable("temperature", "real", min=100, max=200)
        
        assert len(space) == 1
        assert "temperature" in space.get_variable_names()
        assert space.get_categorical_variables() == []
    
    def test_add_integer_variable(self):
        """Test adding an integer variable"""
        space = SearchSpace()
        space.add_variable("iterations", "integer", min=10, max=100)
        
        assert len(space) == 1
        assert "iterations" in space.get_variable_names()
        assert "iterations" in space.get_integer_variables()
    
    def test_add_categorical_variable(self):
        """Test adding a categorical variable"""
        space = SearchSpace()
        space.add_variable("catalyst", "categorical", values=["A", "B", "C"])
        
        assert len(space) == 1
        assert "catalyst" in space.get_variable_names()
        assert "catalyst" in space.get_categorical_variables()
    
    def test_mixed_search_space(self):
        """Test creating a search space with mixed variable types"""
        space = SearchSpace()
        space.add_variable("temp", "real", min=100, max=200)
        space.add_variable("iterations", "integer", min=10, max=100)
        space.add_variable("catalyst", "categorical", values=["A", "B", "C"])
        
        assert len(space) == 3
        assert len(space.get_variable_names()) == 3
        assert len(space.get_categorical_variables()) == 1
        assert len(space.get_integer_variables()) == 1
    
    def test_to_dict_and_from_dict(self):
        """Test serialization to/from dictionary"""
        space = SearchSpace()
        space.add_variable("temp", "real", min=100, max=200)
        space.add_variable("pressure", "integer", min=1, max=10)
        
        # Convert to dict
        space_dict = space.to_dict()
        assert len(space_dict) == 2
        
        # Create new space from dict
        new_space = SearchSpace().from_dict(space_dict)
        assert len(new_space) == 2
        assert new_space.get_variable_names() == space.get_variable_names()


class TestExperimentManager:
    """Test cases for ExperimentManager class"""
    
    def test_create_empty_manager(self):
        """Test creating an empty experiment manager"""
        manager = ExperimentManager()
        assert len(manager) == 0
        assert manager.df.empty
    
    def test_add_single_experiment(self):
        """Test adding a single experiment"""
        manager = ExperimentManager()
        manager.add_experiment(
            {"temp": 150, "pressure": 5},
            output_value=42.0
        )
        
        assert len(manager) == 1
        assert "Output" in manager.df.columns
        assert manager.df["Output"].iloc[0] == 42.0
    
    def test_add_experiment_with_noise(self):
        """Test adding experiment with noise value"""
        manager = ExperimentManager()
        manager.add_experiment(
            {"temp": 150},
            output_value=42.0,
            noise_value=0.1
        )
        
        assert manager.has_noise_data()
        assert "Noise" in manager.df.columns
    
    def test_get_features_and_target(self):
        """Test extracting features and target"""
        manager = ExperimentManager()
        manager.add_experiment({"temp": 150, "pressure": 5}, output_value=42.0)
        manager.add_experiment({"temp": 160, "pressure": 6}, output_value=45.0)
        
        X, y = manager.get_features_and_target()
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == 2
        assert len(y) == 2
        assert "Output" not in X.columns
    
    def test_batch_add_experiments(self):
        """Test adding multiple experiments at once"""
        manager = ExperimentManager()
        
        data = pd.DataFrame({
            "temp": [150, 160, 170],
            "pressure": [5, 6, 7],
            "Output": [42.0, 45.0, 48.0]
        })
        
        manager.add_experiments_batch(data)
        assert len(manager) == 3
    
    def test_clear_experiments(self):
        """Test clearing all experiments"""
        manager = ExperimentManager()
        manager.add_experiment({"temp": 150}, output_value=42.0)
        assert len(manager) == 1
        
        manager.clear()
        assert len(manager) == 0


class TestIntegration:
    """Integration tests for SearchSpace and ExperimentManager together"""
    
    def test_manager_with_search_space(self):
        """Test that ExperimentManager can work with SearchSpace"""
        space = SearchSpace()
        space.add_variable("temp", "real", min=100, max=200)
        space.add_variable("pressure", "integer", min=1, max=10)
        
        manager = ExperimentManager(search_space=space)
        manager.add_experiment({"temp": 150, "pressure": 5}, output_value=42.0)
        
        assert len(manager) == 1
        X, y = manager.get_features_and_target()
        assert list(X.columns) == space.get_variable_names()


def test_core_imports():
    """Test that core package imports work correctly"""
    # These imports should work without error
    from alchemist_core import SearchSpace, ExperimentManager
    
    # Should also work from submodules
    from alchemist_core.data import SearchSpace as SS
    from alchemist_core.data import ExperimentManager as EM
    
    # Verify they're the same classes
    assert SearchSpace is SS
    assert ExperimentManager is EM


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
