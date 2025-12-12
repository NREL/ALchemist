
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from alchemist_core.acquisition.skopt_acquisition import SkoptAcquisition
from alchemist_core.data.search_space import SearchSpace
from skopt.space import Real, Integer, Categorical

class MockModel:
    def __init__(self, predict_func=None):
        self.predict_func = predict_func or (lambda x: np.zeros(len(x)))
        self.model = MagicMock()  # Mock internal model

    def predict(self, X):
        return self.predict_func(X)

    def predict_with_std(self, X):
        preds = self.predict(X)
        return preds, np.zeros_like(preds)

class TestSkoptOptimum:
    def test_find_optimum_continuous(self):
        # Setup search space
        search_space = SearchSpace()
        search_space.add_variable("x1", "real", min=0.0, max=10.0)
        search_space.add_variable("x2", "real", min=0.0, max=10.0)
        
        # Setup acquisition
        acq = SkoptAcquisition(search_space)
        
        # Setup mock model with a simple quadratic function to optimize
        # Maximize -(x1-5)^2 - (x2-5)^2 + 10 -> Optimum at (5, 5) with value 10
        def objective(X):
            if isinstance(X, pd.DataFrame):
                X = X.values
            return -(X[:, 0] - 5)**2 - (X[:, 1] - 5)**2 + 10
            
        model = MockModel(predict_func=objective)
        
        # Find optimum
        result = acq.find_optimum(model, maximize=True)
        
        # Check results
        assert result['value'] is not None
        assert np.isclose(result['value'], 10.0, atol=0.1)
        
        x_opt = result['x_opt']
        assert isinstance(x_opt, pd.DataFrame)
        assert np.isclose(x_opt['x1'].iloc[0], 5.0, atol=0.1)
        assert np.isclose(x_opt['x2'].iloc[0], 5.0, atol=0.1)

    def test_find_optimum_categorical_only(self):
        # Setup search space
        search_space = SearchSpace()
        search_space.add_variable("c1", "categorical", values=["A", "B"])
        search_space.add_variable("c2", "categorical", values=["X", "Y"])
        
        acq = SkoptAcquisition(search_space)
        
        # Define objective for combinations
        # A, X -> 1
        # A, Y -> 2
        # B, X -> 3
        # B, Y -> 4 (Best)
        def objective(X):
            results = []
            for _, row in X.iterrows():
                val = 0
                if row['c1'] == 'B': val += 2
                if row['c2'] == 'Y': val += 1
                results.append(val + 1)
            return np.array(results)
            
        model = MockModel(predict_func=objective)
        
        # Find optimum
        result = acq.find_optimum(model, maximize=True)
        
        assert result['value'] == 4
        assert result['x_opt']['c1'].iloc[0] == 'B'
        assert result['x_opt']['c2'].iloc[0] == 'Y'

    def test_find_optimum_mixed(self):
        # Setup search space
        search_space = SearchSpace()
        search_space.add_variable("x1", "real", min=0.0, max=10.0)
        search_space.add_variable("c1", "categorical", values=["A", "B"])
        
        acq = SkoptAcquisition(search_space)
        
        # Objective:
        # If A: -(x-2)^2 + 5 (Max 5 at x=2)
        # If B: -(x-8)^2 + 10 (Max 10 at x=8) -> Global optimum
        def objective(X):
            results = []
            for _, row in X.iterrows():
                x = row['x1']
                cat = row['c1']
                if cat == 'A':
                    val = -(x - 2)**2 + 5
                else:
                    val = -(x - 8)**2 + 10
                results.append(val)
            return np.array(results)
            
        model = MockModel(predict_func=objective)
        
        # Find optimum
        result = acq.find_optimum(model, maximize=True)
        
        assert np.isclose(result['value'], 10.0, atol=0.1)
        assert result['x_opt']['c1'].iloc[0] == 'B'
        assert np.isclose(result['x_opt']['x1'].iloc[0], 8.0, atol=0.1)

    def test_find_optimum_minimize(self):
        # Setup search space
        search_space = SearchSpace()
        search_space.add_variable("x1", "real", min=0.0, max=10.0)
        
        acq = SkoptAcquisition(search_space)
        
        # Minimize (x-3)^2 -> Min 0 at x=3
        def objective(X):
            if isinstance(X, pd.DataFrame):
                X = X.values
            return (X[:, 0] - 3)**2
            
        model = MockModel(predict_func=objective)
        
        # Find optimum (minimize)
        result = acq.find_optimum(model, maximize=False)
        
        assert np.isclose(result['value'], 0.0, atol=0.1)
        assert np.isclose(result['x_opt']['x1'].iloc[0], 3.0, atol=0.1)

    def test_get_categorical_combinations(self):
        # Setup search space
        search_space = SearchSpace()
        search_space.add_variable("c1", "categorical", values=["A", "B"])
        search_space.add_variable("c2", "categorical", values=["X", "Y"])
        
        acq = SkoptAcquisition(search_space)
        
        # Access private method for testing
        # Dimensions are 0 and 1
        cat_dims = {0: ["A", "B"], 1: ["X", "Y"]}
        combos = acq._get_categorical_combinations(cat_dims)
        
        assert len(combos) == 4
        expected = [
            {0: 'A', 1: 'X'},
            {0: 'A', 1: 'Y'},
            {0: 'B', 1: 'X'},
            {0: 'B', 1: 'Y'}
        ]
        # Sort by values to compare
        def sort_key(d): return str(sorted(d.items()))
        assert sorted(combos, key=sort_key) == sorted(expected, key=sort_key)

    def test_find_optimum_no_std(self):
        # Test model without predict_with_std
        search_space = SearchSpace()
        search_space.add_variable("x1", "real", min=0.0, max=10.0)
        acq = SkoptAcquisition(search_space)
        
        class SimpleModel:
            def __init__(self):
                self.model = MagicMock()
            def predict(self, X):
                return np.array([5.0])
        
        model = SimpleModel()
        result = acq.find_optimum(model)
        
        assert result['value'] == 5.0
        assert result['std'] is None
