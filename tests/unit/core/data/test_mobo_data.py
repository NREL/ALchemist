"""
Unit tests for multi-objective optimization data methods in ExperimentManager.

Tests Pareto frontier computation and hypervolume calculation.
"""

import pytest
import pandas as pd
import numpy as np
from alchemist_core.data.experiment_manager import ExperimentManager


class TestParetoFrontier:
    """Tests for get_pareto_frontier() method."""
    
    def test_pareto_frontier_empty_data(self):
        """Test Pareto frontier with no data returns empty DataFrame."""
        exp_mgr = ExperimentManager(target_columns=['obj1', 'obj2'])
        pareto = exp_mgr.get_pareto_frontier()
        
        assert len(pareto) == 0
        assert isinstance(pareto, pd.DataFrame)
    
    def test_pareto_frontier_single_objective(self):
        """Test Pareto frontier with single objective returns all data."""
        exp_mgr = ExperimentManager(target_columns=['yield'])
        
        # Add some data
        exp_mgr.add_experiment({'x': 1.0}, output_value=10.0)
        exp_mgr.add_experiment({'x': 2.0}, output_value=20.0)
        exp_mgr.add_experiment({'x': 3.0}, output_value=15.0)
        
        pareto = exp_mgr.get_pareto_frontier()
        
        # Single objective: all points are on the "Pareto frontier"
        assert len(pareto) == 3
    
    def test_pareto_frontier_two_objectives_maximize_both(self):
        """Test Pareto frontier with 2 objectives, both maximized."""
        exp_mgr = ExperimentManager(target_columns=['yield', 'selectivity'])
        
        # Add test data:
        # Point 1: (10, 50) - dominated by point 3 and 4
        # Point 2: (20, 55) - Pareto optimal
        # Point 3: (25, 50) - Pareto optimal
        # Point 4: (30, 45) - Pareto optimal
        # Point 5: (15, 40) - dominated by all Pareto points
        
        data = [
            ({'x': 1.0}, {'yield': 10.0, 'selectivity': 50.0}),
            ({'x': 2.0}, {'yield': 20.0, 'selectivity': 55.0}),
            ({'x': 3.0}, {'yield': 25.0, 'selectivity': 50.0}),
            ({'x': 4.0}, {'yield': 30.0, 'selectivity': 45.0}),
            ({'x': 5.0}, {'yield': 15.0, 'selectivity': 40.0}),
        ]
        
        for inputs, outputs in data:
            point = inputs.copy()
            point.update(outputs)
            exp_mgr.add_experiment(point)
        
        pareto = exp_mgr.get_pareto_frontier(directions=['maximize', 'maximize'])
        
        assert len(pareto) == 3  # Points 2, 3, 4 are non-dominated
        assert set(pareto['x'].values) == {2.0, 3.0, 4.0}
    
    def test_pareto_frontier_mixed_directions(self):
        """Test Pareto frontier with mixed maximize/minimize objectives."""
        exp_mgr = ExperimentManager(target_columns=['yield', 'cost'])
        
        # Add test data (maximize yield, minimize cost):
        # Point 1: (10, 100) - dominated
        # Point 2: (20, 80) - Pareto optimal
        # Point 3: (15, 70) - Pareto optimal
        # Point 4: (25, 90) - Pareto optimal
        # Point 5: (12, 95) - dominated
        
        data = [
            ({'x': 1.0}, {'yield': 10.0, 'cost': 100.0}),
            ({'x': 2.0}, {'yield': 20.0, 'cost': 80.0}),
            ({'x': 3.0}, {'yield': 15.0, 'cost': 70.0}),
            ({'x': 4.0}, {'yield': 25.0, 'cost': 90.0}),
            ({'x': 5.0}, {'yield': 12.0, 'cost': 95.0}),
        ]
        
        for inputs, outputs in data:
            point = inputs.copy()
            point.update(outputs)
            exp_mgr.add_experiment(point)
        
        pareto = exp_mgr.get_pareto_frontier(directions=['maximize', 'minimize'])
        
        assert len(pareto) == 3  # Points 2, 3, 4
        assert set(pareto['x'].values) == {2.0, 3.0, 4.0}
    
    def test_pareto_frontier_three_objectives(self):
        """Test Pareto frontier with 3 objectives."""
        exp_mgr = ExperimentManager(target_columns=['obj1', 'obj2', 'obj3'])
        
        # Add test data
        data = [
            ({'x': 1.0}, {'obj1': 1.0, 'obj2': 1.0, 'obj3': 1.0}),  # Dominated
            ({'x': 2.0}, {'obj1': 2.0, 'obj2': 2.0, 'obj3': 0.5}),  # Pareto
            ({'x': 3.0}, {'obj1': 1.5, 'obj2': 3.0, 'obj3': 0.8}),  # Pareto
            ({'x': 4.0}, {'obj1': 3.0, 'obj2': 1.5, 'obj3': 1.2}),  # Pareto
        ]
        
        for inputs, outputs in data:
            point = inputs.copy()
            point.update(outputs)
            exp_mgr.add_experiment(point)
        
        pareto = exp_mgr.get_pareto_frontier(directions=['maximize', 'maximize', 'maximize'])
        
        assert len(pareto) == 3  # Points 2, 3, 4
    
    def test_pareto_frontier_duplicates_removed(self):
        """Test that duplicate Pareto points are removed."""
        exp_mgr = ExperimentManager(target_columns=['yield', 'selectivity'])
        
        # Add duplicate points
        for _ in range(3):
            exp_mgr.add_experiment({'x': 1.0, 'yield': 10.0, 'selectivity': 50.0})
        
        pareto = exp_mgr.get_pareto_frontier()
        
        # Should only keep one duplicate
        assert len(pareto) == 1
    
    def test_pareto_frontier_default_directions(self):
        """Test Pareto frontier uses maximize by default."""
        exp_mgr = ExperimentManager(target_columns=['obj1', 'obj2'])
        
        exp_mgr.add_experiment({'x': 1.0, 'obj1': 10.0, 'obj2': 50.0})
        exp_mgr.add_experiment({'x': 2.0, 'obj1': 20.0, 'obj2': 40.0})
        exp_mgr.add_experiment({'x': 3.0, 'obj1': 30.0, 'obj2': 60.0})
        
        pareto = exp_mgr.get_pareto_frontier()  # No directions specified
        
        # Point 3 dominates all (30, 60)
        assert len(pareto) >= 1
        assert 3.0 in pareto['x'].values
    
    def test_pareto_frontier_invalid_directions_length(self):
        """Test error when directions length doesn't match target columns."""
        exp_mgr = ExperimentManager(target_columns=['obj1', 'obj2'])
        exp_mgr.add_experiment({'x': 1.0, 'obj1': 10.0, 'obj2': 50.0})
        
        with pytest.raises(ValueError, match="Number of directions"):
            exp_mgr.get_pareto_frontier(directions=['maximize'])
    
    def test_pareto_frontier_missing_target_columns(self):
        """Test error when target columns not in data."""
        exp_mgr = ExperimentManager(target_columns=['obj1', 'obj2'])
        exp_mgr.add_experiment({'x': 1.0, 'obj1': 10.0})  # Missing obj2
        
        with pytest.raises(ValueError, match="Target columns .* not found"):
            exp_mgr.get_pareto_frontier()
    
    def test_pareto_frontier_with_nan_values(self):
        """Test error when target columns contain NaN."""
        exp_mgr = ExperimentManager(target_columns=['obj1', 'obj2'])
        exp_mgr.df = pd.DataFrame({
            'x': [1.0, 2.0],
            'obj1': [10.0, np.nan],
            'obj2': [50.0, 40.0]
        })
        
        with pytest.raises(ValueError, match="contain missing values"):
            exp_mgr.get_pareto_frontier()


class TestHypervolume:
    """Tests for compute_hypervolume() method."""
    
    def test_hypervolume_empty_data(self):
        """Test hypervolume with no data returns 0."""
        exp_mgr = ExperimentManager(target_columns=['obj1', 'obj2'])
        hv = exp_mgr.compute_hypervolume([0.0, 0.0])
        
        assert hv == 0.0
    
    def test_hypervolume_single_objective_raises_error(self):
        """Test hypervolume raises error for single objective."""
        exp_mgr = ExperimentManager(target_columns=['yield'])
        exp_mgr.add_experiment({'x': 1.0}, output_value=10.0)
        
        with pytest.raises(ValueError, match="only defined for multi-objective"):
            exp_mgr.compute_hypervolume([0.0])
    
    def test_hypervolume_two_objectives_maximize(self):
        """Test hypervolume computation for 2 objectives (maximize both)."""
        exp_mgr = ExperimentManager(target_columns=['obj1', 'obj2'])
        
        # Add Pareto points
        exp_mgr.add_experiment({'x': 1.0, 'obj1': 2.0, 'obj2': 1.0})
        exp_mgr.add_experiment({'x': 2.0, 'obj1': 1.0, 'obj2': 2.0})
        
        # Reference point below all objectives
        hv = exp_mgr.compute_hypervolume([0.0, 0.0], directions=['maximize', 'maximize'])
        
        # Should be positive (exact value: 2*1 + 1*2 - 1*1 = 3.0)
        assert hv > 0.0
        assert np.isclose(hv, 3.0, atol=0.01)
    
    def test_hypervolume_mixed_directions(self):
        """Test hypervolume with maximize/minimize objectives."""
        exp_mgr = ExperimentManager(target_columns=['yield', 'cost'])
        
        # Maximize yield, minimize cost
        exp_mgr.add_experiment({'x': 1.0, 'yield': 80.0, 'cost': 20.0})
        exp_mgr.add_experiment({'x': 2.0, 'yield': 90.0, 'cost': 30.0})
        exp_mgr.add_experiment({'x': 3.0, 'yield': 70.0, 'cost': 10.0})
        
        # ref_point = [min_yield, max_cost]
        hv = exp_mgr.compute_hypervolume([50.0, 50.0], directions=['maximize', 'minimize'])
        
        assert hv > 0.0
    
    def test_hypervolume_three_objectives(self):
        """Test hypervolume with 3 objectives."""
        exp_mgr = ExperimentManager(target_columns=['obj1', 'obj2', 'obj3'])
        
        exp_mgr.add_experiment({'x': 1.0, 'obj1': 2.0, 'obj2': 2.0, 'obj3': 2.0})
        exp_mgr.add_experiment({'x': 2.0, 'obj1': 3.0, 'obj2': 1.0, 'obj3': 1.0})
        
        hv = exp_mgr.compute_hypervolume([0.0, 0.0, 0.0], 
                                        directions=['maximize', 'maximize', 'maximize'])
        
        assert hv > 0.0
    
    def test_hypervolume_default_directions(self):
        """Test hypervolume uses maximize by default."""
        exp_mgr = ExperimentManager(target_columns=['obj1', 'obj2'])
        
        exp_mgr.add_experiment({'x': 1.0, 'obj1': 2.0, 'obj2': 1.0})
        
        hv = exp_mgr.compute_hypervolume([0.0, 0.0])  # No directions specified
        
        assert hv > 0.0
    
    def test_hypervolume_invalid_ref_point_length(self):
        """Test error when ref_point length doesn't match target columns."""
        exp_mgr = ExperimentManager(target_columns=['obj1', 'obj2'])
        exp_mgr.add_experiment({'x': 1.0, 'obj1': 10.0, 'obj2': 50.0})
        
        with pytest.raises(ValueError, match="Reference point length"):
            exp_mgr.compute_hypervolume([0.0])
    
    def test_hypervolume_increases_with_better_pareto(self):
        """Test that hypervolume increases when Pareto frontier improves."""
        exp_mgr = ExperimentManager(target_columns=['obj1', 'obj2'])
        
        # Initial point
        exp_mgr.add_experiment({'x': 1.0, 'obj1': 1.0, 'obj2': 1.0})
        hv1 = exp_mgr.compute_hypervolume([0.0, 0.0])
        
        # Add dominated point (no change)
        exp_mgr.add_experiment({'x': 2.0, 'obj1': 0.5, 'obj2': 0.5})
        hv2 = exp_mgr.compute_hypervolume([0.0, 0.0])
        assert np.isclose(hv1, hv2)
        
        # Add Pareto-optimal point (should increase)
        exp_mgr.add_experiment({'x': 3.0, 'obj1': 2.0, 'obj2': 2.0})
        hv3 = exp_mgr.compute_hypervolume([0.0, 0.0])
        assert hv3 > hv2
    
    def test_hypervolume_with_list_ref_point(self):
        """Test hypervolume accepts ref_point as list."""
        exp_mgr = ExperimentManager(target_columns=['obj1', 'obj2'])
        exp_mgr.add_experiment({'x': 1.0, 'obj1': 2.0, 'obj2': 1.0})
        
        hv = exp_mgr.compute_hypervolume([0.0, 0.0])
        assert hv > 0.0
    
    def test_hypervolume_with_array_ref_point(self):
        """Test hypervolume accepts ref_point as numpy array."""
        exp_mgr = ExperimentManager(target_columns=['obj1', 'obj2'])
        exp_mgr.add_experiment({'x': 1.0, 'obj1': 2.0, 'obj2': 1.0})
        
        hv = exp_mgr.compute_hypervolume(np.array([0.0, 0.0]))
        assert hv > 0.0


class TestBackwardCompatibility:
    """Tests to ensure new methods don't break existing functionality."""
    
    def test_experiment_manager_init_defaults(self):
        """Test ExperimentManager initializes with default target_columns."""
        exp_mgr = ExperimentManager()
        
        assert exp_mgr.target_columns == ['Output']
    
    def test_single_objective_workflow_unchanged(self):
        """Test that single-objective workflows work as before."""
        exp_mgr = ExperimentManager(target_columns=['yield'])
        
        exp_mgr.add_experiment({'x': 1.0}, output_value=10.0)
        exp_mgr.add_experiment({'x': 2.0}, output_value=20.0)
        
        assert len(exp_mgr.df) == 2
        assert 'yield' in exp_mgr.df.columns
    
    def test_get_features_target_and_noise_unchanged(self):
        """Test existing method still works for single objective."""
        from alchemist_core.data.search_space import SearchSpace
        
        space = SearchSpace()
        space.add_variable('x', 'real', min=0, max=1)
        
        exp_mgr = ExperimentManager(search_space=space, target_columns=['yield'])
        exp_mgr.add_experiment({'x': 0.5}, output_value=10.0, noise_value=0.1)
        
        X, y, noise = exp_mgr.get_features_target_and_noise()
        
        assert X.shape == (1, 1)
        assert y.shape == (1,)
        assert noise.shape == (1,)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
