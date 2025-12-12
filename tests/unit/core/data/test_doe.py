"""
Test Design of Experiments (DoE) functionality.
"""

import pytest
import numpy as np
from alchemist_core.data.search_space import SearchSpace
from alchemist_core.utils.doe import generate_initial_design
from alchemist_core.session import OptimizationSession


class TestDoE:
    """Test initial design generation methods."""
    
    def setup_method(self):
        """Create a simple search space for testing."""
        self.space = SearchSpace()
        self.space.add_variable('temperature', 'real', min=300, max=500)
        self.space.add_variable('pressure', 'real', min=1, max=10)
        self.space.add_variable('catalyst', 'categorical', values=['A', 'B', 'C'])
    
    def test_random_sampling(self):
        """Test random sampling method."""
        points = generate_initial_design(
            self.space,
            method='random',
            n_points=10,
            random_seed=42
        )
        
        assert len(points) == 10
        assert all('temperature' in p for p in points)
        assert all('pressure' in p for p in points)
        assert all('catalyst' in p for p in points)
        
        # Check bounds
        for point in points:
            assert 300 <= point['temperature'] <= 500
            assert 1 <= point['pressure'] <= 10
            assert point['catalyst'] in ['A', 'B', 'C']
    
    def test_lhs_sampling(self):
        """Test Latin Hypercube Sampling."""
        points = generate_initial_design(
            self.space,
            method='lhs',
            n_points=10,
            random_seed=42,
            lhs_criterion='maximin'
        )
        
        assert len(points) == 10
        # LHS should provide good space coverage
        temps = [p['temperature'] for p in points]
        assert max(temps) - min(temps) > 100  # Should span the space
    
    def test_sobol_sampling(self):
        """Test Sobol sequence sampling."""
        points = generate_initial_design(
            self.space,
            method='sobol',
            n_points=10
        )
        
        assert len(points) == 10
    
    def test_halton_sampling(self):
        """Test Halton sequence sampling."""
        points = generate_initial_design(
            self.space,
            method='halton',
            n_points=10
        )
        
        assert len(points) == 10
    
    def test_hammersly_sampling(self):
        """Test Hammersly sequence sampling."""
        points = generate_initial_design(
            self.space,
            method='hammersly',
            n_points=10
        )
        
        assert len(points) == 10
    
    def test_invalid_method(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown sampling method"):
            generate_initial_design(self.space, method='invalid_method')
    
    def test_empty_search_space(self):
        """Test that empty search space raises error."""
        empty_space = SearchSpace()
        with pytest.raises(ValueError, match="no variables"):
            generate_initial_design(empty_space, method='lhs')
    
    def test_reproducibility_with_seed(self):
        """Test that random seed produces reproducible results."""
        points1 = generate_initial_design(
            self.space,
            method='random',
            n_points=5,
            random_seed=123
        )
        
        points2 = generate_initial_design(
            self.space,
            method='random',
            n_points=5,
            random_seed=123
        )
        
        # Should be identical
        for p1, p2 in zip(points1, points2):
            assert p1['temperature'] == p2['temperature']
            assert p1['pressure'] == p2['pressure']
            assert p1['catalyst'] == p2['catalyst']


class TestSessionDoE:
    """Test DoE integration with OptimizationSession."""
    
    def test_session_generate_initial_design(self):
        """Test generate_initial_design() method on Session API."""
        session = OptimizationSession()
        session.add_variable('temp', 'real', bounds=(300, 500))
        session.add_variable('flow', 'real', bounds=(1, 10))
        
        points = session.generate_initial_design(method='lhs', n_points=8)
        
        assert len(points) == 8
        assert all('temp' in p and 'flow' in p for p in points)
    
    def test_session_no_variables_error(self):
        """Test that session raises error if no variables defined."""
        session = OptimizationSession()
        
        with pytest.raises(ValueError, match="No variables defined"):
            session.generate_initial_design()
    
    def test_workflow_initial_design_then_add(self):
        """Test complete workflow: generate design, add experiments, train."""
        session = OptimizationSession()
        session.add_variable('x', 'real', min=0, max=10)
        session.add_variable('y', 'real', min=0, max=10)
        
        # Generate initial design
        points = session.generate_initial_design('lhs', n_points=10, random_seed=42)
        
        # Simulate experiments (use simple function: z = x + y)
        for point in points:
            output = point['x'] + point['y']
            session.add_experiment(point, output=output)
        
        # Verify data was added
        assert len(session.experiment_manager.df) == 10
        
        # Train model
        result = session.train_model(backend='sklearn', kernel='rbf')
        assert result is not None
        assert session.model is not None
