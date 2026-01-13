"""
Unit tests for regret plot visualization.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from alchemist_core.session import OptimizationSession
from alchemist_core.visualization.plots import create_regret_plot


class TestRegretPlot:
    """Test regret plot functionality."""
    
    def test_create_regret_plot_maximize(self):
        """Test basic regret plot for maximization."""
        iterations = np.arange(10)
        values = np.array([1, 3, 2, 5, 4, 7, 6, 8, 7.5, 9])
        
        fig, ax = create_regret_plot(iterations, values, goal='maximize')
        
        assert fig is not None
        assert ax is not None
        assert ax.get_xlabel() == "Experiment Number"
        assert "Objective Function Value" in ax.get_ylabel()
        
        plt.close(fig)
    
    def test_create_regret_plot_minimize(self):
        """Test basic regret plot for minimization."""
        iterations = np.arange(10)
        values = np.array([9, 7, 8, 5, 6, 3, 4, 2, 2.5, 1])
        
        fig, ax = create_regret_plot(iterations, values, goal='minimize')
        
        assert fig is not None
        assert ax is not None
        assert ax.get_xlabel() == "Experiment Number"
        assert "Objective Function Value" in ax.get_ylabel()
        
        plt.close(fig)
    
    def test_regret_plot_shows_all_observations(self):
        """Test that regret plot shows all observations (not just cumulative best)."""
        iterations = np.arange(5)
        values = np.array([1, 5, 3, 7, 4])
        
        fig, ax = create_regret_plot(iterations, values, goal='maximize')
        
        # Get the scatter data - regret plot shows ALL observations, not cumulative best
        collections = ax.collections
        assert len(collections) > 0
        offsets = collections[0].get_offsets()
        y_data = offsets[:, 1]  # Get y values
        
        # Should show all raw values, not cumulative best
        np.testing.assert_array_equal(y_data, values)
        
        plt.close(fig)
    
    def test_regret_plot_with_custom_params(self):
        """Test regret plot with custom parameters."""
        iterations = np.arange(5)
        values = np.array([1, 2, 3, 4, 5])
        
        fig, ax = create_regret_plot(
            iterations, values,
            goal='maximize',
            figsize=(10, 8),
            dpi=150,
            title="Custom Title"
        )
        
        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 8
        assert ax.get_title() == "Custom Title"
        
        plt.close(fig)
    
    def test_regret_plot_with_existing_axes(self):
        """Test regret plot on existing axes."""
        fig, ax = plt.subplots()
        
        iterations = np.arange(5)
        values = np.array([1, 2, 3, 4, 5])
        
        returned_fig, returned_ax = create_regret_plot(
            iterations, values,
            goal='maximize',
            ax=ax
        )
        
        assert returned_fig is fig
        assert returned_ax is ax
        
        plt.close(fig)


class TestSessionRegretPlot:
    """Test regret plot integration with OptimizationSession."""
    
    def test_session_plot_regret_basic(self):
        """Test plot_regret method on session."""
        session = OptimizationSession()
        session.add_variable('x', 'real', bounds=(0, 10))
        
        # Add experiments
        for i in range(5):
            session.add_experiment({'x': i}, output=i**2)
        
        fig = session.plot_regret(goal='maximize')
        
        assert fig is not None
        plt.close(fig)
    
    def test_session_plot_regret_minimize(self):
        """Test plot_regret for minimization."""
        session = OptimizationSession()
        session.add_variable('x', 'real', bounds=(0, 10))
        
        # Add experiments
        for i in range(5):
            session.add_experiment({'x': i}, output=10 - i**2)
        
        fig = session.plot_regret(goal='minimize')
        
        assert fig is not None
        plt.close(fig)
    
    def test_session_plot_regret_insufficient_data(self):
        """Test error with insufficient experiments."""
        session = OptimizationSession()
        session.add_variable('x', 'real', bounds=(0, 10))
        
        # Add only one experiment
        session.add_experiment({'x': 5}, output=25)
        
        with pytest.raises(ValueError, match="Need at least 2 experiments"):
            session.plot_regret(goal='maximize')
    
    def test_session_plot_regret_no_matplotlib(self, monkeypatch):
        """Test error when matplotlib not available."""
        session = OptimizationSession()
        session.add_variable('x', 'real', bounds=(0, 10))
        
        for i in range(5):
            session.add_experiment({'x': i}, output=i)
        
        # Mock matplotlib import failure
        def mock_check_matplotlib():
            raise ImportError("Matplotlib not available")
        
        monkeypatch.setattr(session, '_check_matplotlib', mock_check_matplotlib)
        
        with pytest.raises(ImportError):
            session.plot_regret()


class TestProbabilityOfImprovementPlot:
    """Test probability of improvement convergence plot."""
    
    def test_session_plot_pi_basic(self):
        """Test plot_probability_of_improvement method on session."""
        session = OptimizationSession()
        session.add_variable('x', 'real', bounds=(0, 10))
        
        # Add experiments
        np.random.seed(42)
        for i in range(10):
            x = np.random.uniform(0, 10)
            output = -(x - 5)**2  # Maximum at x=5
            session.add_experiment({'x': x}, output=output)
        
        # Train model first (realistic usage)
        session.train_model(backend='sklearn', kernel='RBF')
        
        # Should work with default parameters (uses trained model's backend/kernel)
        fig = session.plot_probability_of_improvement(
            goal='maximize',
            n_grid_points=50,  # Use fewer points for speed
            start_iteration=5
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_session_plot_pi_insufficient_data(self):
        """Test error with insufficient experiments."""
        session = OptimizationSession()
        session.add_variable('x', 'real', bounds=(0, 10))
        
        # Add only 3 experiments
        for i in range(3):
            session.add_experiment({'x': i}, output=i)
        
        with pytest.raises(ValueError, match="Need at least 5 experiments"):
            session.plot_probability_of_improvement(start_iteration=5)

