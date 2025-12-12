
import pytest
import os
import pandas as pd
from alchemist_core import OptimizationSession

@pytest.fixture
def trained_session_sklearn():
    """Create session with trained sklearn model using catalyst data."""
    session = OptimizationSession()
    
    # Load search space from JSON
    # Go up 4 levels: acquisition -> core -> unit -> tests
    tests_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    search_space_path = os.path.join(tests_dir, 'catalyst_search_space.json')
    session.load_search_space(search_space_path)
    
    # Load experiments from CSV
    experiments_path = os.path.join(tests_dir, 'catalyst_experiments.csv')
    session.load_data(experiments_path)
    
    # Train model
    session.train_model(backend='sklearn', kernel='Matern')
    
    return session

class TestSkoptExtended:
    """Extended tests for Skopt acquisition functions."""

    def test_gp_hedge(self, trained_session_sklearn):
        """Test GP-Hedge acquisition strategy."""
        candidates = trained_session_sklearn.suggest_next(
            strategy='gp_hedge',
            n_suggestions=1,
            goal='maximize'
        )
        assert isinstance(candidates, pd.DataFrame)
        assert len(candidates) == 1

    def test_invalid_strategy(self, trained_session_sklearn):
        """Test invalid strategy raises error."""
        with pytest.raises((ValueError, KeyError)):
            trained_session_sklearn.suggest_next(
                strategy='invalid_strategy',
                n_suggestions=1
            )
