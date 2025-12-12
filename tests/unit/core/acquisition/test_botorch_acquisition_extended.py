
import pytest
import os
import pandas as pd
import torch
import numpy as np
from alchemist_core import OptimizationSession
from alchemist_core.acquisition.botorch_acquisition import BoTorchAcquisition

@pytest.fixture
def trained_session_botorch():
    """Create session with trained BoTorch model using catalyst data."""
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
    session.train_model(backend='botorch', kernel='Matern')
    
    return session

class TestBoTorchExtended:
    """Extended tests for BoTorch acquisition functions."""

    def test_log_ei(self, trained_session_botorch):
        """Test Log Expected Improvement."""
        candidates = trained_session_botorch.suggest_next(
            strategy='logei',
            n_suggestions=1,
            goal='maximize'
        )
        assert isinstance(candidates, pd.DataFrame)
        assert len(candidates) == 1
        
    def test_log_pi(self, trained_session_botorch):
        """Test Log Probability of Improvement."""
        candidates = trained_session_botorch.suggest_next(
            strategy='logpi',
            n_suggestions=1,
            goal='maximize'
        )
        assert isinstance(candidates, pd.DataFrame)
        assert len(candidates) == 1

    def test_qei_batch(self, trained_session_botorch):
        """Test Batch Expected Improvement (qEI)."""
        # qEI is designed for batch selection
        candidates = trained_session_botorch.suggest_next(
            strategy='qei',
            n_suggestions=3,
            goal='maximize'
        )
        assert isinstance(candidates, pd.DataFrame)
        # Should return exactly 3 candidates for q-methods usually
        assert len(candidates) == 3

    def test_qucb_batch(self, trained_session_botorch):
        """Test Batch Upper Confidence Bound (qUCB)."""
        candidates = trained_session_botorch.suggest_next(
            strategy='qucb',
            n_suggestions=3,
            goal='maximize',
            beta=0.5
        )
        assert isinstance(candidates, pd.DataFrame)
        assert len(candidates) == 3

    def test_qipv(self, trained_session_botorch):
        """Test q-Negative Integrated Posterior Variance (qIPV)."""
        # This is computationally expensive, so we test with small batch
        candidates = trained_session_botorch.suggest_next(
            strategy='qipv',
            n_suggestions=2,
            goal='maximize',
            n_mc_points=10 # Reduce MC points for speed in test
        )
        assert isinstance(candidates, pd.DataFrame)
        assert len(candidates) == 2

    def test_categorical_handling(self, trained_session_botorch):
        """Test that categorical variables are handled correctly in optimization."""
        # The catalyst dataset has categorical variables ('Catalyst')
        candidates = trained_session_botorch.suggest_next(
            strategy='qei',
            n_suggestions=1,
            goal='maximize'
        )
        assert 'Catalyst' in candidates.columns
        assert candidates.iloc[0]['Catalyst'] in ['High SAR', 'Low SAR']

    def test_invalid_strategy(self, trained_session_botorch):
        """Test invalid strategy raises error."""
        with pytest.raises(ValueError):
            trained_session_botorch.suggest_next(
                strategy='invalid_strategy',
                n_suggestions=1
            )

    def test_custom_beta_ucb(self, trained_session_botorch):
        """Test UCB with custom beta."""
        candidates = trained_session_botorch.suggest_next(
            strategy='ucb',
            n_suggestions=1,
            goal='maximize',
            beta=2.0
        )
        assert len(candidates) == 1

    def test_custom_mc_samples(self, trained_session_botorch):
        """Test qEI with custom mc_samples."""
        candidates = trained_session_botorch.suggest_next(
            strategy='qei',
            n_suggestions=1,
            goal='maximize',
            mc_samples=64
        )
        assert len(candidates) == 1
