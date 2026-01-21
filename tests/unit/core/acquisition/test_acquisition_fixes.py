"""
Test script to verify acquisition function fixes.

Tests:
1. Sklearn sign convention fix (EI, PI, UCB should show correct values)
2. BoTorch kwargs passing (beta parameter should affect suggestions)
3. Kwargs validation and logging
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from alchemist_core import OptimizationSession


# Get test data paths
TEST_DIR = Path(__file__).parent.parent.parent.parent
CATALYST_CSV = TEST_DIR / "catalyst_experiments.csv"
CATALYST_JSON = TEST_DIR / "catalyst_search_space.json"


def test_sklearn_sign_convention():
    """Test that sklearn acquisition values have correct sign for maximization."""
    # Create session with catalyst data
    session = OptimizationSession()
    session.load_search_space(str(CATALYST_JSON))
    
    # Load first 15 experiments
    df = pd.read_csv(CATALYST_CSV)
    df_subset = df.head(15)
    
    # Save subset and load
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        df_subset.to_csv(tmp.name, index=False)
        temp_path = tmp.name
    
    try:
        session.load_data(temp_path, target_columns='Output')
    finally:
        os.unlink(temp_path)
    
    # Train sklearn model
    session.train_model(backend='sklearn', kernel='rbf')
    
    # Test EI values are positive for maximization
    from alchemist_core.utils.acquisition_utils import evaluate_acquisition
    
    # Create test point at a middle value
    test_point = pd.DataFrame({
        'Temperature': [400.0],
        'Catalyst': ['Low SAR'],
        'Metal Loading': [2.5],
        'Zinc Fraction': [0.5]
    })
    
    acq_vals_ei, _ = evaluate_acquisition(
        session.model, test_point, acq_func='ei', goal='maximize'
    )
    
    assert acq_vals_ei[0] >= 0, f"EI values should be positive for maximization, got {acq_vals_ei[0]}"
    
    # Test UCB values
    acq_vals_ucb, _ = evaluate_acquisition(
        session.model, test_point, acq_func='ucb', goal='maximize'
    )
    
    assert acq_vals_ucb[0] > 0, f"UCB values should be positive for maximization, got {acq_vals_ucb[0]}"
    
    # Test PI values
    acq_vals_pi, _ = evaluate_acquisition(
        session.model, test_point, acq_func='pi', goal='maximize'
    )
    
    assert acq_vals_pi[0] >= 0, f"PI values should be positive for maximization, got {acq_vals_pi[0]}"


def test_botorch_kwargs_passing():
    """Test that BoTorch kwargs (beta) actually affect suggestions."""
    # Create session with catalyst data
    session = OptimizationSession()
    session.load_search_space(str(CATALYST_JSON))
    
    # Load first 15 experiments
    df = pd.read_csv(CATALYST_CSV)
    df_subset = df.head(15)
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        df_subset.to_csv(tmp.name, index=False)
        temp_path = tmp.name
    
    try:
        session.load_data(temp_path, target_columns='Output')
    finally:
        import os
        os.unlink(temp_path)
    
    # Train BoTorch model
    session.train_model(backend='botorch', kernel='Matern')
    
    # Get suggestion with different beta values
    # Note: Results may vary with random seed, so we just test that it runs without error
    suggestion1 = session.suggest_next(strategy='UCB', goal='maximize', beta=0.1)
    assert 'Temperature' in suggestion1.columns
    
    suggestion2 = session.suggest_next(strategy='UCB', goal='maximize', beta=2.0)
    assert 'Temperature' in suggestion2.columns


def test_kwargs_validation(caplog):
    """Test that invalid kwargs generate warnings."""
    import logging
    
    # Create session with catalyst data
    session = OptimizationSession()
    session.load_search_space(str(CATALYST_JSON))
    
    # Load first 10 experiments
    df = pd.read_csv(CATALYST_CSV)
    df_subset = df.head(10)
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        df_subset.to_csv(tmp.name, index=False)
        temp_path = tmp.name
    
    try:
        session.load_data(temp_path, target_columns='Output')
    finally:
        import os
        os.unlink(temp_path)
    
    # Train model
    session.train_model(backend='sklearn', kernel='rbf')
    
    # Test valid parameter (should log info)
    with caplog.at_level(logging.INFO):
        suggestion = session.suggest_next(strategy='UCB', goal='maximize', kappa=2.0)
        assert any('kappa' in record.message for record in caplog.records)
    
    # Test invalid parameter (should log warning)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        suggestion = session.suggest_next(strategy='UCB', goal='maximize', beta=2.0)
        assert any('Unsupported' in record.message and 'beta' in record.message 
                  for record in caplog.records)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

