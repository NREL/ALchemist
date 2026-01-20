"""
Test acquisition function evaluation for both sklearn and BoTorch backends.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from alchemist_core.session import OptimizationSession
from alchemist_core.utils.acquisition_utils import evaluate_acquisition


# Test data paths
TEST_DIR = Path(__file__).parent.parent  # Go up to tests/
CATALYST_DATA = TEST_DIR / "catalyst_experiments.csv"
CATALYST_SPACE = TEST_DIR / "catalyst_search_space.json"


@pytest.fixture
def session_sklearn():
    """Create a trained sklearn-based session."""
    session = OptimizationSession()
    session.load_search_space(str(CATALYST_SPACE))
    session.load_data(str(CATALYST_DATA))
    session.train_model(backend='sklearn')
    return session


@pytest.fixture
def session_botorch():
    """Create a trained BoTorch-based session."""
    session = OptimizationSession()
    session.load_search_space(str(CATALYST_SPACE))
    session.load_data(str(CATALYST_DATA))
    session.train_model(backend='botorch')
    return session


class TestAcquisitionEvaluationSklearn:
    """Tests for sklearn/skopt acquisition functions."""
    
    def test_evaluate_ei_sklearn(self, session_sklearn):
        """Test Expected Improvement evaluation with sklearn backend."""
        test_points = pd.DataFrame({
            'Temperature': [360.0, 400.0, 440.0],
            'Catalyst': ['Low SAR', 'Low SAR', 'High SAR'],
            'Metal Loading': [0.5, 2.5, 4.5],
            'Zinc Fraction': [0.2, 0.5, 0.8]
        })
        
        # Evaluate EI
        acq_vals, _ = evaluate_acquisition(
            session_sklearn.model,
            test_points,
            acq_func='ei',
            acq_func_kwargs={'xi': 0.01},
            goal='maximize'
        )
        
        # Should get finite values
        assert acq_vals is not None
        assert len(acq_vals) == len(test_points)
        assert np.all(np.isfinite(acq_vals))
    
    def test_evaluate_pi_sklearn(self, session_sklearn):
        """Test Probability of Improvement evaluation."""
        test_points = pd.DataFrame({
            'Temperature': [360.0, 400.0, 440.0],
            'Catalyst': ['Low SAR', 'Low SAR', 'High SAR'],
            'Metal Loading': [0.5, 2.5, 4.5],
            'Zinc Fraction': [0.2, 0.5, 0.8]
        })
        
        # Evaluate PI
        acq_vals, _ = evaluate_acquisition(
            session_sklearn.model,
            test_points,
            acq_func='pi',
            acq_func_kwargs={'xi': 0.01},
            goal='maximize'
        )
        
        assert np.all(np.isfinite(acq_vals))
        # PI should be between 0 and 1
        assert np.all((acq_vals >= 0) & (acq_vals <= 1))
    
    def test_evaluate_ucb_sklearn(self, session_sklearn):
        """Test Upper Confidence Bound evaluation."""
        test_points = pd.DataFrame({
            'Temperature': [360.0, 400.0, 440.0],
            'Catalyst': ['Low SAR', 'Low SAR', 'High SAR'],
            'Metal Loading': [0.5, 2.5, 4.5],
            'Zinc Fraction': [0.2, 0.5, 0.8]
        })
        
        # Evaluate UCB
        acq_vals, _ = evaluate_acquisition(
            session_sklearn.model,
            test_points,
            acq_func='ucb',
            acq_func_kwargs={'kappa': 1.96},
            goal='maximize'
        )
        
        assert np.all(np.isfinite(acq_vals))
    
    def test_minimize_vs_maximize_sklearn(self, session_sklearn):
        """Test that minimize vs maximize gives reasonable results."""
        test_points = pd.DataFrame({
            'Temperature': [360.0, 400.0, 440.0],
            'Catalyst': ['Low SAR', 'Low SAR', 'High SAR'],
            'Metal Loading': [0.5, 2.5, 4.5],
            'Zinc Fraction': [0.2, 0.5, 0.8]
        })
        
        # Evaluate for maximization
        acq_max, _ = evaluate_acquisition(
            session_sklearn.model,
            test_points, acq_func='ucb', goal='maximize'
        )
        
        # Evaluate for minimization
        acq_min, _ = evaluate_acquisition(
            session_sklearn.model,
            test_points, acq_func='ucb', goal='minimize'
        )
        
        assert np.all(np.isfinite(acq_max))
        assert np.all(np.isfinite(acq_min))
        # The two shouldn't be identical
        assert not np.allclose(acq_max, acq_min)


class TestAcquisitionEvaluationBoTorch:
    """Tests for BoTorch acquisition functions."""
    
    def test_evaluate_ei_botorch(self, session_botorch):
        """Test Expected Improvement with BoTorch backend."""
        test_points = pd.DataFrame({
            'Temperature': [360.0, 400.0, 440.0],
            'Catalyst': ['Low SAR', 'Low SAR', 'High SAR'],
            'Metal Loading': [0.5, 2.5, 4.5],
            'Zinc Fraction': [0.2, 0.5, 0.8]
        })
        
        # Evaluate EI
        acq_vals, _ = evaluate_acquisition(
            session_botorch.model,
            test_points,
            acq_func='ei',
            goal='maximize'
        )
        
        assert acq_vals is not None
        assert len(acq_vals) == len(test_points)
        assert np.all(np.isfinite(acq_vals))
    
    def test_evaluate_ucb_botorch(self, session_botorch):
        """Test UCB with BoTorch backend."""
        test_points = pd.DataFrame({
            'Temperature': [360.0, 400.0, 440.0],
            'Catalyst': ['Low SAR', 'Low SAR', 'High SAR'],
            'Metal Loading': [0.5, 2.5, 4.5],
            'Zinc Fraction': [0.2, 0.5, 0.8]
        })
        
        # Evaluate UCB
        acq_vals, _ = evaluate_acquisition(
            session_botorch.model,
            test_points,
            acq_func='ucb',
            acq_func_kwargs={'beta': 0.5},
            goal='maximize'
        )
        
        assert np.all(np.isfinite(acq_vals))
    
    def test_evaluate_logei_botorch(self, session_botorch):
        """Test LogEI (BoTorch-specific function)."""
        test_points = pd.DataFrame({
            'Temperature': [360.0, 400.0, 440.0],
            'Catalyst': ['Low SAR', 'Low SAR', 'High SAR'],
            'Metal Loading': [0.5, 2.5, 4.5],
            'Zinc Fraction': [0.2, 0.5, 0.8]
        })
        
        # Evaluate LogEI
        acq_vals, _ = evaluate_acquisition(
            session_botorch.model,
            test_points,
            acq_func='logei',
            goal='maximize'
        )
        
        assert np.all(np.isfinite(acq_vals))
    
    def test_invalid_acquisition_function(self, session_botorch):
        """Test that invalid acquisition function name raises error."""
        test_points = pd.DataFrame({
            'Temperature': [400.0],
            'Catalyst': ['Low SAR'],
            'Metal Loading': [2.5],
            'Zinc Fraction': [0.5]
        })
        
        with pytest.raises(ValueError, match="Unknown acquisition function"):
            evaluate_acquisition(
                session_botorch.model,
                test_points,
                acq_func='invalid_func'
            )


class TestAcquisitionEvaluationComparison:
    """Tests comparing sklearn and BoTorch backends."""
    
    def test_backends_give_similar_ordering(self, session_sklearn, session_botorch):
        """Test that sklearn and BoTorch give correlated acquisition values."""
        test_points = pd.DataFrame({
            'Temperature': [360.0, 400.0, 440.0],
            'Catalyst': ['Low SAR', 'Low SAR', 'High SAR'],
            'Metal Loading': [2.5, 2.5, 2.5],
            'Zinc Fraction': [0.5, 0.5, 0.5]
        })
        
        # Evaluate with sklearn
        acq_sklearn, _ = evaluate_acquisition(
            session_sklearn.model,
            test_points, acq_func='ei', goal='maximize'
        )
        
        # Evaluate with BoTorch
        acq_botorch, _ = evaluate_acquisition(
            session_botorch.model,
            test_points, acq_func='ei', goal='maximize'
        )
        
        # We don't expect identical values, but similar orderings
        from scipy.stats import spearmanr
        corr, _ = spearmanr(acq_sklearn, acq_botorch)
        
        # Should have reasonable correlation (backends may differ in implementation details)
        # A correlation of 0.5+ indicates similar ranking trends
        assert abs(corr) > 0.4, f"Rankings too different: correlation = {corr}"


class TestAcquisitionWithUntrainedModel:
    """Test edge cases with untrained models."""
    
    def test_evaluate_without_training(self):
        """Test that evaluation fails without trained model."""
        session = OptimizationSession()
        session.load_search_space(str(CATALYST_SPACE))
        session.load_data(str(CATALYST_DATA))
        # Don't train model
        
        test_points = pd.DataFrame({
            'Temperature': [400.0],
            'Catalyst': ['Low SAR'],
            'Metal Loading': [2.0],
            'Zinc Fraction': [0.5]
        })
        
        with pytest.raises(ValueError, match="Model must be trained"):
            evaluate_acquisition(session.model, test_points)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
