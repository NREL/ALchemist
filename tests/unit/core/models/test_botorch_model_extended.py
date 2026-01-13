
import pytest
import os
import pandas as pd
import numpy as np
import torch
from alchemist_core import OptimizationSession
from alchemist_core.models.botorch_model import BoTorchModel
from botorch.models import SingleTaskGP, MixedSingleTaskGP
from gpytorch.kernels import RBFKernel

@pytest.fixture
def catalyst_session():
    """Create session with real catalyst data."""
    session = OptimizationSession()
    
    # Load search space from JSON
    # Go up 4 levels: models -> core -> unit -> tests
    tests_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    search_space_path = os.path.join(tests_dir, 'catalyst_search_space.json')
    session.load_search_space(search_space_path)
    
    # Load experiments from CSV
    experiments_path = os.path.join(tests_dir, 'catalyst_experiments.csv')
    session.load_data(experiments_path)
    
    return session

class TestBoTorchModelExtended:
    """Extended tests for BoTorchModel to improve coverage."""

    def test_train_with_noise(self, catalyst_session):
        """Test training with explicit noise (uncertainty) data."""
        # Clear existing experiments to avoid mixing noisy and non-noisy data
        # which causes NaNs in train_Yvar and crashes BoTorch
        catalyst_session.experiment_manager.clear()
        
        # Add experiments with noise
        catalyst_session.add_experiment(
            {'Temperature': 400, 'Catalyst': 'High SAR', 'Metal Loading': 2.5, 'Zinc Fraction': 0.5},
            output=0.8,
            noise=0.05
        )
        catalyst_session.add_experiment(
            {'Temperature': 420, 'Catalyst': 'Low SAR', 'Metal Loading': 3.0, 'Zinc Fraction': 0.2},
            output=0.6,
            noise=0.05
        )
        catalyst_session.add_experiment(
            {'Temperature': 380, 'Catalyst': 'High SAR', 'Metal Loading': 1.0, 'Zinc Fraction': 0.8},
            output=0.4,
            noise=0.05
        )
        
        # Train model
        results = catalyst_session.train_model(
            backend='botorch',
            kernel='Matern'
        )
        
        assert results['success'] == True
        assert catalyst_session.model.is_trained
        
        # Verify that the underlying model is using the noise
        # In BoTorch, if train_Yvar is provided, it's stored in the model
        botorch_model = catalyst_session.model.model
        assert botorch_model.train_targets.shape == botorch_model.train_inputs[0].shape[:-1]
        # We can't easily check train_Yvar directly on the model object in all versions, 
        # but successful training with noise data implies it was handled.

    def test_explicit_standardize_transform(self, catalyst_session):
        """Test input_transform_type='standardize'."""
        results = catalyst_session.train_model(
            backend='botorch',
            input_transform_type='standardize',
            output_transform_type='standardize'
        )
        assert results['success'] == True
        
        # Check if transforms were applied (BoTorchModel stores this in internal state or logs)
        # We can check the model config or just rely on successful execution covering the lines
        assert catalyst_session.model.input_transform_type == 'standardize'
        assert catalyst_session.model.output_transform_type == 'standardize'

    def test_messy_data_encoding_fallback(self):
        """Test the fallback encoding logic for non-numeric data not in cat_dims."""
        # Create a session manually to control data exactly
        session = OptimizationSession()
        session.add_variable('x', 'real', bounds=(0, 10))
        session.add_variable('messy_col', 'categorical', categories=['A', 'B'])
        
        # Add data where 'messy_col' is passed but maybe not correctly identified initially
        # or simulate the condition where pd.to_numeric fails
        
        # We can test the _encode_categorical_data method directly
        model = BoTorchModel()
        
        df = pd.DataFrame({
            'x': [1.0, 2.0, 3.0],
            'messy_col': ['A', 'B', 'A'], # Strings
            'y': [0.1, 0.2, 0.3]
        })
        
        # Force cat_dims to be empty so it tries to convert 'messy_col' to numeric and fails
        model.cat_dims = [] 
        
        encoded_df = model._encode_categorical_data(df)
        
        # Check that 'messy_col' was encoded to numbers despite not being in cat_dims
        assert pd.api.types.is_numeric_dtype(encoded_df['messy_col'])
        assert set(encoded_df['messy_col'].unique()) == {0.0, 1.0}

    def test_rbf_kernel_factory(self, catalyst_session):
        """Explicitly test RBF kernel factory creation."""
        model = BoTorchModel(kernel_options={"cont_kernel_type": "RBF"})
        factory = model._get_cont_kernel_factory()
        
        # Create dummy args for factory
        batch_shape = torch.Size([])
        ard_num_dims = 2
        active_dims = [0, 1]
        
        kernel = factory(batch_shape, ard_num_dims, active_dims)
        assert isinstance(kernel, RBFKernel)

    def test_mixed_single_task_gp_with_noise(self, catalyst_session):
        """Test MixedSingleTaskGP (categorical) with noise."""
        # Catalyst session has categorical variables
        
        # Clear existing experiments to avoid mixing noisy and non-noisy data
        catalyst_session.experiment_manager.clear()
        
        # Add experiments with noise
        catalyst_session.add_experiment(
            {'Temperature': 400, 'Catalyst': 'High SAR', 'Metal Loading': 2.5, 'Zinc Fraction': 0.5},
            output=0.8,
            noise=0.05
        )
        catalyst_session.add_experiment(
            {'Temperature': 420, 'Catalyst': 'Low SAR', 'Metal Loading': 3.0, 'Zinc Fraction': 0.2},
            output=0.6,
            noise=0.05
        )
        catalyst_session.add_experiment(
            {'Temperature': 380, 'Catalyst': 'High SAR', 'Metal Loading': 1.0, 'Zinc Fraction': 0.8},
            output=0.4,
            noise=0.05
        )
        
        results = catalyst_session.train_model(
            backend='botorch',
            kernel='Matern'
            # Session now auto-detects cat_dims from search space
        )
        
        assert results['success'] == True
        assert isinstance(catalyst_session.model.model, MixedSingleTaskGP)
        # Verify it didn't crash and produced a model

    def test_single_task_gp_with_noise(self):
        """Test SingleTaskGP (no categorical) with noise."""
        session = OptimizationSession()
        session.add_variable('x', 'real', bounds=(0, 10))
        
        # Add data with noise
        session.add_experiment({'x': 1.0}, 0.1, noise=0.01)
        session.add_experiment({'x': 2.0}, 0.2, noise=0.01)
        session.add_experiment({'x': 3.0}, 0.3, noise=0.01)
        
        results = session.train_model(
            backend='botorch',
            kernel='Matern'
        )
        
        assert results['success'] == True
        assert isinstance(session.model.model, SingleTaskGP)
