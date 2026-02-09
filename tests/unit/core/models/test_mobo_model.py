"""
Test multi-objective model functionality for BoTorchModel and the sklearn guard.

Tests cover:
- BoTorchModel multi-objective training (_train_multi_objective)
- Multi-objective prediction (_predict_multi_objective)
- Multi-objective hyperparameter retrieval (get_hyperparameters)
- SklearnModel guard that rejects multi-objective data
"""

import pytest
import numpy as np
import pandas as pd
from alchemist_core.models.botorch_model import BoTorchModel
from alchemist_core.models.sklearn_model import SklearnModel
from alchemist_core.data.experiment_manager import ExperimentManager
from alchemist_core.data.search_space import SearchSpace


@pytest.fixture
def multi_objective_experiment_manager():
    """Create an ExperimentManager with 2-objective data."""
    space = SearchSpace()
    space.add_variable('x1', 'real', min=0.0, max=1.0)
    space.add_variable('x2', 'real', min=0.0, max=1.0)

    em = ExperimentManager(search_space=space, target_columns=['yield', 'selectivity'])

    # Create synthetic data with 10 points
    np.random.seed(42)
    data = pd.DataFrame({
        'x1': np.random.uniform(0, 1, 10),
        'x2': np.random.uniform(0, 1, 10),
        'yield': np.random.uniform(50, 100, 10),
        'selectivity': np.random.uniform(70, 95, 10),
    })
    em.df = data
    return em


@pytest.fixture
def single_objective_experiment_manager():
    """Create an ExperimentManager with single-objective data."""
    space = SearchSpace()
    space.add_variable('x1', 'real', min=0.0, max=1.0)
    space.add_variable('x2', 'real', min=0.0, max=1.0)

    em = ExperimentManager(search_space=space, target_columns=['Output'])

    np.random.seed(42)
    data = pd.DataFrame({
        'x1': np.random.uniform(0, 1, 10),
        'x2': np.random.uniform(0, 1, 10),
        'Output': np.random.uniform(0, 1, 10),
    })
    em.df = data
    return em


class TestBoTorchMultiObjectiveTraining:
    """Test BoTorchModel._train_multi_objective."""

    def test_creates_model_list_with_two_sub_models(self, multi_objective_experiment_manager):
        """Training with 2 objectives creates a ModelListGP with 2 sub-models."""
        from botorch.models.model_list_gp_regression import ModelListGP

        model = BoTorchModel(training_iter=10, random_state=42)
        model.train(multi_objective_experiment_manager, cache_cv=False, calibrate_uncertainty=False)

        assert isinstance(model.model, ModelListGP)
        assert len(model.model.models) == 2

    def test_n_objectives_set_correctly(self, multi_objective_experiment_manager):
        """n_objectives is set to 2 after multi-objective training."""
        model = BoTorchModel(training_iter=10, random_state=42)
        model.train(multi_objective_experiment_manager, cache_cv=False, calibrate_uncertainty=False)

        assert model.n_objectives == 2

    def test_objective_names_set_correctly(self, multi_objective_experiment_manager):
        """objective_names matches the target columns from the experiment manager."""
        model = BoTorchModel(training_iter=10, random_state=42)
        model.train(multi_objective_experiment_manager, cache_cv=False, calibrate_uncertainty=False)

        assert model.objective_names == ['yield', 'selectivity']

    def test_y_orig_shape(self, multi_objective_experiment_manager):
        """Y_orig has shape (n_samples, n_objectives) after multi-objective training."""
        model = BoTorchModel(training_iter=10, random_state=42)
        model.train(multi_objective_experiment_manager, cache_cv=False, calibrate_uncertainty=False)

        assert model.Y_orig.shape == (10, 2)

    def test_original_feature_names(self, multi_objective_experiment_manager):
        """original_feature_names are set to the input feature column names."""
        model = BoTorchModel(training_iter=10, random_state=42)
        model.train(multi_objective_experiment_manager, cache_cv=False, calibrate_uncertainty=False)

        assert model.original_feature_names == ['x1', 'x2']

    def test_is_trained_after_training(self, multi_objective_experiment_manager):
        """is_trained is True after successful multi-objective training."""
        model = BoTorchModel(training_iter=10, random_state=42)
        model.train(multi_objective_experiment_manager, cache_cv=False, calibrate_uncertainty=False)

        assert model.is_trained is True


class TestBoTorchMultiObjectivePredict:
    """Test BoTorchModel._predict_multi_objective."""

    @pytest.fixture(autouse=True)
    def _train_model(self, multi_objective_experiment_manager):
        """Train a multi-objective model for use in all predict tests."""
        self.model = BoTorchModel(training_iter=10, random_state=42)
        self.model.train(multi_objective_experiment_manager, cache_cv=False, calibrate_uncertainty=False)
        self.em = multi_objective_experiment_manager

    def test_predict_returns_dict_keyed_by_objective_names(self):
        """Predict returns a dict keyed by objective names."""
        test_X = pd.DataFrame({'x1': [0.5, 0.6], 'x2': [0.3, 0.4]})
        results = self.model.predict(test_X)

        assert isinstance(results, dict)
        assert set(results.keys()) == {'yield', 'selectivity'}

    def test_predict_values_are_mean_std_tuples(self):
        """Each value in the prediction dict is a tuple of (mean, std) arrays."""
        test_X = pd.DataFrame({'x1': [0.5, 0.6], 'x2': [0.3, 0.4]})
        results = self.model.predict(test_X)

        for obj_name in ['yield', 'selectivity']:
            value = results[obj_name]
            assert isinstance(value, tuple), f"Value for '{obj_name}' should be a tuple"
            assert len(value) == 2, f"Tuple for '{obj_name}' should have 2 elements (mean, std)"
            mean, std = value
            assert isinstance(mean, np.ndarray)
            assert isinstance(std, np.ndarray)

    def test_predict_array_shapes(self):
        """Mean and std arrays have the correct shape matching the number of test points."""
        test_X = pd.DataFrame({'x1': [0.5, 0.6, 0.7], 'x2': [0.3, 0.4, 0.5]})
        results = self.model.predict(test_X)

        for obj_name in ['yield', 'selectivity']:
            mean, std = results[obj_name]
            assert mean.shape == (3,), f"Mean shape for '{obj_name}' should be (3,)"
            assert std.shape == (3,), f"Std shape for '{obj_name}' should be (3,)"


class TestBoTorchMultiObjectiveHyperparameters:
    """Test BoTorchModel.get_hyperparameters for multi-objective models."""

    @pytest.fixture(autouse=True)
    def _train_model(self, multi_objective_experiment_manager):
        """Train a multi-objective model for use in all hyperparameter tests."""
        self.model = BoTorchModel(training_iter=10, random_state=42)
        self.model.train(multi_objective_experiment_manager, cache_cv=False, calibrate_uncertainty=False)

    def test_hyperparameters_has_n_objectives(self):
        """Hyperparameters dict contains 'n_objectives' key."""
        params = self.model.get_hyperparameters()

        assert 'n_objectives' in params
        assert params['n_objectives'] == 2

    def test_hyperparameters_has_models_key(self):
        """Hyperparameters dict contains 'models' key with per-objective params."""
        params = self.model.get_hyperparameters()

        assert 'models' in params
        assert isinstance(params['models'], dict)
        assert 'yield' in params['models']
        assert 'selectivity' in params['models']

    def test_hyperparameters_has_kernel_type(self):
        """Hyperparameters dict contains 'kernel_type' key."""
        params = self.model.get_hyperparameters()

        assert 'kernel_type' in params


class TestSklearnMultiObjectiveGuard:
    """Test that SklearnModel raises an error when given multi-objective data."""

    def test_sklearn_raises_on_multi_objective(self, multi_objective_experiment_manager):
        """SklearnModel.train() raises ValueError when target_columns has > 1 target."""
        model = SklearnModel(kernel_options={"kernel_type": "RBF"})

        with pytest.raises(ValueError, match="botorch"):
            model.train(multi_objective_experiment_manager, cache_cv=False, calibrate_uncertainty=False)

    def test_sklearn_error_message_mentions_botorch(self, multi_objective_experiment_manager):
        """The error message from sklearn guard mentions 'botorch' as the alternative."""
        model = SklearnModel(kernel_options={"kernel_type": "RBF"})

        with pytest.raises(ValueError) as exc_info:
            model.train(multi_objective_experiment_manager, cache_cv=False, calibrate_uncertainty=False)

        error_msg = str(exc_info.value).lower()
        assert 'botorch' in error_msg

    def test_sklearn_works_with_single_objective(self, single_objective_experiment_manager):
        """SklearnModel trains successfully with a single target column."""
        model = SklearnModel(kernel_options={"kernel_type": "RBF"})
        model.train(single_objective_experiment_manager, cache_cv=False, calibrate_uncertainty=False)

        assert model.is_trained is True
