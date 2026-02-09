"""
Tests for multi-objective Bayesian optimization (MOBO) acquisition functionality.

Covers:
- BoTorchAcquisition MOBO validation (qEHVI, qNEHVI in VALID_ACQ_FUNCS)
- Constructor acceptance of MOBO-specific parameters
- _compute_default_ref_point correctness
- _create_mobo_acquisition_function for qEHVI and qNEHVI
- SkoptAcquisition guard against multi-objective usage
"""

import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from alchemist_core.acquisition.botorch_acquisition import BoTorchAcquisition
from alchemist_core.acquisition.skopt_acquisition import SkoptAcquisition
from alchemist_core.data.search_space import SearchSpace
from alchemist_core.models.botorch_model import BoTorchModel
from alchemist_core.data.experiment_manager import ExperimentManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def search_space():
    """Create a simple 2-variable continuous search space."""
    space = SearchSpace()
    space.add_variable('x1', 'real', min=0.0, max=1.0)
    space.add_variable('x2', 'real', min=0.0, max=1.0)
    return space


@pytest.fixture
def trained_mobo_model(search_space):
    """Train a real BoTorchModel on 2-objective data."""
    em = ExperimentManager(
        search_space=search_space,
        target_columns=['yield', 'selectivity'],
    )
    np.random.seed(42)
    data = pd.DataFrame({
        'x1': np.random.uniform(0, 1, 10),
        'x2': np.random.uniform(0, 1, 10),
        'yield': np.random.uniform(50, 100, 10),
        'selectivity': np.random.uniform(70, 95, 10),
    })
    em.df = data
    model = BoTorchModel(training_iter=10, random_state=42)
    model.train(em, cache_cv=False, calibrate_uncertainty=False)
    return model


# ---------------------------------------------------------------------------
# BoTorchAcquisition MOBO validation
# ---------------------------------------------------------------------------

class TestBoTorchMOBOValidation:
    """Tests for MOBO acquisition function validation and construction."""

    def test_qehvi_in_valid_acq_funcs(self):
        """qehvi should be listed as a valid acquisition function."""
        assert 'qehvi' in BoTorchAcquisition.VALID_ACQ_FUNCS

    def test_qnehvi_in_valid_acq_funcs(self):
        """qnehvi should be listed as a valid acquisition function."""
        assert 'qnehvi' in BoTorchAcquisition.VALID_ACQ_FUNCS

    def test_constructor_accepts_mobo_params(self, search_space):
        """Constructor should accept ref_point, directions, objective_names,
        and outcome_constraints without error."""
        acq = BoTorchAcquisition(
            search_space=search_space,
            acq_func='qehvi',
            ref_point=[0.0, 0.0],
            directions=['maximize', 'minimize'],
            objective_names=['yield', 'selectivity'],
            outcome_constraints=[lambda Y: Y[..., 0] - 80.0],
        )
        assert acq.ref_point == [0.0, 0.0]
        assert acq.directions == ['maximize', 'minimize']
        assert acq.objective_names == ['yield', 'selectivity']
        assert acq.outcome_constraints is not None
        assert len(acq.outcome_constraints) == 1

    def test_invalid_acq_func_raises(self, search_space):
        """An unrecognised acquisition function name should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid acquisition function"):
            BoTorchAcquisition(
                search_space=search_space,
                acq_func='totally_bogus',
            )


# ---------------------------------------------------------------------------
# _compute_default_ref_point
# ---------------------------------------------------------------------------

class TestComputeDefaultRefPoint:
    """Tests for the _compute_default_ref_point helper."""

    def test_ref_point_length(self, search_space):
        """Returned ref_point should have one entry per objective."""
        acq = BoTorchAcquisition(search_space=search_space, acq_func='qehvi')
        Y_max = torch.tensor([[1.0, 2.0], [3.0, 1.0], [2.0, 3.0]])
        ref_point = acq._compute_default_ref_point(Y_max)
        assert len(ref_point) == 2

    def test_ref_point_below_min_each_column(self, search_space):
        """Each ref_point element should be strictly below the column minimum."""
        acq = BoTorchAcquisition(search_space=search_space, acq_func='qehvi')
        Y_max = torch.tensor([[1.0, 2.0], [3.0, 1.0], [2.0, 3.0]])
        ref_point = acq._compute_default_ref_point(Y_max)
        # Min of col 0 is 1.0, min of col 1 is 1.0
        assert ref_point[0] < 1.0
        assert ref_point[1] < 1.0

    def test_ref_point_with_negative_values(self, search_space):
        """Should work correctly when Y_max contains negative values."""
        acq = BoTorchAcquisition(search_space=search_space, acq_func='qehvi')
        Y_max = torch.tensor([[-5.0, -2.0], [-3.0, -8.0]])
        ref_point = acq._compute_default_ref_point(Y_max)
        assert len(ref_point) == 2
        # Min of col 0 is -5.0, ref should be below that
        assert ref_point[0] < -5.0
        # Min of col 1 is -8.0, ref should be below that
        assert ref_point[1] < -8.0

    def test_ref_point_three_objectives(self, search_space):
        """Should handle arbitrary number of objectives."""
        acq = BoTorchAcquisition(search_space=search_space, acq_func='qehvi')
        Y_max = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        ref_point = acq._compute_default_ref_point(Y_max)
        assert len(ref_point) == 3
        # Mins are 1.0, 2.0, 3.0 respectively
        assert ref_point[0] < 1.0
        assert ref_point[1] < 2.0
        assert ref_point[2] < 3.0


# ---------------------------------------------------------------------------
# _create_mobo_acquisition_function (qEHVI)
# ---------------------------------------------------------------------------

class TestCreateMOBOAcquisitionQEHVI:
    """Tests for qEHVI acquisition function creation."""

    def test_qehvi_creates_acq_function(self, search_space, trained_mobo_model):
        """Passing a trained multi-objective model with qehvi should produce
        a non-None acq_function."""
        acq = BoTorchAcquisition(
            search_space=search_space,
            model=trained_mobo_model,
            acq_func='qehvi',
            directions=['maximize', 'maximize'],
        )
        assert acq.acq_function is not None

    def test_qehvi_model_has_two_objective_columns(self, trained_mobo_model):
        """The trained model's Y_orig should have 2 columns (one per objective)."""
        assert trained_mobo_model.Y_orig.shape[1] == 2

    def test_qehvi_with_explicit_ref_point(self, search_space, trained_mobo_model):
        """qEHVI should accept an explicit ref_point and still create the
        acquisition function."""
        acq = BoTorchAcquisition(
            search_space=search_space,
            model=trained_mobo_model,
            acq_func='qehvi',
            ref_point=[40.0, 60.0],
            directions=['maximize', 'maximize'],
        )
        assert acq.acq_function is not None


# ---------------------------------------------------------------------------
# _create_mobo_acquisition_function (qNEHVI)
# ---------------------------------------------------------------------------

class TestCreateMOBOAcquisitionQNEHVI:
    """Tests for qNEHVI acquisition function creation."""

    def test_qnehvi_creates_acq_function(self, search_space, trained_mobo_model):
        """Passing a trained multi-objective model with qnehvi should produce
        a non-None acq_function."""
        acq = BoTorchAcquisition(
            search_space=search_space,
            model=trained_mobo_model,
            acq_func='qnehvi',
            directions=['maximize', 'maximize'],
        )
        assert acq.acq_function is not None

    def test_qnehvi_with_outcome_constraints(self, search_space, trained_mobo_model):
        """Outcome constraints should be passed through to the qNEHVI
        acquisition function without error."""
        # Constraint: first objective must be >= 60 (expressed as Y[...,0] - 60 >= 0)
        constraints = [lambda Y: Y[..., 0] - 60.0]
        acq = BoTorchAcquisition(
            search_space=search_space,
            model=trained_mobo_model,
            acq_func='qnehvi',
            directions=['maximize', 'maximize'],
            outcome_constraints=constraints,
        )
        assert acq.acq_function is not None

    def test_qnehvi_with_minimize_direction(self, search_space, trained_mobo_model):
        """qNEHVI should work when one objective is set to minimize."""
        acq = BoTorchAcquisition(
            search_space=search_space,
            model=trained_mobo_model,
            acq_func='qnehvi',
            directions=['maximize', 'minimize'],
        )
        assert acq.acq_function is not None


# ---------------------------------------------------------------------------
# SkoptAcquisition multi-objective guard
# ---------------------------------------------------------------------------

class TestSkoptMultiObjectiveGuard:
    """Tests for the SkoptAcquisition guard against multi-objective models."""

    def test_select_next_raises_for_multi_objective(self):
        """select_next() should raise ValueError when the model wrapper
        reports more than one objective."""
        space = SearchSpace()
        space.add_variable('x1', 'real', min=0.0, max=1.0)

        acq = SkoptAcquisition(search_space=space, acq_func='ei')

        # Attach a mock model wrapper that reports 2 objectives
        acq._model_wrapper = MagicMock()
        acq._model_wrapper.n_objectives = 2

        with pytest.raises(ValueError, match="Multi-objective"):
            acq.select_next()

    def test_select_next_works_for_single_objective(self):
        """select_next() should NOT raise when n_objectives is 1."""
        space = SearchSpace()
        space.add_variable('x1', 'real', min=0.0, max=1.0)

        acq = SkoptAcquisition(search_space=space, acq_func='ei')

        # Attach a mock model wrapper that reports 1 objective
        acq._model_wrapper = MagicMock()
        acq._model_wrapper.n_objectives = 1

        # Should not raise -- the optimizer.ask() call is the normal path
        result = acq.select_next()
        assert result is not None
