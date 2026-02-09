"""
Tests for multi-objective (MOBO) visualization adaptations.

Covers:
A. Per-objective target_column parameter (parity, contour, slice, voxel, qq, calibration,
   uncertainty_contour, uncertainty_voxel, metrics)
B. Hypervolume convergence plot (regret replacement)
C. Suggested next → Pareto overlay
D. Acquisition surfaces with MOBO
E. Blocked methods (probability_of_improvement)
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import MagicMock, patch, PropertyMock

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from alchemist_core import OptimizationSession
from alchemist_core.visualization.plots import create_hypervolume_convergence_plot


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mobo_session():
    """MOBO session with 2-objective data (10 experiments)."""
    session = OptimizationSession()
    session.add_variable('x1', 'real', bounds=(0.0, 1.0))
    session.add_variable('x2', 'real', bounds=(0.0, 1.0))

    np.random.seed(42)
    data = pd.DataFrame({
        'x1': np.random.uniform(0, 1, 10),
        'x2': np.random.uniform(0, 1, 10),
        'yield': np.random.uniform(50, 100, 10),
        'selectivity': np.random.uniform(70, 95, 10),
    })

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
        data.to_csv(f, index=False)
        filepath = f.name

    try:
        session.load_data(filepath, target_columns=['yield', 'selectivity'])
    finally:
        os.unlink(filepath)

    return session


@pytest.fixture
def single_session():
    """Single-objective session with 10 experiments."""
    session = OptimizationSession()
    session.add_variable('x1', 'real', bounds=(0.0, 1.0))
    session.add_variable('x2', 'real', bounds=(0.0, 1.0))

    np.random.seed(42)
    data = pd.DataFrame({
        'x1': np.random.uniform(0, 1, 10),
        'x2': np.random.uniform(0, 1, 10),
        'Output': np.random.uniform(50, 100, 10),
    })

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
        data.to_csv(f, index=False)
        filepath = f.name

    try:
        session.load_data(filepath, target_columns='Output')
    finally:
        os.unlink(filepath)

    return session


def _mock_mobo_model(session):
    """Install a mock model that returns dict predictions for MOBO."""
    model = MagicMock()
    model.is_trained = True
    model.original_feature_names = ['x1', 'x2']

    n_obj = session.n_objectives
    obj_names = session.objective_names

    # CV results per objective
    model.cv_cached_results = None
    model.cv_cached_results_multi = {}
    for obj in obj_names:
        n = 10
        model.cv_cached_results_multi[obj] = {
            'y_true': np.random.uniform(50, 100, n),
            'y_pred': np.random.uniform(50, 100, n),
            'y_std': np.random.uniform(0.1, 5.0, n),
        }

    def predict_side_effect(inputs, return_std=True):
        n = len(inputs)
        result = {}
        for obj in obj_names:
            result[obj] = (np.random.uniform(50, 100, n),
                           np.random.uniform(0.1, 5.0, n))
        return result

    model.predict.side_effect = predict_side_effect

    session.model = model
    session.model_backend = 'botorch'


def _mock_single_model(session):
    """Install a mock model that returns (mean, std) for single-objective."""
    model = MagicMock()
    model.is_trained = True
    model.original_feature_names = ['x1', 'x2']
    model.cv_cached_results = {
        'y_true': np.random.uniform(50, 100, 10),
        'y_pred': np.random.uniform(50, 100, 10),
        'y_std': np.random.uniform(0.1, 5.0, 10),
    }
    model.cv_cached_results_calibrated = None

    def predict_side_effect(inputs, return_std=True):
        n = len(inputs)
        return (np.random.uniform(50, 100, n),
                np.random.uniform(0.1, 5.0, n))

    model.predict.side_effect = predict_side_effect

    session.model = model
    session.model_backend = 'sklearn'


# ============================================================
# A. Per-objective target_column tests
# ============================================================

class TestResolveTargetColumn:
    """Tests for _resolve_target_column helper."""

    def test_single_objective_returns_first_target(self, single_session):
        result = single_session._resolve_target_column(None)
        assert result == 'Output'

    def test_single_objective_ignores_target_column(self, single_session):
        # Single-obj ignores target_column
        result = single_session._resolve_target_column('anything')
        assert result == 'Output'

    def test_mobo_none_raises(self, mobo_session):
        with pytest.raises(ValueError, match="target_column"):
            mobo_session._resolve_target_column(None)

    def test_mobo_specific_objective(self, mobo_session):
        result = mobo_session._resolve_target_column('yield')
        assert result == 'yield'

    def test_mobo_all_returns_list(self, mobo_session):
        result = mobo_session._resolve_target_column('all')
        assert isinstance(result, list)
        assert result == ['yield', 'selectivity']

    def test_mobo_unknown_raises(self, mobo_session):
        with pytest.raises(ValueError, match="Unknown objective"):
            mobo_session._resolve_target_column('nonexistent')


class TestParityMOBO:
    """Tests for plot_parity with MOBO."""

    def teardown_method(self):
        plt.close('all')

    def test_parity_single_objective_unchanged(self, single_session):
        _mock_single_model(single_session)
        fig = single_session.plot_parity()
        assert isinstance(fig, Figure)

    def test_parity_mobo_specific_objective(self, mobo_session):
        _mock_mobo_model(mobo_session)
        fig = mobo_session.plot_parity(target_column='yield')
        assert isinstance(fig, Figure)

    def test_parity_mobo_all(self, mobo_session):
        _mock_mobo_model(mobo_session)
        fig = mobo_session.plot_parity(target_column='all')
        assert isinstance(fig, Figure)
        # Should have 2 subplots
        assert len(fig.axes) == 2

    def test_parity_mobo_none_raises(self, mobo_session):
        _mock_mobo_model(mobo_session)
        with pytest.raises(ValueError, match="target_column"):
            mobo_session.plot_parity()


class TestSliceMOBO:
    """Tests for plot_slice with MOBO."""

    def teardown_method(self):
        plt.close('all')

    def test_slice_single_objective_unchanged(self, single_session):
        _mock_single_model(single_session)
        fig = single_session.plot_slice('x1')
        assert isinstance(fig, Figure)

    def test_slice_mobo_specific_objective(self, mobo_session):
        _mock_mobo_model(mobo_session)
        fig = mobo_session.plot_slice('x1', target_column='selectivity')
        assert isinstance(fig, Figure)

    def test_slice_mobo_all(self, mobo_session):
        _mock_mobo_model(mobo_session)
        fig = mobo_session.plot_slice('x1', target_column='all')
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2

    def test_slice_mobo_none_raises(self, mobo_session):
        _mock_mobo_model(mobo_session)
        with pytest.raises(ValueError, match="target_column"):
            mobo_session.plot_slice('x1')


class TestContourMOBO:
    """Tests for plot_contour with MOBO."""

    def teardown_method(self):
        plt.close('all')

    def test_contour_single_unchanged(self, single_session):
        _mock_single_model(single_session)
        fig = single_session.plot_contour('x1', 'x2', grid_resolution=5)
        assert isinstance(fig, Figure)

    def test_contour_mobo_specific(self, mobo_session):
        _mock_mobo_model(mobo_session)
        fig = mobo_session.plot_contour('x1', 'x2', grid_resolution=5, target_column='yield')
        assert isinstance(fig, Figure)

    def test_contour_mobo_all(self, mobo_session):
        _mock_mobo_model(mobo_session)
        fig = mobo_session.plot_contour('x1', 'x2', grid_resolution=5, target_column='all')
        assert isinstance(fig, Figure)
        # 2 subplots + 2 colorbars = 4 axes
        assert len(fig.axes) >= 2

    def test_contour_mobo_none_raises(self, mobo_session):
        _mock_mobo_model(mobo_session)
        with pytest.raises(ValueError, match="target_column"):
            mobo_session.plot_contour('x1', 'x2', grid_resolution=5)


class TestQQMOBO:
    """Tests for plot_qq with MOBO."""

    def teardown_method(self):
        plt.close('all')

    def test_qq_single_unchanged(self, single_session):
        _mock_single_model(single_session)
        fig = single_session.plot_qq()
        assert isinstance(fig, Figure)

    def test_qq_mobo_specific(self, mobo_session):
        _mock_mobo_model(mobo_session)
        fig = mobo_session.plot_qq(target_column='yield')
        assert isinstance(fig, Figure)

    def test_qq_mobo_all(self, mobo_session):
        _mock_mobo_model(mobo_session)
        fig = mobo_session.plot_qq(target_column='all')
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2

    def test_qq_mobo_none_raises(self, mobo_session):
        _mock_mobo_model(mobo_session)
        with pytest.raises(ValueError, match="target_column"):
            mobo_session.plot_qq()


class TestCalibrationMOBO:
    """Tests for plot_calibration with MOBO."""

    def teardown_method(self):
        plt.close('all')

    def test_calibration_single_unchanged(self, single_session):
        _mock_single_model(single_session)
        fig = single_session.plot_calibration()
        assert isinstance(fig, Figure)

    def test_calibration_mobo_specific(self, mobo_session):
        _mock_mobo_model(mobo_session)
        fig = mobo_session.plot_calibration(target_column='selectivity')
        assert isinstance(fig, Figure)

    def test_calibration_mobo_all(self, mobo_session):
        _mock_mobo_model(mobo_session)
        fig = mobo_session.plot_calibration(target_column='all')
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2

    def test_calibration_mobo_none_raises(self, mobo_session):
        _mock_mobo_model(mobo_session)
        with pytest.raises(ValueError, match="target_column"):
            mobo_session.plot_calibration()


class TestUncertaintyContourMOBO:
    """Tests for plot_uncertainty_contour with MOBO."""

    def teardown_method(self):
        plt.close('all')

    def test_unc_contour_single_unchanged(self, single_session):
        _mock_single_model(single_session)
        fig = single_session.plot_uncertainty_contour('x1', 'x2', grid_resolution=5)
        assert isinstance(fig, Figure)

    def test_unc_contour_mobo_specific(self, mobo_session):
        _mock_mobo_model(mobo_session)
        fig = mobo_session.plot_uncertainty_contour('x1', 'x2', grid_resolution=5,
                                                     target_column='yield')
        assert isinstance(fig, Figure)

    def test_unc_contour_mobo_all(self, mobo_session):
        _mock_mobo_model(mobo_session)
        fig = mobo_session.plot_uncertainty_contour('x1', 'x2', grid_resolution=5,
                                                     target_column='all')
        assert isinstance(fig, Figure)
        assert len(fig.axes) >= 2

    def test_unc_contour_mobo_none_raises(self, mobo_session):
        _mock_mobo_model(mobo_session)
        with pytest.raises(ValueError, match="target_column"):
            mobo_session.plot_uncertainty_contour('x1', 'x2', grid_resolution=5)


class TestMetricsMOBO:
    """Tests for plot_metrics blocking in MOBO."""

    def test_metrics_mobo_raises(self, mobo_session):
        _mock_mobo_model(mobo_session)
        with pytest.raises(ValueError, match="not yet supported for multi-objective"):
            mobo_session.plot_metrics()


# ============================================================
# B. Hypervolume convergence plot
# ============================================================

class TestHypervolumeConvergencePlot:
    """Tests for create_hypervolume_convergence_plot pure function."""

    def teardown_method(self):
        plt.close('all')

    def test_basic_plot(self):
        iterations = np.arange(1, 11)
        observed_hv = np.cumsum(np.random.uniform(0, 10, 10))
        fig, ax = create_hypervolume_convergence_plot(iterations, observed_hv)
        assert isinstance(fig, Figure)
        assert ax.get_xlabel() == 'Iteration'
        assert ax.get_ylabel() == 'Hypervolume'

    def test_with_predicted_hv(self):
        iterations = np.arange(1, 11)
        observed_hv = np.cumsum(np.random.uniform(0, 10, 10))
        predicted_hv = observed_hv + np.random.uniform(-1, 1, 10)
        predicted_std = np.random.uniform(0.5, 2.0, 10)
        fig, ax = create_hypervolume_convergence_plot(
            iterations, observed_hv,
            predicted_hv=predicted_hv,
            predicted_hv_std=predicted_std,
            sigma_bands=[1.0, 2.0],
        )
        assert isinstance(fig, Figure)

    def test_with_ref_point_annotation(self):
        iterations = np.arange(1, 6)
        observed_hv = np.array([1, 3, 5, 7, 8], dtype=float)
        fig, ax = create_hypervolume_convergence_plot(
            iterations, observed_hv,
            ref_point=[0.0, 0.0],
        )
        assert 'ref point' in ax.get_title().lower()

    def test_custom_title(self):
        iterations = np.arange(1, 6)
        observed_hv = np.array([1, 2, 3, 4, 5], dtype=float)
        fig, ax = create_hypervolume_convergence_plot(
            iterations, observed_hv, title='Custom Title'
        )
        assert ax.get_title() == 'Custom Title'

    def test_pre_existing_axes(self):
        existing_fig, existing_ax = plt.subplots()
        iterations = np.arange(1, 6)
        observed_hv = np.array([1, 2, 3, 4, 5], dtype=float)
        ret_fig, ret_ax = create_hypervolume_convergence_plot(
            iterations, observed_hv, ax=existing_ax
        )
        assert ret_ax is existing_ax


class TestRegretMOBO:
    """Tests for plot_regret routing to hypervolume convergence for MOBO."""

    def teardown_method(self):
        plt.close('all')

    def test_regret_mobo_requires_ref_point(self, mobo_session):
        _mock_mobo_model(mobo_session)
        with pytest.raises(ValueError, match="ref_point is required"):
            mobo_session.plot_regret()

    def test_regret_mobo_with_ref_point(self, mobo_session):
        _mock_mobo_model(mobo_session)
        fig = mobo_session.plot_regret(ref_point=[0.0, 0.0])
        assert isinstance(fig, Figure)
        # Should have 'Hypervolume' in y-axis label
        ax = fig.axes[0]
        assert 'hypervolume' in ax.get_ylabel().lower() or 'hypervolume' in ax.get_title().lower()


# ============================================================
# C. Suggested next → Pareto overlay
# ============================================================

class TestSuggestedNextMOBO:
    """Tests for plot_suggested_next routing to Pareto in MOBO."""

    def teardown_method(self):
        plt.close('all')

    def test_no_suggestions_raises(self, mobo_session):
        _mock_mobo_model(mobo_session)
        with pytest.raises(ValueError, match="No suggestions"):
            mobo_session.plot_suggested_next('x1')

    def test_mobo_routes_to_pareto(self, mobo_session):
        _mock_mobo_model(mobo_session)
        # Simulate stored suggestions
        mobo_session.last_suggestions = [
            {'x1': 0.5, 'x2': 0.3},
            {'x1': 0.7, 'x2': 0.8},
        ]
        fig = mobo_session.plot_suggested_next('x1')
        assert isinstance(fig, Figure)
        # Should contain Pareto-related elements
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert any('Suggested' in t for t in legend_texts)


# ============================================================
# D. Acquisition surfaces with MOBO
# ============================================================

class TestAcquisitionSurfaceMOBO:
    """Tests for acquisition surface plots with MOBO."""

    def teardown_method(self):
        plt.close('all')

    def test_acq_slice_mobo_no_acq_function_raises(self, mobo_session):
        _mock_mobo_model(mobo_session)
        mobo_session.acquisition = None
        with pytest.raises(ValueError, match="No MOBO acquisition"):
            mobo_session.plot_acquisition_slice('x1')

    def test_acq_slice_mobo_with_stored_acq(self, mobo_session):
        _mock_mobo_model(mobo_session)
        import torch
        # Mock acquisition
        mock_acq = MagicMock()
        mock_acq.return_value = torch.zeros(100)  # n_points default

        mock_acquisition_obj = MagicMock()
        mock_acquisition_obj.acq_function = mock_acq

        mobo_session.acquisition = mock_acquisition_obj
        mobo_session.last_suggestions = []

        # Mock _encode_categorical_data on model
        mobo_session.model._encode_categorical_data = lambda df: df

        fig = mobo_session.plot_acquisition_slice('x1', n_points=100)
        assert isinstance(fig, Figure)

    def test_acq_contour_mobo_with_stored_acq(self, mobo_session):
        _mock_mobo_model(mobo_session)
        import torch
        mock_acq = MagicMock()
        mock_acq.return_value = torch.linspace(0, 1, 25)  # 5x5 grid, varying values

        mock_acquisition_obj = MagicMock()
        mock_acquisition_obj.acq_function = mock_acq

        mobo_session.acquisition = mock_acquisition_obj
        mobo_session.last_suggestions = []

        mobo_session.model._encode_categorical_data = lambda df: df

        fig = mobo_session.plot_acquisition_contour('x1', 'x2', grid_resolution=5)
        assert isinstance(fig, Figure)


# ============================================================
# E. Blocked methods
# ============================================================

class TestBlockedMethods:
    """Tests for methods blocked in MOBO."""

    def test_probability_of_improvement_blocked(self, mobo_session):
        _mock_mobo_model(mobo_session)
        with pytest.raises(ValueError, match="not available for multi-objective"):
            mobo_session.plot_probability_of_improvement()

    def test_probability_of_improvement_ok_single(self, single_session):
        """PI plot should not raise the MOBO error for single-objective."""
        _mock_single_model(single_session)
        # It will fail for other reasons (not enough experiments, etc.)
        # but NOT with the MOBO error
        with pytest.raises(Exception) as exc_info:
            single_session.plot_probability_of_improvement()
        assert "not available for multi-objective" not in str(exc_info.value)


# ============================================================
# F. _get_predictions_for_objective helper
# ============================================================

class TestGetPredictionsForObjective:
    """Tests for _get_predictions_for_objective helper."""

    def test_extracts_from_dict(self, mobo_session):
        pred_dict = {
            'yield': (np.array([1.0, 2.0]), np.array([0.1, 0.2])),
            'selectivity': (np.array([3.0, 4.0]), np.array([0.3, 0.4])),
        }
        mean, std = mobo_session._get_predictions_for_objective(pred_dict, 'yield')
        np.testing.assert_array_equal(mean, [1.0, 2.0])
        np.testing.assert_array_equal(std, [0.1, 0.2])

    def test_returns_tuple_for_single_obj(self, single_session):
        pred_tuple = (np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        result = single_session._get_predictions_for_objective(pred_tuple, 'Output')
        assert result == pred_tuple


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
