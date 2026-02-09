"""
Unit tests for create_pareto_plot() in alchemist_core.visualization.plots.

Tests the Pareto frontier plotting function with various parameter combinations
including suggested points, constraint boundaries, reference points, custom
titles/figsize, and pre-existing axes.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from alchemist_core.visualization.plots import create_pareto_plot


@pytest.fixture
def pareto_data():
    """Create sample 2-objective data with Pareto mask."""
    np.random.seed(42)
    Y = np.random.uniform(0, 100, (20, 2))
    # Simple Pareto mask: mark first 5 as Pareto-optimal
    pareto_mask = np.zeros(20, dtype=bool)
    pareto_mask[:5] = True
    return Y, pareto_mask


class TestParetoPlot:
    """Test Pareto frontier plot creation."""

    def teardown_method(self):
        """Close all figures after each test to avoid memory leaks."""
        plt.close('all')

    def test_basic_2_objective_plot(self, pareto_data):
        """Test basic 2-objective Pareto plot returns (Figure, Axes) tuple."""
        Y, pareto_mask = pareto_data
        objective_names = ['yield', 'selectivity']

        result = create_pareto_plot(Y, pareto_mask, objective_names)

        assert isinstance(result, tuple)
        assert len(result) == 2
        fig, ax = result
        assert isinstance(fig, Figure)
        assert ax is not None
        assert ax.get_xlabel() == 'yield'
        assert ax.get_ylabel() == 'selectivity'
        assert ax.get_title() == 'Pareto Frontier'

    def test_with_suggested_points(self, pareto_data):
        """Test Pareto plot with suggested points overlay."""
        Y, pareto_mask = pareto_data
        objective_names = ['yield', 'selectivity']
        suggested_points = np.array([[50.0, 70.0], [80.0, 30.0], [60.0, 60.0]])

        fig, ax = create_pareto_plot(
            Y, pareto_mask, objective_names,
            suggested_points=suggested_points,
        )

        assert isinstance(fig, Figure)
        assert ax is not None
        # Check that legend includes 'Suggested' label
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert any('Suggested' in text for text in legend_texts)

    def test_with_constraint_boundaries(self, pareto_data):
        """Test Pareto plot with constraint boundary lines."""
        Y, pareto_mask = pareto_data
        objective_names = ['yield', 'selectivity']
        constraint_boundaries = {'yield': 60.0, 'selectivity': 80.0}

        fig, ax = create_pareto_plot(
            Y, pareto_mask, objective_names,
            constraint_boundaries=constraint_boundaries,
        )

        assert isinstance(fig, Figure)
        assert ax is not None
        # Check that legend includes constraint boundary labels
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert any('yield bound' in text for text in legend_texts)
        assert any('selectivity bound' in text for text in legend_texts)

    def test_with_ref_point(self, pareto_data):
        """Test Pareto plot with explicit reference point and hypervolume shading."""
        Y, pareto_mask = pareto_data
        objective_names = ['yield', 'selectivity']
        ref_point = [0.0, 0.0]

        fig, ax = create_pareto_plot(
            Y, pareto_mask, objective_names,
            ref_point=ref_point,
            show_hypervolume=True,
        )

        assert isinstance(fig, Figure)
        assert ax is not None
        # Check that legend includes hypervolume and ref point labels
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert any('Hypervolume' in text for text in legend_texts)
        assert any('Ref point' in text for text in legend_texts)

    def test_custom_title_and_figsize(self, pareto_data):
        """Test Pareto plot with custom title and figure size."""
        Y, pareto_mask = pareto_data
        objective_names = ['yield', 'selectivity']
        custom_title = 'Multi-Objective Optimization Results'
        custom_figsize = (12, 9)

        fig, ax = create_pareto_plot(
            Y, pareto_mask, objective_names,
            title=custom_title,
            figsize=custom_figsize,
        )

        assert ax.get_title() == custom_title
        # Verify figure size was applied (in inches)
        fig_width, fig_height = fig.get_size_inches()
        assert abs(fig_width - 12.0) < 0.01
        assert abs(fig_height - 9.0) < 0.01

    def test_with_pre_existing_axes(self, pareto_data):
        """Test Pareto plot rendered onto a pre-existing axes object."""
        Y, pareto_mask = pareto_data
        objective_names = ['yield', 'selectivity']

        existing_fig, existing_ax = plt.subplots()

        returned_fig, returned_ax = create_pareto_plot(
            Y, pareto_mask, objective_names,
            ax=existing_ax,
        )

        # Should return the same axes and figure that were passed in
        assert returned_ax is existing_ax
        assert returned_fig is existing_fig

    def test_no_pareto_points(self):
        """Test Pareto plot when no points are marked as Pareto-optimal."""
        np.random.seed(99)
        Y = np.random.uniform(0, 100, (10, 2))
        pareto_mask = np.zeros(10, dtype=bool)
        objective_names = ['obj1', 'obj2']

        fig, ax = create_pareto_plot(Y, pareto_mask, objective_names)

        assert isinstance(fig, Figure)
        assert ax is not None

    def test_all_pareto_points(self):
        """Test Pareto plot when all points are Pareto-optimal."""
        np.random.seed(7)
        Y = np.random.uniform(0, 100, (8, 2))
        pareto_mask = np.ones(8, dtype=bool)
        objective_names = ['obj1', 'obj2']

        fig, ax = create_pareto_plot(Y, pareto_mask, objective_names)

        assert isinstance(fig, Figure)
        assert ax is not None
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert any('Pareto optimal' in text for text in legend_texts)

    def test_directions_parameter(self, pareto_data):
        """Test that directions parameter is accepted without error."""
        Y, pareto_mask = pareto_data
        objective_names = ['yield', 'selectivity']
        directions = ['maximize', 'minimize']

        fig, ax = create_pareto_plot(
            Y, pareto_mask, objective_names,
            directions=directions,
        )

        assert isinstance(fig, Figure)
        assert ax is not None

    def test_show_hypervolume_false(self, pareto_data):
        """Test Pareto plot with hypervolume shading disabled."""
        Y, pareto_mask = pareto_data
        objective_names = ['yield', 'selectivity']
        ref_point = [0.0, 0.0]

        fig, ax = create_pareto_plot(
            Y, pareto_mask, objective_names,
            ref_point=ref_point,
            show_hypervolume=False,
        )

        assert isinstance(fig, Figure)
        assert ax is not None
        # Hypervolume shading should not appear in legend
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert not any('Hypervolume' in text for text in legend_texts)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
