"""
Tests for multi-objective session orchestration in OptimizationSession.

Covers MOBO properties, input/outcome constraint APIs, multi-objective
train_model guards, and suggest_next MOBO validation.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from alchemist_core import OptimizationSession
from alchemist_core.data.search_space import SearchSpace
from alchemist_core.data.experiment_manager import ExperimentManager


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mobo_session():
    """Session with 2-objective data loaded."""
    session = OptimizationSession()
    session.add_variable('x1', 'real', bounds=(0.0, 1.0))
    session.add_variable('x2', 'real', bounds=(0.0, 1.0))

    # Create 2-objective CSV data
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
def single_objective_session():
    """Session with single-objective data loaded."""
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


# ============================================================
# Tests: MOBO Properties
# ============================================================

class TestMOBOProperties:
    """Tests for multi-objective property accessors on OptimizationSession."""

    def test_is_multi_objective_false_for_single(self, single_objective_session):
        """is_multi_objective returns False when there is one target column."""
        assert single_objective_session.is_multi_objective is False

    def test_is_multi_objective_true_for_two_targets(self, mobo_session):
        """is_multi_objective returns True when there are 2+ target columns."""
        assert mobo_session.is_multi_objective is True

    def test_n_objectives_single(self, single_objective_session):
        """n_objectives returns 1 for a single-target session."""
        assert single_objective_session.n_objectives == 1

    def test_n_objectives_multi(self, mobo_session):
        """n_objectives returns 2 for a two-target session."""
        assert mobo_session.n_objectives == 2

    def test_objective_names_single(self, single_objective_session):
        """objective_names returns the single target column name."""
        names = single_objective_session.objective_names
        assert names == ['Output']

    def test_objective_names_multi(self, mobo_session):
        """objective_names returns both target column names in order."""
        names = mobo_session.objective_names
        assert names == ['yield', 'selectivity']


# ============================================================
# Tests: Input Constraint API
# ============================================================

class TestInputConstraintAPI:
    """Tests for add_input_constraint delegation to SearchSpace."""

    def test_add_input_constraint_delegates_to_search_space(self, mobo_session):
        """add_input_constraint delegates to search_space.add_constraint."""
        mobo_session.add_input_constraint(
            'inequality', {'x1': 1.0, 'x2': 1.0}, rhs=1.5
        )
        constraints = mobo_session.search_space.get_constraints()
        assert len(constraints) == 1
        assert constraints[0]['type'] == 'inequality'
        assert constraints[0]['coefficients'] == {'x1': 1.0, 'x2': 1.0}
        assert constraints[0]['rhs'] == 1.5

    def test_constraints_stored_on_search_space(self, mobo_session):
        """Multiple constraints accumulate on the search space."""
        mobo_session.add_input_constraint(
            'inequality', {'x1': 1.0, 'x2': 1.0}, rhs=1.5
        )
        mobo_session.add_input_constraint(
            'equality', {'x1': 1.0}, rhs=0.5, name='fix_x1'
        )
        constraints = mobo_session.search_space.get_constraints()
        assert len(constraints) == 2
        assert constraints[1]['type'] == 'equality'
        assert constraints[1]['name'] == 'fix_x1'


# ============================================================
# Tests: Outcome Constraint API
# ============================================================

class TestOutcomeConstraintAPI:
    """Tests for add_outcome_constraint and get_outcome_constraints."""

    def test_add_outcome_constraint_stores_constraint(self, mobo_session):
        """add_outcome_constraint appends to _outcome_constraints."""
        mobo_session.add_outcome_constraint('selectivity', 'lower', 80.0)
        constraints = mobo_session.get_outcome_constraints()
        assert len(constraints) == 1
        assert constraints[0]['objective_name'] == 'selectivity'
        assert constraints[0]['bound_type'] == 'lower'
        assert constraints[0]['value'] == 80.0

    def test_add_outcome_constraint_invalid_bound_type_raises(self, mobo_session):
        """Invalid bound_type raises ValueError."""
        with pytest.raises(ValueError, match="bound_type must be 'lower' or 'upper'"):
            mobo_session.add_outcome_constraint('selectivity', 'invalid', 80.0)

    def test_get_outcome_constraints_returns_copies(self, mobo_session):
        """get_outcome_constraints returns copies, not references to internal list."""
        mobo_session.add_outcome_constraint('yield', 'upper', 95.0)
        constraints = mobo_session.get_outcome_constraints()

        # Mutating the returned dict should not affect internal state
        constraints[0]['value'] = 999.0
        internal = mobo_session.get_outcome_constraints()
        assert internal[0]['value'] == 95.0

    def test_multiple_outcome_constraints(self, mobo_session):
        """Multiple outcome constraints can be added and retrieved."""
        mobo_session.add_outcome_constraint('selectivity', 'lower', 80.0)
        mobo_session.add_outcome_constraint('yield', 'upper', 95.0)
        constraints = mobo_session.get_outcome_constraints()
        assert len(constraints) == 2
        assert constraints[0]['objective_name'] == 'selectivity'
        assert constraints[1]['objective_name'] == 'yield'


# ============================================================
# Tests: train_model with multi-objective
# ============================================================

class TestTrainModelMOBO:
    """Tests for train_model with multi-objective sessions."""

    def test_sklearn_backend_raises_for_multi_objective(self, mobo_session):
        """Calling train_model(backend='sklearn') on a MOBO session raises ValueError."""
        with pytest.raises(ValueError, match='botorch'):
            mobo_session.train_model(backend='sklearn')

    def test_botorch_backend_succeeds_for_multi_objective(self, mobo_session):
        """Training with botorch backend succeeds for 2-objective session."""
        result = mobo_session.train_model(backend='botorch', kernel='Matern')
        assert result['success'] is True

    def test_botorch_model_has_correct_n_objectives(self, mobo_session):
        """After training, the model reports n_objectives == 2."""
        mobo_session.train_model(backend='botorch', kernel='Matern')
        assert mobo_session.model.n_objectives == 2

    def test_train_model_returns_dict(self, mobo_session):
        """train_model returns a dict with expected keys."""
        result = mobo_session.train_model(backend='botorch', kernel='Matern')
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'backend' in result
        assert 'n_objectives' in result
        assert result['n_objectives'] == 2


# ============================================================
# Tests: suggest_next MOBO validation
# ============================================================

class TestSuggestNextMOBOValidation:
    """Tests for suggest_next strategy validation in multi-objective mode."""

    def test_invalid_strategy_raises_for_multi_objective(self, mobo_session):
        """Non-MOBO strategies raise ValueError for multi-objective sessions."""
        mobo_session.train_model(backend='botorch', kernel='Matern')
        with pytest.raises(ValueError, match="qEHVI.*qNEHVI"):
            mobo_session.suggest_next(strategy='EI', goal='maximize')

    def test_pi_strategy_raises_for_multi_objective(self, mobo_session):
        """PI strategy raises ValueError for multi-objective sessions."""
        mobo_session.train_model(backend='botorch', kernel='Matern')
        with pytest.raises(ValueError, match="qEHVI.*qNEHVI"):
            mobo_session.suggest_next(strategy='PI', goal='maximize')

    def test_ucb_strategy_raises_for_multi_objective(self, mobo_session):
        """UCB strategy raises ValueError for multi-objective sessions."""
        mobo_session.train_model(backend='botorch', kernel='Matern')
        with pytest.raises(ValueError, match="qEHVI.*qNEHVI"):
            mobo_session.suggest_next(strategy='UCB', goal='maximize')

    def test_goal_as_string_accepted(self, mobo_session):
        """A single string goal is broadcast to all objectives."""
        mobo_session.train_model(backend='botorch', kernel='Matern')
        # Should not raise -- qNEHVI is a valid MOBO strategy
        suggestions = mobo_session.suggest_next(
            strategy='qNEHVI', goal='maximize'
        )
        assert isinstance(suggestions, pd.DataFrame)
        assert len(suggestions) >= 1

    def test_goal_as_list_accepted(self, mobo_session):
        """A list of per-objective goals is accepted."""
        mobo_session.train_model(backend='botorch', kernel='Matern')
        suggestions = mobo_session.suggest_next(
            strategy='qNEHVI', goal=['maximize', 'maximize']
        )
        assert isinstance(suggestions, pd.DataFrame)
        assert len(suggestions) >= 1
