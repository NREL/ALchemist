"""End-to-end integration test for multi-objective Bayesian optimization workflow."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from alchemist_core import OptimizationSession


class TestMOBOWorkflow:
    """Full MOBO workflow: define space -> load data -> train -> suggest -> find_optimum."""

    @pytest.fixture
    def mobo_session(self):
        """Set up a complete MOBO session with data."""
        session = OptimizationSession()

        # Step 1: Define search space
        session.add_variable('x1', 'real', bounds=(0.0, 1.0))
        session.add_variable('x2', 'real', bounds=(0.0, 1.0))

        # Step 2: Add input constraint (x1 + x2 <= 1.5)
        session.add_input_constraint('inequality', {'x1': 1.0, 'x2': 1.0}, rhs=1.5)

        # Step 3: Create and load 2-objective data
        np.random.seed(42)
        n = 15
        x1 = np.random.uniform(0, 1, n)
        x2 = np.random.uniform(0, 1, n)
        # yield = f(x1, x2) + noise
        y = 50 + 30 * x1 + 20 * x2 + np.random.normal(0, 2, n)
        # selectivity = g(x1, x2) + noise
        sel = 80 - 10 * x1 + 15 * x2 + np.random.normal(0, 2, n)

        data = pd.DataFrame({
            'x1': x1, 'x2': x2,
            'yield': y, 'selectivity': sel,
        })

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            data.to_csv(f, index=False)
            filepath = f.name

        try:
            session.load_data(filepath, target_columns=['yield', 'selectivity'])
        finally:
            os.unlink(filepath)

        return session

    def test_full_mobo_workflow(self, mobo_session):
        """End-to-end MOBO: train -> suggest -> find_optimum -> predict."""
        session = mobo_session

        # Verify multi-objective properties
        assert session.is_multi_objective is True
        assert session.n_objectives == 2
        assert session.objective_names == ['yield', 'selectivity']

        # Verify input constraints
        constraints = session.search_space.get_constraints()
        assert len(constraints) == 1
        assert constraints[0]['type'] == 'inequality'

        # Step 4: Add outcome constraint (selectivity >= 75)
        session.add_outcome_constraint('selectivity', 'lower', 75.0)
        assert len(session.get_outcome_constraints()) == 1

        # Step 5: Train model with botorch
        result = session.train_model(backend='botorch', kernel='Matern')
        assert result['success'] is True
        assert session.model is not None
        assert session.model.n_objectives == 2

        # Step 6: Suggest next experiments with qNEHVI
        suggestions = session.suggest_next(
            strategy='qNEHVI',
            goal=['maximize', 'maximize'],
            n_suggestions=1,
        )
        assert suggestions is not None
        # Result should be a DataFrame or dict with x1, x2
        if isinstance(suggestions, pd.DataFrame):
            assert 'x1' in suggestions.columns
            assert 'x2' in suggestions.columns
        elif isinstance(suggestions, dict):
            assert 'x1' in suggestions
            assert 'x2' in suggestions

        # Step 7: Find Pareto optimum
        pareto_result = session.find_optimum(goal=['maximize', 'maximize'])
        assert 'pareto_frontier' in pareto_result
        assert 'predicted_values' in pareto_result
        assert 'n_pareto' in pareto_result
        assert pareto_result['n_pareto'] > 0

        # Step 8: Multi-objective predictions
        test_points = pd.DataFrame({
            'x1': [0.3, 0.5, 0.7],
            'x2': [0.4, 0.5, 0.6],
        })
        predictions = session.predict(test_points)
        # Multi-objective predict returns dict keyed by objective name
        assert isinstance(predictions, dict)
        assert 'yield' in predictions
        assert 'selectivity' in predictions
        for name, (mean, std) in predictions.items():
            assert len(mean) == 3
            assert len(std) == 3

    def test_sklearn_guard_on_mobo(self, mobo_session):
        """Verify sklearn backend raises helpful error for multi-objective."""
        with pytest.raises(ValueError, match="botorch"):
            mobo_session.train_model(backend='sklearn', kernel='Matern')

    def test_invalid_mobo_strategy_raises(self, mobo_session):
        """Verify non-MOBO strategies raise error for multi-objective."""
        mobo_session.train_model(backend='botorch', kernel='Matern')
        with pytest.raises(ValueError):
            mobo_session.suggest_next(strategy='EI', goal='maximize')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
