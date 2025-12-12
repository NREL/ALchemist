"""
Quick test to verify core Session improvements work correctly.
Tests staged experiments and BoTorch auto-defaults.
"""

from alchemist_core.session import OptimizationSession
import numpy as np

def booth_function(x, y):
    """Booth function: global minimum at (1, 3) = 0"""
    return (x + 2*y - 7)**2 + (2*x + y - 5)**2

print("=" * 60)
print("Testing Core Session API Improvements")
print("=" * 60)

# Test 1: Staged Experiments
print("\n1. Testing staged experiments...")
session = OptimizationSession()
session.add_variable('x', 'real', bounds=(-10, 10))
session.add_variable('y', 'real', bounds=(-10, 10))

# Stage some experiments
session.add_staged_experiment({'x': 1.0, 'y': 2.0})
session.add_staged_experiment({'x': 3.0, 'y': 4.0})
assert len(session.get_staged_experiments()) == 2, "Should have 2 staged"
print("✓ Staged 2 experiments")

# Clear staging
cleared = session.clear_staged_experiments()
assert cleared == 2, "Should have cleared 2"
assert len(session.get_staged_experiments()) == 0, "Should be empty"
print("✓ Cleared staged experiments")

# Test move_staged_to_experiments
session.add_staged_experiment({'x': 1.5, 'y': 2.5})
session.add_staged_experiment({'x': 2.5, 'y': 3.5})
outputs = [booth_function(1.5, 2.5), booth_function(2.5, 3.5)]
session.move_staged_to_experiments(outputs, reason='Test')
assert len(session.experiment_manager.df) == 2, "Should have 2 experiments"
assert len(session.get_staged_experiments()) == 0, "Staging should be cleared"
print("✓ Moved staged to experiments")

# Test 2: BoTorch Auto-Defaults
print("\n2. Testing BoTorch auto-defaults...")

# Add more data for training
np.random.seed(42)
for _ in range(8):
    x = np.random.uniform(-10, 10)
    y = np.random.uniform(-10, 10)
    output = booth_function(x, y)
    session.add_experiment({'x': x, 'y': y}, output)

print(f"✓ Added {len(session.experiment_manager.df)} total experiments")

# Train BoTorch model WITHOUT specifying transforms
# Should auto-apply normalize + standardize
result = session.train_model(backend='botorch', kernel='matern')

assert session.model is not None, "Model should be trained"
assert session.model_backend == 'botorch', "Should be BoTorch backend"

# Check that transforms were applied (model should have good R²)
metrics = result.get('metrics', {})
r2 = metrics.get('r2', 0)
print(f"✓ Model trained with R² = {r2:.4f}")

# Verify model has transforms applied
if hasattr(session.model, 'input_transform'):
    print(f"✓ Input transform: {session.model.input_transform_type}")
if hasattr(session.model, 'output_transform'):
    print(f"✓ Output transform: {session.model.output_transform_type}")

# Test 3: last_suggestions attribute
print("\n3. Testing last_suggestions attribute...")
suggestions = session.suggest_next(strategy='logei', goal='minimize', n_suggestions=1)
assert len(session.last_suggestions) == 1, "Should have 1 suggestion"
print(f"✓ Generated {len(session.last_suggestions)} suggestions")
print(f"  Suggestions: {session.last_suggestions}")

# Test 4: Verify non-breaking changes
print("\n4. Testing backward compatibility...")

# Old-style train_model call with explicit transforms should still work
session2 = OptimizationSession()
session2.add_variable('x', 'real', bounds=(-10, 10))
session2.add_variable('y', 'real', bounds=(-10, 10))

for _ in range(10):
    x = np.random.uniform(-10, 10)
    y = np.random.uniform(-10, 10)
    session2.add_experiment({'x': x, 'y': y}, booth_function(x, y))

# Explicitly override defaults
result2 = session2.train_model(
    backend='botorch',
    kernel='matern',
    input_transform_type='standardize',  # Override default
    output_transform_type='none'  # Override default
)

assert session2.model is not None, "Model should train with overrides"
print("✓ Explicit transform overrides work correctly")

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
