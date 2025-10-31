"""
Pytest configuration for API tests.
"""

import sys
from pathlib import Path

# Add project root to path - must be done before any api imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"Added to sys.path: {project_root}")
