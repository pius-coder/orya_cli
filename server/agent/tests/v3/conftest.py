"""pytest configuration for agent tests.

Adds the server/ directory to sys.path so `import agent` works.
"""

import sys
from pathlib import Path

# Add server/ to path so `import agent` resolves
_server_dir = Path(__file__).parent.parent.parent.parent  # server/
sys.path.insert(0, str(_server_dir))
