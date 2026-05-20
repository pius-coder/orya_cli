"""pytest configuration for agent v3 tests."""
import sys
from pathlib import Path

# Add project root to path so 'agent' imports resolve
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
