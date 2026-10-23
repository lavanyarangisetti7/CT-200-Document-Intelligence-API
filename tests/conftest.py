"""
pytest configuration — ensures app/ is importable as a package.
"""
import sys
from pathlib import Path

# Add project root to sys.path so `import app` works from the tests/ dir.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
