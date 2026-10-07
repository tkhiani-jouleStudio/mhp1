"""Conftest for agent unit tests — ensures app/ is on sys.path."""
import sys
from pathlib import Path

APP_DIR = str(Path(__file__).parent.parent / "app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
