# ABOUTME: Puts the tests directory on sys.path so every suite can import the shared builders.
# ABOUTME: The package itself is installed by uv sync, so nothing here bootstraps restorebench.

import sys
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
