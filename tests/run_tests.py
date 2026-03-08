"""
Convenience runner: python tests/run_tests.py
Equivalent to: python -m pytest tests/ -v
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v"],
        cwd=repo_root,
    )
    sys.exit(result.returncode)
