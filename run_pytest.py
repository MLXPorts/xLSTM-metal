#!/usr/bin/env python
"""
Convenient test runner script for xLSTM-Metal tests.

Usage:
    python run_pytest.py                    # Run all tests
    python run_pytest.py -k test_mlx        # Run tests matching pattern
    python run_pytest.py -m unit            # Run tests with 'unit' marker
    python run_pytest.py -m "not slow"      # Skip slow tests
    python run_pytest.py --cov              # Run with coverage
    python run_pytest.py --help             # Show pytest help
"""

import sys
import subprocess
from pathlib import Path


def main():
    """Run pytest with the given arguments."""
    # Ensure we're in the project root
    project_root = Path(__file__).parent
    
    # Build pytest command
    cmd = [sys.executable, "-m", "pytest"]
    
    # Add any command-line arguments passed to this script
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    else:
        # Default: run all tests in tests/ directory
        cmd.append("tests/")
    
    # Run pytest
    try:
        result = subprocess.run(cmd, cwd=project_root)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n\nTest run interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error running tests: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
