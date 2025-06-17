#!/usr/bin/env python3
"""Run tests for the agentic-trading project.
"""
import argparse
import os
import sys
from pathlib import Path

import pytest


def main():
    """Run tests using pytest.

    By default, this runs all unit tests (tests not marked as 'integration').
    Use the --integration flag to run only integration tests.
    Use the --all flag to run all tests.
    """
    parser = argparse.ArgumentParser(description="Run project tests.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--integration",
        action="store_true",
        help="Run only integration tests (requires credentials)."
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run all tests, including integration tests."
    )
    parser.add_argument(
        "--no-cov",
        action="store_true",
        help="Disable coverage reporting."
    )
    parser.add_argument(
        'pytest_args',
        nargs=argparse.REMAINDER,
        help='Additional arguments to pass to pytest'
    )
    args = parser.parse_args()

    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Let pytest discover tests based on pytest.ini::testpaths.
    pytest_cmd: list[str] = []

    if not args.no_cov:
        pytest_cmd.extend(["--cov=src", "--cov-report=term-missing"])

    if args.integration:
        print("Running INTEGRATION tests only.")
        pytest_cmd.extend(["-m", "integration"])
    elif args.all:
        print("Running ALL tests (unit and integration).")
        # No marker needed, pytest runs all by default.
    else:
        print("Running UNIT tests only (use --all or --integration to run others).")
        pytest_cmd.extend(["-m", "not integration"])

    # Add any other arguments passed by the user
    if args.pytest_args:
        # If the first arg is '--', strip it. argparse.REMAINDER includes it.
        if args.pytest_args and args.pytest_args[0] == '--':
            pytest_cmd.extend(args.pytest_args[1:])
        else:
            pytest_cmd.extend(args.pytest_args)

    print(f"Executing: pytest {' '.join(pytest_cmd)}")
    print("="*60)

    exit_code = pytest.main(pytest_cmd)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
