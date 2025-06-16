# Development Guide

This document provides guidelines for developers working on the Agentic Trading project, including instructions for running tests, maintaining code quality, and contributing.

## Running Tests

The project has a comprehensive test suite using `pytest`. A helper script is provided to run different categories of tests.

```bash
# Run unit tests (default, fast)
uv run python scripts/run_tests.py

# Run all tests (unit and integration)
uv run python scripts/run_tests.py --all

# Run only integration tests (requires credentials)
uv run python scripts/run_tests.py --integration

# Run a specific test by keyword
uv run python scripts/run_tests.py --integration -k "test_real_get_live_price"
```

For more granular control, you can also invoke `pytest` directly:

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=src

# Run a specific test file
uv run pytest tests/test_forex_com_broker.py -v
```

## Code Quality

We use `ruff` for linting and formatting and `mypy` for static type checking to ensure high code quality.

```bash
# Type checking with mypy
uv run mypy src/

# Linting with ruff
uv run ruff check src/

# Formatting with ruff
uv run ruff format src/
```

## Contributing

We welcome contributions! Please follow these steps:

1.  **Fork the repository** on GitHub.
2.  **Create a feature branch** for your changes (`git checkout -b feature/my-new-feature`).
3.  **Make your changes** and commit them with clear, descriptive messages.
4.  **Add tests** for your new code. We aim to maintain 100% test coverage on all critical components.
5.  **Ensure all tests and quality checks pass.**
6.  **Submit a pull request** to the `main` branch of the original repository.
