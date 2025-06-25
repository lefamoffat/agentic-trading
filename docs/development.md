# Development Guide

This document provides guidelines for developers working on the Agentic Trading project, including instructions for running tests, maintaining code quality, and contributing.

## Experiment Data Systems

The platform uses two complementary systems for experiment data management:

### Messaging System (`@/messaging`)

-   **Purpose**: Real-time communication and live updates
-   **Storage**: Redis (production) or Memory (development)
-   **Data Lifespan**: Session-based, temporary
-   **Use Cases**: Live progress monitoring, dashboard updates, CLI notifications

### Tracking System (`@/tracking`)

-   **Purpose**: Persistent experiment logging and analysis
-   **Storage**: Aim backend (persistent files)
-   **Data Lifespan**: Permanent, for historical analysis
-   **Use Cases**: Experiment comparison, model management, performance analysis

**Architecture Guidelines:**

**✅ ALLOWED:**

-   Backend-specific imports within `src/tracking/` module
-   Backend-specific imports within `src/messaging/` module
-   Integration tests in `integration_tests/` for cross-module testing

**❌ NOT ALLOWED:**

-   Backend-specific imports in application code outside these modules
-   Aim/Redis imports outside their respective modules
-   Backend-specific tests in `src/*/tests/` (use `integration_tests/` instead)

**📚 Detailed Documentation:**

-   [Messaging System README](../src/messaging/README.md) - Complete messaging guide
-   [Tracking System README](../src/tracking/README.md) - Complete tracking guide
-   [Experiment Data Management](experiment_data_management.md) - How systems work together

### Correct Usage Patterns

```python
# ✅ CORRECT: Generic usage (anywhere in codebase)
from src.tracking import get_ml_tracker, get_experiment_repository
tracker = await get_ml_tracker()  # Currently returns Aim backend

# ❌ WRONG: Backend-specific usage outside tracking module
from src.tracking.factory import configure_aim_backend  # DON'T DO THIS
```

### Test Guidelines by Location

-   `src/tracking/tests/`: Unit/component tests (module-only, no external deps)
-   `integration_tests/`: Cross-module tests (Aim backend, environment, CLI)

## Running Tests

Tests follow strict patterns defined in `@testing.mdc` with comprehensive health monitoring and schema validation.

### Test Structure

```
src/*/tests/              # Unit/component tests (module-only, no external deps)
integration_tests/        # Integration tests (cross-module, external deps)
```

### Test Commands

```bash
# Run by test type (following @testing.mdc markers)
uv run pytest -m unit                    # Pure logic tests, <50ms
uv run pytest -m component               # Component tests with mocks, <1s
uv run pytest -m integration             # Integration tests with real backends

# Run by location
uv run pytest src/*/tests                # Unit/component tests (fast)
uv run pytest integration_tests/         # Integration tests (comprehensive)

# Coverage report
uv run pytest -m "not integration" --cov=src

# Health monitoring validation (catches schema issues)
uv run pytest integration_tests/test_backend_schema_validation.py

# CLI integration testing
uv run pytest integration_tests/test_cli_health_monitoring.py
```

### Key Test Features

Our test suite would **automatically catch** the SQLite schema issues we encountered:

-   ✅ **Schema Validation**: `test_detect_empty_database_schema_issue()` catches "no such table: run"
-   ✅ **Health Monitoring**: Comprehensive backend health validation
-   ✅ **CLI Integration**: Complete workflow testing with error detection
-   ✅ **Generic Testing**: Uses proper interfaces, no backend lock-in

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
