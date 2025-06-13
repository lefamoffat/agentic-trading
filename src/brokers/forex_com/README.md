# Forex.com Broker Module

This directory contains the integration for the Forex.com (Gain Capital) broker, built using a modular, composition-based architecture.

## 🏛️ Architecture: Composition over Inheritance

Instead of a single monolithic class, the `ForexComBroker` is designed as an orchestrator that is _composed_ of several specialized handler classes. This aligns with the "Composition over Inheritance" principle, leading to a more modular, maintainable, and testable design.

The main broker class, `ForexComBroker` (in `broker.py`), delegates specific responsibilities to its component handlers.

## 🧩 Handler Responsibilities

Each file in this module has a single, clear responsibility:

-   **`broker.py`**: The public-facing `ForexComBroker` class. It initializes all other handlers and exposes the public methods required by the `BaseBroker` interface (e.g., `authenticate`, `get_account_info`, `place_order`).
-   **`auth.py` (`AuthenticationHandler`)**: Handles the authentication flow, including session token management. It is responsible for making the initial connection to the `/Session` endpoint.
-   **`api.py` (`ApiClient`)**: A low-level client responsible for making all authenticated HTTP requests to the Forex.com API. It uses the session token from the `AuthenticationHandler` and contains robust JSON parsing and error handling. It also manages the symbol-to-market-ID mapping.
-   **`account.py` (`AccountHandler`)**: Handles all account-related queries, such as fetching the account balance, currency, and available funds.
-   **`positions.py` (`PositionHandler`)**: Responsible for fetching and parsing current open trading positions.
-   **`orders.py` (`OrderHandler`)**: Responsible for fetching active orders, placing new orders, and canceling existing orders.
-   **`data.py` (`DataHandler`)**: Handles all market data requests, including fetching historical price bars and live price data.

## 🧪 Testing Strategy

The tests for this broker are located within the `tests/` subdirectory, ensuring they are co-located with the code they test.

-   **Unit Tests**: Each handler (`api.py`, `auth.py`, etc.) has a corresponding unit test file (`test_api.py`, `test_auth.py`). These tests mock out dependencies (like the `ApiClient`) to test each component in isolation.
-   **Integration Tests**: Real-world tests that require credentials are in `test_integration.py`. These tests are marked with `@pytest.mark.integration` and are skipped by default.
-   **Shared Fixtures (`conftest.py`)**: Shared test utilities, most importantly the `create_async_session_mock` helper for mocking `aiohttp` sessions, are defined in `conftest.py`. This makes them automatically available to all test files in this directory, promoting DRY principles in our test code.
