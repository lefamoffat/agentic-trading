"""
Shared fixtures and test utilities for Forex.com tests.
"""

import json
from unittest.mock import AsyncMock, MagicMock

def create_async_session_mock(response_data, status_code=200):
    """
    Helper to create a properly mocked aiohttp.ClientSession that
    handles nested async context managers, as required by the aiohttp library.
    """
    # 1. The final response object that the inner context manager returns.
    #    It has async methods like .json() and .text().
    mock_response = AsyncMock()
    mock_response.status = status_code
    mock_response.json.return_value = response_data
    mock_response.text.return_value = json.dumps(response_data) if response_data else ""

    # 2. The context manager that session.request() returns.
    #    Its __aenter__ must be a coroutine.
    request_context_manager = AsyncMock()
    request_context_manager.__aenter__.return_value = mock_response

    # 3. The session object. Its .request() method is SYNCHRONOUS and returns
    #    the context manager from step 2. We use MagicMock for this.
    mock_session = MagicMock()
    mock_session.request.return_value = request_context_manager

    # 4. The main context manager for the session itself, which is entered
    #    via "async with aiohttp.ClientSession() as session:".
    session_context_manager = AsyncMock()
    session_context_manager.__aenter__.return_value = mock_session
    
    return session_context_manager 