"""
Custom exceptions for broker interactions.
"""

class JsonParseError(Exception):
    """
    Raised when an API response that should be JSON cannot be parsed.
    This indicates a violation of the API contract.
    """
    pass 