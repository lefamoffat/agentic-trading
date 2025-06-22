"""Unit tests for cache-to-qlib conversion logic."""

from datetime import datetime, timezone
import pytest

from src.market_data.contracts import QlibDataSpec, MarketDataRequest
from src.types import DataSource, Timeframe

@pytest.mark.unit 
class TestQlibPathValidation:
    """Unit tests specifically for qlib path validation logic."""
    
    def test_symbol_slash_replacement_comprehensive(self):
        """Comprehensive test of symbol slash replacement in various contexts."""
        test_cases = [
            # (input_symbol, expected_dir_component)
            ("EUR/USD", "EUR_USD"),
            ("GBP/JPY", "GBP_JPY"),
            ("BTC/USD", "BTC_USD"),
            ("AAPL", "AAPL"),  # No slash
            ("A/B/C", "A_B_C"),  # Multiple slashes
            ("", ""),  # Empty string
            ("USD/", "USD_"),  # Trailing slash
            ("/EUR", "_EUR"),  # Leading slash
        ]
        
        for input_symbol, expected_dir in test_cases:
            spec = QlibDataSpec(
                symbol=input_symbol,
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
                timeframe=Timeframe.H1,
                qlib_dir="/test"
            )
            
            result_dir = spec.get_qlib_symbol_dir()
            expected_full_dir = f"/test/{expected_dir}"
            
            assert result_dir == expected_full_dir, f"Failed for '{input_symbol}': got '{result_dir}', expected '{expected_full_dir}'"
            
            # Also test file path
            result_file = spec.get_qlib_file_path()
            expected_full_file = f"/test/{expected_dir}/1h.bin"
            
            assert result_file == expected_full_file, f"File path failed for '{input_symbol}': got '{result_file}', expected '{expected_full_file}'"
    
    def test_qlib_path_generation_edge_cases(self):
        """Test qlib path generation for edge cases and various symbol formats."""
        test_cases = [
            # (symbol, expected_dir_name)
            ("EUR/USD", "EUR_USD"),
            ("AAPL", "AAPL"),  # No slash
            ("BTC/USDT", "BTC_USDT"),
            ("NASDAQ:GOOGL", "NASDAQ:GOOGL"),  # Different separator
            ("SPX500", "SPX500"),  # No special characters
            ("GBP/JPY", "GBP_JPY"),
        ]
        
        for symbol, expected_dir in test_cases:
            request = MarketDataRequest(
                symbol=symbol,
                source=DataSource.FOREX_COM,
                timeframe=Timeframe.H1,
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 2, tzinfo=timezone.utc)
            )
            
            # Test QlibDataSpec path generation
            spec = QlibDataSpec(
                symbol=symbol,
                start_date=request.start_date,
                end_date=request.end_date,
                timeframe=request.timeframe,
                qlib_dir="/mock/qlib"
            )
            
            predicted_dir = spec.get_qlib_symbol_dir()
            expected_path = f"/mock/qlib/{expected_dir}"
            
            assert predicted_dir == expected_path, f"Path generation failed for {symbol}: got {predicted_dir}, expected {expected_path}"
    
    def test_path_consistency_between_methods(self):
        """Test that different path generation methods are consistent."""
        symbol = "EUR/USD"
        
        spec = QlibDataSpec(
            symbol=symbol,
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
            timeframe=Timeframe.H1,
            qlib_dir="/test/qlib"
        )
        
        symbol_dir = spec.get_qlib_symbol_dir()
        file_path = spec.get_qlib_file_path()
        
        # File path should be symbol dir + timeframe + .bin
        expected_file_path = f"{symbol_dir}/1h.bin"
        assert file_path == expected_file_path, f"File path inconsistent: got {file_path}, expected {expected_file_path}"
        
        # Symbol dir should end with the sanitized symbol
        assert symbol_dir.endswith("EUR_USD"), f"Symbol dir should end with EUR_USD: {symbol_dir}" 