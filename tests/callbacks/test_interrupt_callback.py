import pytest
import signal
from unittest.mock import MagicMock, patch

from src.callbacks.interrupt_callback import GracefulShutdownCallback

@pytest.mark.unit
class TestGracefulShutdownCallback:
    """Test suite for the GracefulShutdownCallback."""

    def setup_method(self):
        """Set up a callback instance with a mock model for each test."""
        self.callback = GracefulShutdownCallback()
        mock_model = MagicMock()
        mock_model.logger = MagicMock()
        self.callback.model = mock_model
    
    @patch('signal.signal')
    @patch('signal.getsignal')
    def test_handler_registration_on_training_start(self, mock_getsignal, mock_signal):
        """
        Test that the callback registers the SIGINT handler on training start
        and stores the original handler.
        """
        original_handler = MagicMock()
        mock_getsignal.return_value = original_handler
        
        self.callback._on_training_start()

        mock_getsignal.assert_called_once_with(signal.SIGINT)
        mock_signal.assert_called_once_with(signal.SIGINT, self.callback._signal_handler)
        assert self.callback.original_handler == original_handler

    def test_on_step_logic(self):
        """
        Test that _on_step returns True by default and False after a shutdown
        signal has been received.
        """
        # Should continue training by default
        assert self.callback._on_step() is True

        # Should stop training after shutdown is requested
        self.callback.shutdown_requested = True
        assert self.callback._on_step() is False
        
    def test_signal_handler_sets_flag_and_restores_handler(self):
        """
        Test that the signal handler sets the shutdown_requested flag and
        restores the original signal handler.
        """
        assert self.callback.shutdown_requested is False
        
        with patch('signal.signal') as mock_signal:
            self.callback.original_handler = MagicMock()
            # Manually trigger the handler
            self.callback._signal_handler(signal.SIGINT, None)
            
            assert self.callback.shutdown_requested is True
            # Verify that the original handler is restored to allow force-exit
            mock_signal.assert_called_once_with(signal.SIGINT, self.callback.original_handler) 