import signal
from unittest.mock import MagicMock, patch

import pytest

from src.callbacks.interrupt_callback import GracefulShutdownCallback


@pytest.mark.unit
class TestGracefulShutdownCallback:
    """Test suite for the GracefulShutdownCallback."""

    def setup_method(self):
        self.callback = GracefulShutdownCallback()
        self.callback.model = MagicMock()

    @patch("signal.signal")
    @patch("signal.getsignal")
    def test_handler_registration_on_training_start(self, mock_getsignal, mock_signal):
        original_handler = MagicMock()
        mock_getsignal.return_value = original_handler

        self.callback.on_training_start(locals(), globals())

        mock_getsignal.assert_called_once_with(signal.SIGINT)
        mock_signal.assert_called_once_with(signal.SIGINT, self.callback._signal_handler)
        assert self.callback.original_handler == original_handler

    def test_on_step_logic(self):
        assert self.callback._on_step() is True
        self.callback.shutdown_requested = True
        assert self.callback._on_step() is False

    def test_signal_handler_sets_flag_and_restores_handler(self):
        assert self.callback.shutdown_requested is False
        self.callback.original_handler = MagicMock()

        with patch("signal.signal") as mock_signal:
            self.callback._signal_handler(signal.SIGINT, None)
            assert self.callback.shutdown_requested is True
            mock_signal.assert_called_once_with(signal.SIGINT, self.callback.original_handler) 