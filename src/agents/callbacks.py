#!/usr/bin/env python3
"""Agent training callbacks for Stable-Baselines3.
"""
# Standard libs
import signal
import logging

# Third-party
from stable_baselines3.common.callbacks import BaseCallback

class GracefulShutdownCallback(BaseCallback):
    """A custom callback to save the model and exit gracefully on SIGINT (Ctrl+C).

    This callback listens for a shutdown signal and, when received, stops the
    training loop cleanly. The main script is responsible for catching the
    signal and performing the actual save operations.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.shutdown_requested = False
        self.original_handler = None

    def _on_training_start(self) -> None:
        """This method is called before the first rollout starts.
        It registers the signal handler.
        """
        self.original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._signal_handler)
        logging.getLogger(__name__).info("Registered SIGINT handler for graceful shutdown.")

    def _signal_handler(self, signum, frame):
        """The handler that sets the shutdown flag.
        """
        logging.getLogger(__name__).warning("Shutdown signal received! Finishing current step and saving...")
        self.shutdown_requested = True
        # Restore original handler to allow force-exit if needed again
        if self.original_handler:
            signal.signal(signal.SIGINT, self.original_handler)

    def _on_step(self) -> bool:
        """This method is called after each step in the training process.
        It checks if a shutdown has been requested.

        Returns:
            bool: False if training should stop, True otherwise.

        """
        if self.shutdown_requested:
            logging.getLogger(__name__).info("Stopping training gracefully.")
            return False
        return True 