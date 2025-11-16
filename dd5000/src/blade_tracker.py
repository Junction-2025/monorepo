"""
Blade Count Tracker - Maintains a robust estimate of blade count from KNN detections.

This module provides a stateful tracker that accumulates blade count observations
and provides a confidence-based estimate for RPM calculations.
"""

import numpy as np
from collections import deque
from typing import Any
from src.logger import get_logger

logger = get_logger()


class BladeCountTracker:
    """
    Tracks blade count observations over time and provides a robust estimate.

    Uses a sliding window of observations and statistical methods to filter
    out noise and provide a stable blade count for RPM estimation.
    """

    def __init__(
        self,
        window_size: int = 10,
        min_observations: int = 3,
        default_blade_count: int = 3
    ):
        """
        Initialize the blade count tracker.

        Args:
            window_size: Number of recent observations to keep
            min_observations: Minimum observations needed before trusting the estimate
            default_blade_count: Fallback blade count when no observations exist
        """
        self.window_size = window_size
        self.min_observations = min_observations
        self.default_blade_count = default_blade_count

        # Sliding window of blade count observations
        self.observations: deque[int] = deque(maxlen=window_size)

        # Last confirmed blade count
        self._current_estimate = default_blade_count

    def add_observation(self, blade_count_array: list[tuple[int, Any]]) -> None:
        """
        Add a new blade count observation from KNN detection.

        Args:
            blade_count_array: List of (blade_count, mask) tuples from get_blade_count()
        """
        if not blade_count_array:
            logger.debug("No blade count data to add")
            return

        # Extract blade counts from the array (ignore masks)
        blade_counts = [count for count, _ in blade_count_array if count > 0]

        if not blade_counts:
            logger.debug("No valid blade counts (all zero or negative)")
            return

        # Use the average blade count across all detected propellers
        avg_blade_count = int(np.round(np.mean(blade_counts)))

        self.observations.append(avg_blade_count)
        logger.info(f"Added blade count observation: {avg_blade_count}, window size: {len(self.observations)}")

        # Update current estimate if we have enough observations
        if len(self.observations) >= self.min_observations:
            self._update_estimate()

    def _update_estimate(self) -> None:
        """
        Update the current blade count estimate using statistical methods.

        Uses the mode (most common value) for robustness against outliers.
        Falls back to median if no clear mode exists.
        """
        if not self.observations:
            return

        # Convert to numpy array for easier statistics
        obs_array = np.array(list(self.observations))

        # Find the mode (most common value)
        values, counts = np.unique(obs_array, return_counts=True)
        mode_idx = np.argmax(counts)
        mode_value = int(values[mode_idx])
        mode_count = counts[mode_idx]

        # Use mode if it appears frequently enough (>40% of observations)
        if mode_count / len(obs_array) > 0.4:
            new_estimate = mode_value
        else:
            # Fall back to median for stability
            new_estimate = int(np.median(obs_array))

        if new_estimate != self._current_estimate:
            logger.info(
                f"Blade count estimate updated: {self._current_estimate} -> {new_estimate} "
                f"(based on {len(self.observations)} observations)"
            )
            self._current_estimate = new_estimate

    def get_blade_count(self) -> int:
        """
        Get the current best estimate of blade count.

        Returns:
            The current blade count estimate, or default if insufficient data
        """
        return self._current_estimate

    def has_confident_estimate(self) -> bool:
        """
        Check if we have enough observations to trust the estimate.

        Returns:
            True if we have at least min_observations observations
        """
        return len(self.observations) >= self.min_observations

    def reset(self) -> None:
        """Reset the tracker to initial state."""
        self.observations.clear()
        self._current_estimate = self.default_blade_count
        logger.info("Blade count tracker reset")

    def get_stats(self) -> dict:
        """
        Get statistics about current observations.

        Returns:
            Dictionary with observation statistics
        """
        if not self.observations:
            return {
                "count": 0,
                "current_estimate": self._current_estimate,
                "is_default": True,
                "is_confident": False
            }

        obs_array = np.array(list(self.observations))
        return {
            "count": len(self.observations),
            "current_estimate": self._current_estimate,
            "mean": float(np.mean(obs_array)),
            "median": float(np.median(obs_array)),
            "std": float(np.std(obs_array)),
            "min": int(np.min(obs_array)),
            "max": int(np.max(obs_array)),
            "is_default": False,
            "is_confident": self.has_confident_estimate()
        }
