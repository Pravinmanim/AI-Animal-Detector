"""
TemporalConfirmation — rolling window confirmation logic with
probability-based confidence tracking and exponential decay weighting.

Maintains a fixed-size deque of the last N analyzed frames' max target
confidence scores. Uses exponential decay weights so that recent frames
contribute more than older ones. An alert is only eligible when the
weighted sum exceeds the `required_hits` threshold.

This module has zero dependencies on camera, model, or I/O —
it is fully unit-testable in isolation.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List


@dataclass
class TemporalConfirmation:
    """
    Rolling-window temporal confirmation gate with exponential decay.

    Args:
        window_size:    Number of most-recent frames to consider (default 5).
        required_hits:  Minimum weighted sum to confirm an alert (default 2.5).
        decay_factor:   Exponential decay base — higher = more weight on recent
                        frames (default 1.5). A value of 1.0 means equal weights.
    """
    window_size: int = 5
    required_hits: float = 0.45
    decay_factor: float = 1.5
    _window: Deque[float] = field(default_factory=deque, init=False, repr=False)
    _weights: List[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if self.required_hits <= 0:
            raise ValueError("required_hits must be > 0")
        self._window = deque(maxlen=self.window_size)
        self._weights = self._compute_weights()

    def _compute_weights(self) -> List[float]:
        """
        Compute normalized exponential decay weights for the window.

        With decay_factor=1.5 and window_size=5:
          raw   = [1.0, 1.5, 2.25, 3.375, 5.0625]
          total = 13.1875
          normed = [0.076, 0.114, 0.171, 0.256, 0.384]

        The most recent frame (last element) gets the highest weight.
        All weights sum to 1.0.
        """
        raw = [self.decay_factor ** i for i in range(self.window_size)]
        total = sum(raw)
        return [w / total for w in raw]

    def record(self, confidence: float) -> None:
        """Push a frame's max target confidence into the rolling window."""
        self._window.append(confidence)

    @property
    def weighted_sum(self) -> float:
        """
        Compute the weighted sum of confidences in the current window.

        Shorter windows (during initial fill-up) are weighted proportionally
        using only the most-recent weights.
        """
        n = len(self._window)
        if n == 0:
            return 0.0
        # Use the last n weights (most recent) when window is not full
        weights = self._weights[-n:]
        return sum(c * w for c, w in zip(self._window, weights))

    def is_confirmed(self) -> bool:
        """
        Return True iff the weighted sum meets or exceeds `required_hits`.
        Returns False if the window hasn't been filled yet.
        """
        if len(self._window) < self.window_size:
            return False
        return self.weighted_sum >= self.required_hits

    @property
    def hit_count(self) -> int:
        """Number of positive detections currently in the window."""
        return sum(1 for c in self._window if c > 0)

    @property
    def total_count(self) -> int:
        """Total frames currently in the window."""
        return len(self._window)

    def summary(self) -> str:
        """Human-readable confirmation ratio, e.g. '0.64/2.5'."""
        return f"{self.weighted_sum:.2f}/{self.required_hits}"

    def reset(self) -> None:
        """Clear the window (e.g., after camera restart or class change)."""
        self._window.clear()
