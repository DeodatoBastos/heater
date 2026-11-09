"""Exponential Moving Average (EMA) filter and rolling history buffer."""

from collections import deque


class EmaFilter:
    """Computes an Exponential Moving Average over a stream of scalar values."""

    def __init__(self, smoothing_alpha: float = 0.2) -> None:
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError(
                f"Expected smoothing_alpha in (0.0, 1.0], got {smoothing_alpha}"
            )
        self._alpha: float = smoothing_alpha
        self._current_ema: float | None = None

    def update_estimate(self, sample_value: float) -> float:
        """Incorporates a new sample and returns the updated EMA.

        Example:
            >>> tracker = EmaFilter(0.5)
            >>> tracker.update_estimate(50.0)
            50.0
        """
        if self._current_ema is None:
            self._current_ema = sample_value
            return self._current_ema
        self._current_ema = (
            self._alpha * sample_value + (1.0 - self._alpha) * self._current_ema
        )
        return self._current_ema

    @property
    def current_value(self) -> float:
        """Returns the latest EMA value or 0.0 if uninitialized."""
        return self._current_ema if self._current_ema is not None else 0.0


class RollingMetricBuffer:
    """Fixed-capacity ring buffer for storing telemetry time-series."""

    def __init__(self, max_samples: int = 120) -> None:
        if max_samples < 2:
            raise ValueError(
                f"Expected max_samples >= 2, got {max_samples}"
            )
        self._samples: deque[float] = deque(maxlen=max_samples)

    def append_sample(self, value: float) -> None:
        """Appends a new scalar metric to the rolling buffer.

        Example:
            >>> buf = RollingMetricBuffer(10)
            >>> buf.append_sample(42.5)
        """
        self._samples.append(value)

    def recent_values(self, count: int) -> list[float]:
        """Returns up to the last `count` recorded samples in chronological order.

        Example:
            >>> buf = RollingMetricBuffer(5)
            >>> buf.append_sample(10.0)
            >>> buf.recent_values(3)
            [10.0]
        """
        if count <= 0:
            return []
        all_items = list(self._samples)
        return all_items[-count:]
