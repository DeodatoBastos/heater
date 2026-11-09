"""Closed-loop PID thermal regulator with anti-windup and safety cutoff."""

from heater.models import PidCoefficients


def clamp_duty_cycle(raw_duty: float) -> float:
    """Clamps a raw duty value into the valid [0.0, 1.0] interval.

    Example:
        >>> clamp_duty_cycle(1.4)
        1.0
    """
    return max(0.0, min(1.0, raw_duty))


class PidThermalRegulator:
    """Regulates thermal load duty cycle [0.0, 1.0] to reach a target temperature."""

    def __init__(self, coefficients: PidCoefficients) -> None:
        self._coeffs: PidCoefficients = coefficients
        self._integral_term: float = 0.0
        self._previous_error: float | None = None

    def compute_duty_cycle(
        self,
        current_temp_c: float,
        target_temp_c: float,
        dt_seconds: float,
    ) -> float:
        """Calculates the next duty cycle [0.0, 1.0] given current and target temps.

        Example:
            >>> reg = PidThermalRegulator(PidCoefficients())
            >>> reg.compute_duty_cycle(50.0, 65.0, 0.5) > 0.0
            True
        """
        if dt_seconds <= 0.0:
            raise ValueError(
                f"Expected positive dt_seconds > 0.0, got {dt_seconds}"
            )
        if current_temp_c >= self._coeffs.hard_ceiling_c:
            self.reset_state()
            return 0.0
        error = target_temp_c - current_temp_c
        return self._step_pid(error, dt_seconds)

    def _step_pid(self, error: float, dt_seconds: float) -> float:
        """Updates internal PID terms with anti-windup and returns clamped duty."""
        candidate_integral = self._integral_term + error * dt_seconds * self._coeffs.ki
        self._integral_term = clamp_duty_cycle(candidate_integral)
        derivative = 0.0
        if self._previous_error is not None:
            derivative = (error - self._previous_error) / dt_seconds
        self._previous_error = error
        raw_output = (
            self._coeffs.kp * error
            + self._integral_term
            + self._coeffs.kd * derivative
        )
        return clamp_duty_cycle(raw_output)

    def reset_state(self) -> None:
        """Clears accumulated integral and derivative state.

        Example:
            >>> reg = PidThermalRegulator(PidCoefficients())
            >>> reg.reset_state()
        """
        self._integral_term = 0.0
        self._previous_error = None
