//! Closed-loop PID thermal regulator with anti-windup and safety cutoff.

use crate::models::PidCoefficients;

/// Clamps a raw duty cycle value into the valid `[0.0, 1.0]` interval.
///
/// # Examples
/// ```
/// use heater::pid_regulator::clamp_duty_cycle;
/// assert_eq!(clamp_duty_cycle(1.4), 1.0);
/// assert_eq!(clamp_duty_cycle(-0.2), 0.0);
/// ```
pub fn clamp_duty_cycle(raw_duty: f64) -> f64 {
    raw_duty.clamp(0.0, 1.0)
}

/// Regulates thermal load duty cycle `[0.0, 1.0]` to reach a target temperature.
#[derive(Debug, Clone)]
pub struct PidThermalRegulator {
    coeffs: PidCoefficients,
    integral_term: f64,
    previous_error: Option<f64>,
}

impl PidThermalRegulator {
    /// Creates a new `PidThermalRegulator` with the given PID coefficients.
    ///
    /// # Examples
    /// ```
    /// use heater::models::PidCoefficients;
    /// use heater::pid_regulator::PidThermalRegulator;
    /// let reg = PidThermalRegulator::new(PidCoefficients::default());
    /// ```
    pub fn new(coeffs: PidCoefficients) -> Self {
        Self {
            coeffs,
            integral_term: 0.0,
            previous_error: None,
        }
    }

    /// Calculates the next duty cycle `[0.0, 1.0]` given current and target temps.
    ///
    /// # Examples
    /// ```
    /// use heater::models::PidCoefficients;
    /// use heater::pid_regulator::PidThermalRegulator;
    /// let mut reg = PidThermalRegulator::new(PidCoefficients::default());
    /// let duty = reg.compute_duty_cycle(50.0, 65.0, 0.25).unwrap();
    /// assert!(duty > 0.5);
    /// ```
    pub fn compute_duty_cycle(
        &mut self,
        current_temp_c: f64,
        target_temp_c: f64,
        dt_seconds: f64,
    ) -> Result<f64, String> {
        if dt_seconds <= 0.0 {
            return Err(format!(
                "Expected positive dt_seconds > 0.0, got {dt_seconds}"
            ));
        }
        if current_temp_c >= self.coeffs.hard_ceiling_c {
            self.reset_state();
            return Ok(0.0);
        }
        let error = target_temp_c - current_temp_c;
        Ok(self.step_pid(error, dt_seconds))
    }

    fn step_pid(&mut self, error: f64, dt_seconds: f64) -> f64 {
        let candidate_integral = self.integral_term + error * dt_seconds * self.coeffs.ki;
        self.integral_term = clamp_duty_cycle(candidate_integral);
        let derivative = match self.previous_error {
            Some(prev) => (error - prev) / dt_seconds,
            None => 0.0,
        };
        self.previous_error = Some(error);
        let raw_output = self.coeffs.kp * error + self.integral_term + self.coeffs.kd * derivative;
        clamp_duty_cycle(raw_output)
    }

    /// Clears accumulated integral and derivative state.
    ///
    /// # Examples
    /// ```
    /// use heater::models::PidCoefficients;
    /// use heater::pid_regulator::PidThermalRegulator;
    /// let mut reg = PidThermalRegulator::new(PidCoefficients::default());
    /// reg.reset_state();
    /// ```
    pub fn reset_state(&mut self) {
        self.integral_term = 0.0;
        self.previous_error = None;
    }
}
