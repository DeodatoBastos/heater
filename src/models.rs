//! Domain models and configuration types for the PC thermal regulator.

/// Primary temperature sensor tracked by the PID regulator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensorSource {
    Gpu,
    Cpu,
}

impl SensorSource {
    /// Returns a human-readable uppercase label for the sensor source.
    ///
    /// # Examples
    /// ```
    /// use heater::models::SensorSource;
    /// assert_eq!(SensorSource::Gpu.label(), "GPU");
    /// ```
    pub fn label(self) -> &'static str {
        match self {
            Self::Gpu => "GPU",
            Self::Cpu => "CPU",
        }
    }

    /// Toggles between GPU and CPU sensor sources.
    ///
    /// # Examples
    /// ```
    /// use heater::models::SensorSource;
    /// assert_eq!(SensorSource::Gpu.toggled(), SensorSource::Cpu);
    /// ```
    pub fn toggled(self) -> Self {
        match self {
            Self::Gpu => Self::Cpu,
            Self::Cpu => Self::Gpu,
        }
    }
}

/// Operating mode of the thermal controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlMode {
    Auto,
    Manual,
    Paused,
}

impl ControlMode {
    /// Returns a human-readable uppercase label for the control mode.
    ///
    /// # Examples
    /// ```
    /// use heater::models::ControlMode;
    /// assert_eq!(ControlMode::Auto.label(), "AUTO");
    /// ```
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "AUTO",
            Self::Manual => "MANUAL",
            Self::Paused => "PAUSED",
        }
    }

    /// Advances to the next control mode in sequence (Auto -> Manual -> Paused -> Auto).
    ///
    /// # Examples
    /// ```
    /// use heater::models::ControlMode;
    /// assert_eq!(ControlMode::Auto.cycled(), ControlMode::Manual);
    /// ```
    pub fn cycled(self) -> Self {
        match self {
            Self::Auto => Self::Manual,
            Self::Manual => Self::Paused,
            Self::Paused => Self::Auto,
        }
    }
}

/// Instantaneous hardware telemetry reading from GPU and CPU sensors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermalReading {
    pub gpu_temp_c: f64,
    pub cpu_temp_c: f64,
    pub gpu_power_w: f64,
    pub gpu_power_limit_w: f64,
    pub elapsed_dt_sec: f64,
}

/// Processed telemetry snapshot including EMA and active load levels for UI rendering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TelemetrySnapshot {
    pub instant_gpu_temp_c: f64,
    pub ema_gpu_temp_c: f64,
    pub instant_cpu_temp_c: f64,
    pub ema_cpu_temp_c: f64,
    pub gpu_power_w: f64,
    pub gpu_power_limit_w: f64,
    pub target_temp_c: f64,
    pub duty_cycle: f64,
    pub active_sensor: SensorSource,
    pub mode: ControlMode,
    pub cpu_workers_enabled: bool,
    pub gpu_worker_enabled: bool,
}

/// Proportional-Integral-Derivative tuning parameters for thermal control.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PidCoefficients {
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
    pub hard_ceiling_c: f64,
}

impl Default for PidCoefficients {
    fn default() -> Self {
        Self {
            kp: 0.08,
            ki: 0.015,
            kd: 0.02,
            hard_ceiling_c: 85.0,
        }
    }
}
