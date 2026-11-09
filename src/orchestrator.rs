//! Orchestrates sensors, EMA filters, PID regulator, and load actuators.

use crate::ema_tracker::{EmaFilter, RollingMetricBuffer};
use crate::load_workers::ThermalLoadActuator;
use crate::models::{ControlMode, SensorSource, TelemetrySnapshot, ThermalReading};
use crate::pid_regulator::{PidThermalRegulator, clamp_duty_cycle};
use crate::thermal_sensors::ThermalSensorProbe;

/// Coordinates telemetry reading, EMA smoothing, PID control, and load workers.
pub struct ThermalOrchestrator<P: ThermalSensorProbe, A: ThermalLoadActuator> {
    probe: P,
    regulator: PidThermalRegulator,
    actuator: A,
    target_temp_c: f64,
    mode: ControlMode,
    manual_duty: f64,
    active_sensor: SensorSource,
    gpu_enabled: bool,
    cpu_enabled: bool,
    gpu_ema: EmaFilter,
    cpu_ema: EmaFilter,
    pub instant_history: RollingMetricBuffer,
    pub ema_history: RollingMetricBuffer,
    pub target_history: RollingMetricBuffer,
    pub power_history: RollingMetricBuffer,
}

impl<P: ThermalSensorProbe, A: ThermalLoadActuator> ThermalOrchestrator<P, A> {
    /// Creates a new `ThermalOrchestrator` with injected probe, regulator, and actuator.
    pub fn new(
        probe: P,
        regulator: PidThermalRegulator,
        actuator: A,
        initial_target_c: f64,
        ema_alpha: f64,
    ) -> Result<Self, String> {
        Ok(Self {
            probe,
            regulator,
            actuator,
            target_temp_c: initial_target_c.clamp(30.0, 82.0),
            mode: ControlMode::Auto,
            manual_duty: 0.0,
            active_sensor: SensorSource::Gpu,
            gpu_enabled: true,
            cpu_enabled: true,
            gpu_ema: EmaFilter::new(ema_alpha)?,
            cpu_ema: EmaFilter::new(ema_alpha)?,
            instant_history: RollingMetricBuffer::new(120)?,
            ema_history: RollingMetricBuffer::new(120)?,
            target_history: RollingMetricBuffer::new(120)?,
            power_history: RollingMetricBuffer::new(120)?,
        })
    }

    /// Executes one control loop tick: read sensors, update EMA/PID, and apply load.
    pub fn step_cycle(&mut self) -> Result<TelemetrySnapshot, String> {
        let reading = self.probe.read_telemetry()?;
        let ema_gpu = self.gpu_ema.update_estimate(reading.gpu_temp_c);
        let ema_cpu = self.cpu_ema.update_estimate(reading.cpu_temp_c);
        let (instant_tracked, ema_tracked) = match self.active_sensor {
            SensorSource::Gpu => (reading.gpu_temp_c, ema_gpu),
            SensorSource::Cpu => (reading.cpu_temp_c, ema_cpu),
        };
        let duty = self.determine_duty(ema_tracked, reading.elapsed_dt_sec)?;
        self.actuator
            .apply_duty_cycle(duty, self.gpu_enabled, self.cpu_enabled);
        self.record_history(instant_tracked, ema_tracked, reading.gpu_power_w);
        Ok(self.build_snapshot(&reading, ema_gpu, ema_cpu, duty))
    }

    fn determine_duty(&mut self, tracked_temp_c: f64, dt_sec: f64) -> Result<f64, String> {
        match self.mode {
            ControlMode::Paused => Ok(0.0),
            ControlMode::Manual => {
                if tracked_temp_c >= 85.0 {
                    Ok(0.0)
                } else {
                    Ok(self.manual_duty)
                }
            }
            ControlMode::Auto => {
                self.regulator
                    .compute_duty_cycle(tracked_temp_c, self.target_temp_c, dt_sec)
            }
        }
    }

    fn record_history(&mut self, instant_c: f64, ema_c: f64, power_w: f64) {
        self.instant_history.append_sample(instant_c);
        self.ema_history.append_sample(ema_c);
        self.target_history.append_sample(self.target_temp_c);
        self.power_history.append_sample(power_w);
    }

    fn build_snapshot(
        &self,
        reading: &ThermalReading,
        ema_gpu: f64,
        ema_cpu: f64,
        duty: f64,
    ) -> TelemetrySnapshot {
        TelemetrySnapshot {
            instant_gpu_temp_c: reading.gpu_temp_c,
            ema_gpu_temp_c: ema_gpu,
            instant_cpu_temp_c: reading.cpu_temp_c,
            ema_cpu_temp_c: ema_cpu,
            gpu_power_w: reading.gpu_power_w,
            gpu_power_limit_w: reading.gpu_power_limit_w,
            target_temp_c: self.target_temp_c,
            duty_cycle: duty,
            active_sensor: self.active_sensor,
            mode: self.mode,
            cpu_workers_enabled: self.cpu_enabled,
            gpu_worker_enabled: self.gpu_enabled,
        }
    }

    /// Shifts target temperature by `delta_c` within safe bounds `[30.0, 82.0]` C.
    pub fn adjust_target_temperature(&mut self, delta_c: f64) -> f64 {
        self.target_temp_c = (self.target_temp_c + delta_c).clamp(30.0, 82.0);
        self.target_temp_c
    }

    /// Toggles between `Paused` and `Auto` thermal regulation modes.
    pub fn toggle_pause_mode(&mut self) -> ControlMode {
        self.mode = match self.mode {
            ControlMode::Paused => ControlMode::Auto,
            _ => ControlMode::Paused,
        };
        self.regulator.reset_state();
        self.mode
    }

    /// Cycles control mode through `Auto -> Manual -> Paused -> Auto`.
    pub fn cycle_control_mode(&mut self) -> ControlMode {
        self.mode = self.mode.cycled();
        self.regulator.reset_state();
        self.mode
    }

    /// Switches to `Manual` mode and adjusts manual duty cycle by `delta_duty`.
    pub fn adjust_manual_duty(&mut self, delta_duty: f64) -> f64 {
        self.mode = ControlMode::Manual;
        self.manual_duty = clamp_duty_cycle(self.manual_duty + delta_duty);
        self.manual_duty
    }

    /// Switches primary regulated sensor between `Gpu` and `Cpu`.
    pub fn toggle_active_sensor(&mut self) -> SensorSource {
        self.active_sensor = self.active_sensor.toggled();
        self.regulator.reset_state();
        self.active_sensor
    }

    /// Enables or disables GPU compute worker heat generation.
    pub fn toggle_gpu_heating(&mut self) -> bool {
        self.gpu_enabled = !self.gpu_enabled;
        self.gpu_enabled
    }

    /// Enables or disables CPU compute worker heat generation.
    pub fn toggle_cpu_heating(&mut self) -> bool {
        self.cpu_enabled = !self.cpu_enabled;
        self.cpu_enabled
    }

    /// Stops all actuators and closes hardware sensor probes.
    pub fn shutdown(&mut self) {
        self.actuator.shutdown_workers();
        self.probe.close_probe();
    }
}
