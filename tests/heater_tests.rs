//! Integration and unit tests using named fake classes for sensor and actuator I/O.

use heater::ema_tracker::{EmaFilter, RollingMetricBuffer};
use heater::load_workers::{ThermalLoadActuator, compute_slice_timings};
use heater::models::{ControlMode, PidCoefficients, SensorSource, ThermalReading};
use heater::orchestrator::ThermalOrchestrator;
use heater::pid_regulator::{PidThermalRegulator, clamp_duty_cycle};
use heater::thermal_sensors::ThermalSensorProbe;

/// Named fake sensor probe returning configured readings without hardware access.
pub struct FakeThermalSensorProbe {
    pub current_reading: ThermalReading,
    pub closed: bool,
}

impl FakeThermalSensorProbe {
    pub fn new(reading: ThermalReading) -> Self {
        Self {
            current_reading: reading,
            closed: false,
        }
    }
}

impl ThermalSensorProbe for FakeThermalSensorProbe {
    fn read_telemetry(&mut self) -> Result<ThermalReading, String> {
        Ok(self.current_reading)
    }

    fn close_probe(&mut self) {
        self.closed = true;
    }
}

/// Named fake load actuator recording applied duty cycles and enable flags.
#[derive(Default)]
pub struct FakeThermalLoadActuator {
    pub last_duty_cycle: f64,
    pub last_gpu_enabled: bool,
    pub last_cpu_enabled: bool,
    pub was_shutdown: bool,
}

impl ThermalLoadActuator for FakeThermalLoadActuator {
    fn apply_duty_cycle(&mut self, duty_cycle: f64, enable_gpu: bool, enable_cpu: bool) {
        self.last_duty_cycle = duty_cycle;
        self.last_gpu_enabled = enable_gpu;
        self.last_cpu_enabled = enable_cpu;
    }

    fn shutdown_workers(&mut self) {
        self.was_shutdown = true;
    }
}

#[test]
fn ema_filter_smooths_and_validates_alpha() {
    assert!(EmaFilter::new(0.0).is_err());
    assert!(EmaFilter::new(1.2).is_err());
    let mut filter = EmaFilter::new(0.5).unwrap();
    assert_eq!(filter.update_estimate(40.0), 40.0);
    assert_eq!(filter.update_estimate(60.0), 50.0);
}

#[test]
fn rolling_buffer_maintains_capacity_and_summary() {
    let mut buf = RollingMetricBuffer::new(3).unwrap();
    for val in [30.0, 40.0, 50.0, 60.0] {
        buf.append_sample(val);
    }
    assert_eq!(buf.summary_stats(), (40.0, 60.0, 60.0));
    assert_eq!(buf.as_chart_points().len(), 3);
}

#[test]
fn pid_regulator_increases_duty_and_cuts_off_at_ceiling() {
    let mut reg = PidThermalRegulator::new(PidCoefficients::default());
    let duty = reg.compute_duty_cycle(50.0, 65.0, 0.25).unwrap();
    assert!(duty > 0.5 && duty <= 1.0);
    let ceiling_duty = reg.compute_duty_cycle(85.5, 90.0, 0.25).unwrap();
    assert_eq!(ceiling_duty, 0.0);
    assert_eq!(clamp_duty_cycle(1.5), 1.0);
}

#[test]
fn orchestrator_steps_and_handles_mode_toggles() {
    let reading = ThermalReading {
        gpu_temp_c: 45.0,
        cpu_temp_c: 55.0,
        gpu_power_w: 18.0,
        gpu_power_limit_w: 110.0,
        elapsed_dt_sec: 0.25,
    };
    let probe = FakeThermalSensorProbe::new(reading);
    let actuator = FakeThermalLoadActuator::default();
    let regulator = PidThermalRegulator::new(PidCoefficients::default());
    let mut orch = ThermalOrchestrator::new(probe, regulator, actuator, 65.0, 0.5).unwrap();
    let snap = orch.step_cycle().unwrap();
    assert_eq!(snap.instant_gpu_temp_c, 45.0);
    assert!(snap.duty_cycle > 0.0);
    assert_eq!(orch.toggle_pause_mode(), ControlMode::Paused);
    let snap_paused = orch.step_cycle().unwrap();
    assert_eq!(snap_paused.duty_cycle, 0.0);
    assert_eq!(orch.toggle_active_sensor(), SensorSource::Cpu);
    assert!(!orch.toggle_gpu_heating());
    assert!(!orch.toggle_cpu_heating());
    orch.shutdown();
}

#[test]
fn slice_timings_splits_window_accurately() {
    let (busy, sleep) = compute_slice_timings(0.25, 0.1).unwrap();
    assert!((busy - 0.025).abs() < 1e-6);
    assert!((sleep - 0.075).abs() < 1e-6);
    assert!(compute_slice_timings(-0.1, 0.1).is_err());
}
