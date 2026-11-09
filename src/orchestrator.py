"""Orchestrates sensors, EMA filters, PID regulator, and load actuators."""

from ema_tracker import EmaFilter, RollingMetricBuffer
from load_workers import ThermalLoadActuator
from models import ControlMode, SensorSource, TelemetrySnapshot, ThermalReading
from pid_regulator import PidThermalRegulator, clamp_duty_cycle
from thermal_sensors import ThermalSensorProbe


class ThermalOrchestrator:
    """Coordinates telemetry reading, EMA smoothing, PID control, and load workers."""

    def __init__(
        self,
        sensor_probe: ThermalSensorProbe,
        regulator: PidThermalRegulator,
        actuator: ThermalLoadActuator,
        initial_target_c: float = 65.0,
        ema_alpha: float = 0.2,
    ) -> None:
        self._probe = sensor_probe
        self._regulator = regulator
        self._actuator = actuator
        self._target_temp_c: float = initial_target_c
        self._mode: ControlMode = "auto"
        self._manual_duty: float = 0.0
        self._active_sensor: SensorSource = "gpu"
        self._gpu_enabled: bool = True
        self._cpu_enabled: bool = True
        self._gpu_ema = EmaFilter(ema_alpha)
        self._cpu_ema = EmaFilter(ema_alpha)
        self._last_timestamp: float | None = None
        self.temp_history = RollingMetricBuffer(120)
        self.ema_history = RollingMetricBuffer(120)
        self.power_history = RollingMetricBuffer(120)

    def step_cycle(self) -> TelemetrySnapshot:
        """Executes one control loop tick: read sensors, update EMA/PID, apply load.

        Example:
            >>> # snapshot = orchestrator.step_cycle()
        """
        reading = self._probe.read_telemetry()
        dt_sec = self._compute_dt(reading.timestamp_sec)
        ema_gpu = self._gpu_ema.update_estimate(reading.gpu_temp_c)
        ema_cpu = self._cpu_ema.update_estimate(reading.cpu_temp_c)
        tracked_temp = ema_gpu if self._active_sensor == "gpu" else ema_cpu
        instant_tracked = (
            reading.gpu_temp_c if self._active_sensor == "gpu" else reading.cpu_temp_c
        )
        duty = self._determine_duty(tracked_temp, dt_sec)
        self._actuator.apply_duty_cycle(duty, self._gpu_enabled, self._cpu_enabled)
        self._record_history(instant_tracked, tracked_temp, reading.gpu_power_w)
        return self._build_snapshot(reading, ema_gpu, ema_cpu, duty)

    def _compute_dt(self, now_sec: float) -> float:
        """Computes elapsed time since last tick, defaulting to 0.25s on first tick."""
        if self._last_timestamp is None:
            self._last_timestamp = now_sec
            return 0.25
        dt = max(0.01, now_sec - self._last_timestamp)
        self._last_timestamp = now_sec
        return dt

    def _determine_duty(self, tracked_temp_c: float, dt_sec: float) -> float:
        """Calculates duty cycle based on active control mode and safety ceiling."""
        if self._mode == "paused":
            return 0.0
        if self._mode == "manual":
            if tracked_temp_c >= 85.0:
                return 0.0
            return self._manual_duty
        return self._regulator.compute_duty_cycle(
            tracked_temp_c, self._target_temp_c, dt_sec
        )

    def _record_history(
        self, instant_c: float, ema_c: float, power_w: float
    ) -> None:
        """Appends current telemetry to rolling history buffers."""
        self.temp_history.append_sample(instant_c)
        self.ema_history.append_sample(ema_c)
        self.power_history.append_sample(power_w)

    def _build_snapshot(
        self,
        reading: ThermalReading,
        ema_gpu: float,
        ema_cpu: float,
        duty: float,
    ) -> TelemetrySnapshot:
        """Constructs an immutable TelemetrySnapshot for UI rendering."""
        return TelemetrySnapshot(
            instant_gpu_temp_c=reading.gpu_temp_c,
            ema_gpu_temp_c=ema_gpu,
            instant_cpu_temp_c=reading.cpu_temp_c,
            ema_cpu_temp_c=ema_cpu,
            gpu_power_w=reading.gpu_power_w,
            gpu_power_limit_w=reading.gpu_power_limit_w,
            target_temp_c=self._target_temp_c,
            duty_cycle=duty,
            active_sensor=self._active_sensor,
            mode=self._mode,
            cpu_workers_enabled=self._cpu_enabled,
            gpu_worker_enabled=self._gpu_enabled,
        )

    def adjust_target_temperature(self, delta_c: float) -> float:
        """Shifts target temperature by delta_c within safe bounds [30.0, 82.0] C.

        Example:
            >>> # orchestrator.adjust_target_temperature(1.0)
        """
        self._target_temp_c = max(30.0, min(82.0, self._target_temp_c + delta_c))
        return self._target_temp_c

    def toggle_pause_mode(self) -> ControlMode:
        """Toggles between 'paused' and 'auto' thermal regulation modes."""
        if self._mode == "paused":
            self._mode = "auto"
            self._regulator.reset_state()
        else:
            self._mode = "paused"
        return self._mode

    def cycle_control_mode(self) -> ControlMode:
        """Cycles mode through auto -> manual -> paused -> auto."""
        order: tuple[ControlMode, ...] = ("auto", "manual", "paused")
        idx = (order.index(self._mode) + 1) % len(order)
        self._mode = order[idx]
        self._regulator.reset_state()
        return self._mode

    def adjust_manual_duty(self, delta_duty: float) -> float:
        """Switches to manual mode and adjusts manual duty cycle by delta_duty."""
        self._mode = "manual"
        self._manual_duty = clamp_duty_cycle(self._manual_duty + delta_duty)
        return self._manual_duty

    def toggle_active_sensor(self) -> SensorSource:
        """Switches primary regulated sensor between 'gpu' and 'cpu'."""
        self._active_sensor = "cpu" if self._active_sensor == "gpu" else "gpu"
        self._regulator.reset_state()
        return self._active_sensor

    def toggle_gpu_heating(self) -> bool:
        """Enables or disables GPU compute worker heat generation."""
        self._gpu_enabled = not self._gpu_enabled
        return self._gpu_enabled

    def toggle_cpu_heating(self) -> bool:
        """Enables or disables CPU compute worker heat generation."""
        self._cpu_enabled = not self._cpu_enabled
        return self._cpu_enabled

    def shutdown(self) -> None:
        """Stops all actuators and closes hardware sensor probes."""
        self._actuator.shutdown_workers()
        self._probe.close_probe()
