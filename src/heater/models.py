"""Domain models and configuration for the thermal regulation system."""

from dataclasses import dataclass
from typing import Literal

SensorSource = Literal["gpu", "cpu"]
ControlMode = Literal["auto", "manual", "paused"]


@dataclass(frozen=True)
class ThermalReading:
    """Instantaneous hardware telemetry reading from GPU and CPU sensors."""

    gpu_temp_c: float
    cpu_temp_c: float
    gpu_power_w: float
    gpu_power_limit_w: float
    timestamp_sec: float


@dataclass(frozen=True)
class TelemetrySnapshot:
    """Processed telemetry snapshot including EMA and active load levels."""

    instant_gpu_temp_c: float
    ema_gpu_temp_c: float
    instant_cpu_temp_c: float
    ema_cpu_temp_c: float
    gpu_power_w: float
    gpu_power_limit_w: float
    target_temp_c: float
    duty_cycle: float
    active_sensor: SensorSource
    mode: ControlMode
    cpu_workers_enabled: bool
    gpu_worker_enabled: bool


@dataclass(frozen=True)
class PidCoefficients:
    """Proportional-Integral-Derivative tuning parameters for thermal control."""

    kp: float = 0.08
    ki: float = 0.015
    kd: float = 0.02
    hard_ceiling_c: float = 85.0
