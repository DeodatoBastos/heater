"""Named fake classes for unit testing without hardware or filesystem I/O."""

from heater.models import ThermalReading


class FakeThermalSensorProbe:
    """In-memory fake sensor probe returning queued or static ThermalReadings."""

    def __init__(self, initial_reading: ThermalReading) -> None:
        self.current_reading: ThermalReading = initial_reading
        self.closed: bool = False

    def read_telemetry(self) -> ThermalReading:
        """Returns the configured fake ThermalReading."""
        return self.current_reading

    def close_probe(self) -> None:
        """Marks the fake probe as closed."""
        self.closed = True


class FakeThermalLoadActuator:
    """In-memory fake load actuator recording applied duty cycles and flags."""

    def __init__(self) -> None:
        self.last_duty_cycle: float = 0.0
        self.last_gpu_enabled: bool = False
        self.last_cpu_enabled: bool = False
        self.was_shutdown: bool = False

    def apply_duty_cycle(
        self,
        duty_cycle: float,
        enable_gpu: bool,
        enable_cpu: bool,
    ) -> None:
        """Records the requested duty cycle and hardware enable flags."""
        self.last_duty_cycle = duty_cycle
        self.last_gpu_enabled = enable_gpu
        self.last_cpu_enabled = enable_cpu

    def shutdown_workers(self) -> None:
        """Marks the fake actuator as shut down."""
        self.was_shutdown = True
