"""Hardware sensor abstractions and NVML/sysfs implementations."""

import ctypes
import glob
import time
from typing import Protocol

from models import ThermalReading


class ThermalSensorProbe(Protocol):
    """Interface for querying GPU and CPU thermal and power telemetry."""

    def read_telemetry(self) -> ThermalReading:
        """Returns an instantaneous snapshot of GPU/CPU temperatures and power."""
        ...

    def close_probe(self) -> None:
        """Releases any underlying hardware handles."""
        ...


def discover_cpu_temp_sysfs_path(
    sysfs_glob: str = "/sys/class/thermal/thermal_zone*",
) -> str | None:
    """Finds the best sysfs temperature file for CPU package temperature.

    Example:
        >>> path = discover_cpu_temp_sysfs_path()
    """
    fallback_path: str | None = None
    for zone_dir in sorted(glob.glob(sysfs_glob)):
        zone_type = _read_text_Line(f"{zone_dir}/type")
        temp_file = f"{zone_dir}/temp"
        if zone_type in ("x86_pkg_temp", "TCPU"):
            return temp_file
        if zone_type and fallback_path is None:
            fallback_path = temp_file
    return fallback_path


def _read_text_Line(file_path: str) -> str:
    """Reads and strips a single line from a sysfs file, returning empty on error."""
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def read_sysfs_millidegrees(file_path: str | None) -> float:
    """Reads a millidegree Celsius sysfs file and returns degrees Celsius.

    Example:
        >>> read_sysfs_millidegrees(None)
        0.0
    """
    if not file_path:
        return 0.0
    raw_text = _read_text_Line(file_path)
    if not raw_text:
        return 0.0
    try:
        return float(raw_text) / 1000.0
    except ValueError as exc:
        raise ValueError(
            f"Expected integer millidegrees in {file_path}, got {raw_text!r}"
        ) from exc


class NvmlLibraryWrapper:
    """Thin project-owned wrapper around libnvidia-ml.so.1."""

    def __init__(self, library_name: str = "libnvidia-ml.so.1") -> None:
        self._lib = ctypes.CDLL(library_name)
        self._handle = ctypes.c_void_p()
        self._init_device()

    def _init_device(self) -> None:
        """Initializes NVML and acquires handle for GPU index 0."""
        res = self._lib.nvmlInit_v2()
        if res != 0:
            raise RuntimeError(
                f"Expected nvmlInit_v2 return code 0, got {res}"
            )
        res = self._lib.nvmlDeviceGetHandleByIndex_v2(
            0, ctypes.byref(self._handle)
        )
        if res != 0:
            raise RuntimeError(
                f"Expected nvmlDeviceGetHandleByIndex_v2 code 0, got {res}"
            )

    def query_gpu_metrics(self) -> tuple[float, float, float]:
        """Returns (temp_c, power_w, power_limit_w) from NVML device 0.

        Example:
            >>> # nvml.query_gpu_metrics() -> (45.0, 18.5, 110.0)
        """
        temp_c = ctypes.c_uint()
        power_mw = ctypes.c_uint()
        limit_mw = ctypes.c_uint()
        self._lib.nvmlDeviceGetTemperature(
            self._handle, 0, ctypes.byref(temp_c)
        )
        self._lib.nvmlDeviceGetPowerUsage(self._handle, ctypes.byref(power_mw))
        self._lib.nvmlDeviceGetEnforcedPowerLimit(
            self._handle, ctypes.byref(limit_mw)
        )
        limit_w = limit_mw.value / 1000.0 if limit_mw.value > 0 else 110.0
        return float(temp_c.value), power_mw.value / 1000.0, limit_w

    def shutdown(self) -> None:
        """Shuts down the NVML library."""
        self._lib.nvmlShutdown()


class LinuxNvmlThermalProbe:
    """Combines NVML GPU telemetry and Linux sysfs CPU thermal telemetry."""

    def __init__(
        self,
        nvml_wrapper: NvmlLibraryWrapper,
        cpu_temp_path: str | None,
    ) -> None:
        self._nvml = nvml_wrapper
        self._cpu_path = cpu_temp_path

    def read_telemetry(self) -> ThermalReading:
        """Reads current GPU and CPU telemetry into a ThermalReading.

        Example:
            >>> # probe.read_telemetry()
        """
        gpu_temp, gpu_power, gpu_limit = self._nvml.query_gpu_metrics()
        cpu_temp = read_sysfs_millidegrees(self._cpu_path)
        if cpu_temp <= 0.0:
            cpu_temp = gpu_temp
        return ThermalReading(
            gpu_temp_c=gpu_temp,
            cpu_temp_c=cpu_temp,
            gpu_power_w=gpu_power,
            gpu_power_limit_w=gpu_limit,
            timestamp_sec=time.monotonic(),
        )

    def close_probe(self) -> None:
        """Releases NVML resources."""
        self._nvml.shutdown()
