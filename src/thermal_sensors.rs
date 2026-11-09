//! Hardware sensor trait and NVML (`libnvidia-ml.so.1`) + Linux `sysfs` implementations.

use std::ffi::c_void;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use libloading::{Library, Symbol};

use crate::models::ThermalReading;

/// Interface for querying GPU and CPU thermal and power telemetry.
pub trait ThermalSensorProbe {
    /// Returns an instantaneous snapshot of GPU/CPU temperatures and power.
    fn read_telemetry(&mut self) -> Result<ThermalReading, String>;

    /// Releases any underlying hardware handles.
    fn close_probe(&mut self);
}

/// Discovers the best Linux `sysfs` thermal zone file for CPU package temperature.
///
/// # Examples
/// ```
/// use heater::thermal_sensors::discover_cpu_temp_sysfs_path;
/// let _path = discover_cpu_temp_sysfs_path("/sys/class/thermal");
/// ```
pub fn discover_cpu_temp_sysfs_path(thermal_root: &str) -> Option<PathBuf> {
    let entries = fs::read_dir(thermal_root).ok()?;
    let mut fallback: Option<PathBuf> = None;
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if !name.starts_with("thermal_zone") {
            continue;
        }
        let zone_type = fs::read_to_string(path.join("type"))
            .unwrap_or_default()
            .trim()
            .to_string();
        let temp_path = path.join("temp");
        if zone_type == "x86_pkg_temp" || zone_type == "TCPU" {
            return Some(temp_path);
        }
        if !zone_type.is_empty() && fallback.is_none() {
            fallback = Some(temp_path);
        }
    }
    fallback
}

/// Reads a millidegree Celsius `sysfs` file and returns degrees Celsius.
///
/// # Examples
/// ```
/// use heater::thermal_sensors::read_sysfs_millidegrees;
/// assert_eq!(read_sysfs_millidegrees(None), Ok(0.0));
/// ```
pub fn read_sysfs_millidegrees(file_path: Option<&Path>) -> Result<f64, String> {
    let Some(path) = file_path else {
        return Ok(0.0);
    };
    let Ok(raw_text) = fs::read_to_string(path) else {
        return Ok(0.0);
    };
    let trimmed = raw_text.trim();
    if trimmed.is_empty() {
        return Ok(0.0);
    }
    trimmed.parse::<f64>().map(|v| v / 1000.0).map_err(|_| {
        format!(
            "Expected integer millidegrees in {}, got {:?}",
            path.display(),
            trimmed
        )
    })
}

/// Thin project-owned dynamic library wrapper around `libnvidia-ml.so.1`.
pub struct NvmlLibraryWrapper {
    lib: Library,
    device_handle: *mut c_void,
}

// Safety: NVML functions are thread-safe after initialization.
unsafe impl Send for NvmlLibraryWrapper {}

impl NvmlLibraryWrapper {
    /// Loads `libnvidia-ml.so.1`, initializes NVML, and acquires GPU handle 0.
    pub fn open_default() -> Result<Self, String> {
        unsafe {
            let lib = Library::new("libnvidia-ml.so.1")
                .map_err(|e| format!("Expected libnvidia-ml.so.1 to load, got error: {e}"))?;
            let init_fn: Symbol<unsafe extern "C" fn() -> i32> =
                lib.get(b"nvmlInit_v2\0").map_err(|e| e.to_string())?;
            let rc = init_fn();
            if rc != 0 {
                return Err(format!("Expected nvmlInit_v2 code 0, got {rc}"));
            }
            let get_handle: Symbol<unsafe extern "C" fn(u32, *mut *mut c_void) -> i32> = lib
                .get(b"nvmlDeviceGetHandleByIndex_v2\0")
                .map_err(|e| e.to_string())?;
            let mut handle: *mut c_void = std::ptr::null_mut();
            let rc_dev = get_handle(0, &mut handle);
            if rc_dev != 0 {
                return Err(format!(
                    "Expected nvmlDeviceGetHandleByIndex_v2 code 0, got {rc_dev}"
                ));
            }
            Ok(Self {
                lib,
                device_handle: handle,
            })
        }
    }

    /// Queries `(temp_c, power_w, power_limit_w)` from NVML GPU 0.
    pub fn query_gpu_metrics(&self) -> (f64, f64, f64) {
        unsafe {
            let mut temp_c: u32 = 0;
            let mut power_mw: u32 = 0;
            let mut limit_mw: u32 = 0;
            if let Ok(get_temp) = self
                .lib
                .get::<unsafe extern "C" fn(*mut c_void, u32, *mut u32) -> i32>(
                    b"nvmlDeviceGetTemperature\0",
                )
            {
                let _ = get_temp(self.device_handle, 0, &mut temp_c);
            }
            if let Ok(get_pwr) = self
                .lib
                .get::<unsafe extern "C" fn(*mut c_void, *mut u32) -> i32>(
                    b"nvmlDeviceGetPowerUsage\0",
                )
            {
                let _ = get_pwr(self.device_handle, &mut power_mw);
            }
            if let Ok(get_lim) = self
                .lib
                .get::<unsafe extern "C" fn(*mut c_void, *mut u32) -> i32>(
                    b"nvmlDeviceGetEnforcedPowerLimit\0",
                )
            {
                let _ = get_lim(self.device_handle, &mut limit_mw);
            }
            let limit_w = if limit_mw > 0 {
                f64::from(limit_mw) / 1000.0
            } else {
                110.0
            };
            (f64::from(temp_c), f64::from(power_mw) / 1000.0, limit_w)
        }
    }

    /// Shuts down the NVML library cleanly.
    pub fn shutdown(&self) {
        unsafe {
            if let Ok(shut_fn) = self
                .lib
                .get::<unsafe extern "C" fn() -> i32>(b"nvmlShutdown\0")
            {
                let _ = shut_fn();
            }
        }
    }
}

/// Combines NVML GPU telemetry and Linux `sysfs` CPU thermal telemetry.
pub struct LinuxNvmlThermalProbe {
    nvml: NvmlLibraryWrapper,
    cpu_temp_path: Option<PathBuf>,
    last_instant: Option<Instant>,
}

impl LinuxNvmlThermalProbe {
    /// Constructs a new `LinuxNvmlThermalProbe` from an initialized NVML wrapper and sysfs path.
    pub fn new(nvml: NvmlLibraryWrapper, cpu_temp_path: Option<PathBuf>) -> Self {
        Self {
            nvml,
            cpu_temp_path,
            last_instant: None,
        }
    }
}

impl ThermalSensorProbe for LinuxNvmlThermalProbe {
    fn read_telemetry(&mut self) -> Result<ThermalReading, String> {
        let now = Instant::now();
        let dt_sec = match self.last_instant {
            Some(prev) => now.duration_since(prev).as_secs_f64().max(0.01),
            None => 0.25,
        };
        self.last_instant = Some(now);
        let (gpu_temp, gpu_power, gpu_limit) = self.nvml.query_gpu_metrics();
        let raw_cpu = read_sysfs_millidegrees(self.cpu_temp_path.as_deref())?;
        let cpu_temp = if raw_cpu > 0.0 { raw_cpu } else { gpu_temp };
        Ok(ThermalReading {
            gpu_temp_c: gpu_temp,
            cpu_temp_c: cpu_temp,
            gpu_power_w: gpu_power,
            gpu_power_limit_w: gpu_limit,
            elapsed_dt_sec: dt_sec,
        })
    }

    fn close_probe(&mut self) {
        self.nvml.shutdown();
    }
}
