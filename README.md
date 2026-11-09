# PC Thermal Regulator (`heater` in Rust + `ratatui`)

A native Rust terminal user interface (TUI) for regulating PC temperature as a desk/space heater using combined NVIDIA GPU (via CUDA Driver API `libcuda.so.1`) and multithreaded CPU compute load, with sub-millisecond NVML (`libnvidia-ml.so.1`) and Linux `sysfs` telemetry.

## Features

- **2D Braille Line Charts (`ratatui`)**:
  - **Temperature History Chart**: Plots **Target Temperature (Red)**, **Instant Temperature (Gray)**, and **EMA Smoothed Temperature (Cyan)** together on a 2D coordinate grid (`25°C – 90°C`) with live `Min / Max / EMA / Target` readouts.
  - **GPU Power History Chart**: Plots real-time GPU wattage (`0W – 110W`) with live `Min / Max / Current` readouts.
  - **Horizontal Gauges**: Dedicated progress bars for EMA Temperature, GPU Power Draw, and Heater Duty Cycle.
- **Native Multithreading + Direct CUDA Driver FFI**:
  - Uses `std::thread` with lock-free `AtomicU64` duty cycles (no Python GIL or heavy subprocess overhead).
  - Directly loads `libcuda.so.1` and `libnvidia-ml.so.1` via `libloading`.

## Running the TUI

```bash
cargo run --release -- --target 65.0
```

Or run the compiled binary directly:

```bash
./target/release/heater --target 65.0
```

### Keyboard Controls

| Key | Action |
| :--- | :--- |
| `Up` / `k` / `+` | Increase target temperature by `1.0 C` |
| `Down` / `j` / `-` | Decrease target temperature by `1.0 C` |
| `Right` / `Left` | Adjust target temperature by `+5.0 C` / `-5.0 C` |
| `Space` | Pause / Resume heating immediately |
| `m` | Cycle control mode (`AUTO` -> `MANUAL` -> `PAUSED`) |
| `[` / `]` | Decrease / Increase manual duty cycle by `5%` |
| `s` | Switch primary regulated sensor (`GPU` <-> `CPU`) |
| `g` / `c` | Toggle GPU or CPU heater threads independently |
| `q` / `Esc` | Quit cleanly |

## Running Tests

```bash
cargo test
```
