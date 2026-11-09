# PC Thermal Regulator (`heater`)

A terminal user interface (TUI) for regulating PC temperature as a space/desk heater using combined NVIDIA GPU (via CUDA Driver API `libcuda.so.1`) and multi-core CPU compute load, with sub-millisecond NVML (`libnvidia-ml.so.1`) and Linux `sysfs` telemetry.

## Features

- **Real-time Telemetry**: Instantaneous GPU & CPU temperature, Exponential Moving Average (EMA) temperature, instantaneous GPU power draw (Watts), enforced power limit, and active duty cycle.
- **Closed-Loop PID Thermal Regulator**: Automatically modulates GPU CUDA FMA bursts and multi-core CPU worker duty cycles (`0%` to `100%`) to reach and hold your target temperature, with anti-windup and a hard safety cutoff at `85.0 C`.
- **Interactive Curses TUI**: ASCII gauges and rolling history sparklines for Instant Temperature, EMA Temperature, and GPU Power.

## Usage

Run the TUI directly:

```bash
PYTHONPATH=src python3 -m heater.cli --target 65.0
```

### Keyboard Controls

- `Up` / `k` / `+`: Increase target temperature by `1.0 C`
- `Down` / `j` / `-`: Decrease target temperature by `1.0 C`
- `Right` / `Left`: Adjust target temperature by `+5.0 C` / `-5.0 C`
- `Space`: Pause / Resume heating immediately
- `m`: Cycle control mode (`AUTO` -> `MANUAL` -> `PAUSED`)
- `[` / `]`: Decrease / Increase manual duty cycle by `5%`
- `s`: Switch primary regulated sensor (`GPU` <-> `CPU`)
- `g` / `c`: Toggle GPU or CPU heater workers independently
- `q`: Quit cleanly

## Running Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
