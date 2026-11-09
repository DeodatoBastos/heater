"""CLI entrypoint for the PC Thermal Regulator TUI."""

import argparse
import curses
import sys

from heater.load_workers import MultiprocessLoadActuator
from heater.models import PidCoefficients
from heater.orchestrator import ThermalOrchestrator
from heater.pid_regulator import PidThermalRegulator
from heater.thermal_sensors import (
    LinuxNvmlThermalProbe,
    NvmlLibraryWrapper,
    discover_cpu_temp_sysfs_path,
)
from heater.tui_renderer import CursesThermalTui


def parse_cli_arguments(argv: list[str]) -> argparse.Namespace:
    """Parses command-line arguments for target temperature and EMA smoothing.

    Example:
        >>> args = parse_cli_arguments(["--target", "70"])
        >>> args.target
        70.0
    """
    parser = argparse.ArgumentParser(
        description="PC Thermal Regulator TUI using GPU and CPU load."
    )
    parser.add_argument(
        "--target",
        type=float,
        default=65.0,
        help="Initial target temperature in Celsius (30.0 to 82.0).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="EMA smoothing factor alpha in (0.0, 1.0].",
    )
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=None,
        help="Number of CPU worker processes to spawn.",
    )
    return parser.parse_args(argv)


def build_default_orchestrator(args: argparse.Namespace) -> ThermalOrchestrator:
    """Wires hardware probes, PID controller, and multiprocess load workers."""
    nvml = NvmlLibraryWrapper()
    cpu_path = discover_cpu_temp_sysfs_path()
    probe = LinuxNvmlThermalProbe(nvml, cpu_path)
    regulator = PidThermalRegulator(PidCoefficients())
    actuator = MultiprocessLoadActuator(cpu_worker_count=args.cpu_workers)
    return ThermalOrchestrator(
        sensor_probe=probe,
        regulator=regulator,
        actuator=actuator,
        initial_target_c=max(30.0, min(82.0, args.target)),
        ema_alpha=args.alpha,
    )


def main(argv: list[str] | None = None) -> int:
    """Runs the interactive curses thermal regulation TUI."""
    args = parse_cli_arguments(argv if argv is not None else sys.argv[1:])
    orchestrator = build_default_orchestrator(args)
    tui = CursesThermalTui(orchestrator)
    try:
        curses.wrapper(tui.run_interactive_loop)
    except KeyboardInterrupt:
        pass
    finally:
        orchestrator.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
