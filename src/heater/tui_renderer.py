"""Curses terminal user interface for PC thermal regulation."""

import curses
import time

from heater.models import TelemetrySnapshot
from heater.orchestrator import ThermalOrchestrator


def format_progress_bar(fraction: float, width: int) -> str:
    """Formats a plain-text ASCII progress bar `[####----]` of total `width`.

    Example:
        >>> format_progress_bar(0.5, 10)
        '[####----]'
    """
    if width < 4:
        raise ValueError(f"Expected width >= 4, got {width}")
    inner = width - 2
    clamped = max(0.0, min(1.0, fraction))
    filled = int(round(clamped * inner))
    return "[" + ("#" * filled) + ("-" * (inner - filled)) + "]"


def render_sparkline_row(
    samples: list[float],
    width: int,
    low_bound: float,
    high_bound: float,
) -> str:
    """Maps scalar samples to an ASCII height bar string of length `width`.

    Example:
        >>> len(render_sparkline_row([40.0, 60.0], 5, 30.0, 80.0))
        5
    """
    if width <= 0:
        raise ValueError(f"Expected positive width > 0, got {width}")
    levels = "._-:=+*#"
    tail = samples[-width:]
    padded = ([low_bound] * max(0, width - len(tail))) + tail
    span = max(1.0, high_bound - low_bound)
    chars: list[str] = []
    for val in padded:
        norm = max(0.0, min(1.0, (val - low_bound) / span))
        idx = min(len(levels) - 1, int(norm * (len(levels) - 1)))
        chars.append(levels[idx])
    return "".join(chars)


class CursesThermalTui:
    """Renders telemetry and handles keyboard controls in a curses terminal."""

    def __init__(self, orchestrator: ThermalOrchestrator) -> None:
        self._orch = orchestrator
        self._running: bool = True

    def run_interactive_loop(self, stdscr: curses.window) -> None:
        """Main curses event and telemetry loop."""
        self._configure_curses(stdscr)
        while self._running:
            t_start = time.monotonic()
            snapshot = self._orch.step_cycle()
            self._draw_dashboard(stdscr, snapshot)
            self._poll_input_keys(stdscr, t_start)

    def _configure_curses(self, stdscr: curses.window) -> None:
        """Initializes curses colors, cursor visibility, and non-blocking input."""
        curses.curs_set(0)
        stdscr.nodelay(True)
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_CYAN, -1)
            curses.init_pair(2, curses.COLOR_GREEN, -1)
            curses.init_pair(3, curses.COLOR_YELLOW, -1)
            curses.init_pair(4, curses.COLOR_RED, -1)

    def _poll_input_keys(self, stdscr: curses.window, tick_start: float) -> None:
        """Polls keyboard input for up to 250ms per telemetry cycle."""
        while time.monotonic() - tick_start < 0.25 and self._running:
            key = stdscr.getch()
            if key != -1:
                self._handle_keypress(key)
                break
            time.sleep(0.02)

    def _handle_keypress(self, key: int) -> None:
        """Dispatches a single key code to the orchestrator."""
        if key in (ord("q"), ord("Q")):
            self._running = False
        elif key in (curses.KEY_UP, ord("k"), ord("+"), ord("=")):
            self._orch.adjust_target_temperature(1.0)
        elif key in (curses.KEY_DOWN, ord("j"), ord("-"), ord("_")):
            self._orch.adjust_target_temperature(-1.0)
        elif key == curses.KEY_RIGHT:
            self._orch.adjust_target_temperature(5.0)
        elif key == curses.KEY_LEFT:
            self._orch.adjust_target_temperature(-5.0)
        else:
            self._handle_toggle_keys(key)

    def _handle_toggle_keys(self, key: int) -> None:
        """Handles mode, sensor, and worker toggle keys."""
        if key == ord(" "):
            self._orch.toggle_pause_mode()
        elif key in (ord("m"), ord("M")):
            self._orch.cycle_control_mode()
        elif key in (ord("s"), ord("S")):
            self._orch.toggle_active_sensor()
        elif key in (ord("g"), ord("G")):
            self._orch.toggle_gpu_heating()
        elif key in (ord("c"), ord("C")):
            self._orch.toggle_cpu_heating()
        elif key == ord("]"):
            self._orch.adjust_manual_duty(0.05)
        elif key == ord("["):
            self._orch.adjust_manual_duty(-0.05)

    def _draw_dashboard(
        self, stdscr: curses.window, snap: TelemetrySnapshot
    ) -> None:
        """Clears and redraws the entire TUI dashboard frame."""
        stdscr.erase()
        max_y, max_x = stdscr.getmaxyx()
        width = max(40, min(88, max_x - 4))
        self._draw_header(stdscr, snap, width)
        self._draw_metrics_table(stdscr, snap, width)
        self._draw_bars(stdscr, snap, width)
        self._draw_charts(stdscr, snap, width)
        self._draw_footer(stdscr, max_y, width)
        stdscr.refresh()

    def _draw_header(
        self, stdscr: curses.window, snap: TelemetrySnapshot, width: int
    ) -> None:
        """Draws top title bar and system status line."""
        border = "=" * width
        self._safe_addstr(stdscr, 1, 2, border, curses.A_BOLD)
        title = (
            f" PC THERMAL REGULATOR  |  MODE: {snap.mode.upper()}  |  "
            f"PRIMARY SENSOR: {snap.active_sensor.upper()} "
        )
        self._safe_addstr(stdscr, 2, 2, title[:width], curses.A_BOLD | curses.color_pair(1))
        self._safe_addstr(stdscr, 3, 2, border, curses.A_BOLD)

    def _draw_metrics_table(
        self, stdscr: curses.window, snap: TelemetrySnapshot, width: int
    ) -> None:
        """Draws target, instant, EMA temperatures, and GPU power readouts."""
        row1 = (
            f"TARGET TEMP : {snap.target_temp_c:5.1f} C    |  "
            f"GPU INSTANT : {snap.instant_gpu_temp_c:5.1f} C    |  "
            f"GPU EMA : {snap.ema_gpu_temp_c:5.2f} C"
        )
        row2 = (
            f"GPU POWER   : {snap.gpu_power_w:5.1f} W    |  "
            f"CPU INSTANT : {snap.instant_cpu_temp_c:5.1f} C    |  "
            f"CPU EMA : {snap.ema_cpu_temp_c:5.2f} C"
        )
        workers = (
            f"HEATERS     : GPU={'ON ' if snap.gpu_worker_enabled else 'OFF'} "
            f"CPU={'ON ' if snap.cpu_workers_enabled else 'OFF'}  |  "
            f"POWER LIMIT : {snap.gpu_power_limit_w:5.1f} W    |  "
            f"DUTY    : {snap.duty_cycle * 100.0:5.1f} %"
        )
        self._safe_addstr(stdscr, 5, 2, row1[:width], curses.color_pair(2))
        self._safe_addstr(stdscr, 6, 2, row2[:width], curses.color_pair(2))
        self._safe_addstr(stdscr, 7, 2, workers[:width], curses.color_pair(3))

    def _draw_bars(
        self, stdscr: curses.window, snap: TelemetrySnapshot, width: int
    ) -> None:
        """Draws horizontal ASCII gauges for temperature, GPU power, and duty cycle."""
        bar_w = max(10, width - 28)
        primary_ema = (
            snap.ema_gpu_temp_c if snap.active_sensor == "gpu" else snap.ema_cpu_temp_c
        )
        t_frac = (primary_ema - 25.0) / 65.0
        p_frac = snap.gpu_power_w / max(1.0, snap.gpu_power_limit_w)
        t_bar = format_progress_bar(t_frac, bar_w)
        p_bar = format_progress_bar(p_frac, bar_w)
        d_bar = format_progress_bar(snap.duty_cycle, bar_w)
        self._safe_addstr(
            stdscr, 9, 2, f"EMA TEMP  ({primary_ema:5.1f}C) : {t_bar}"[:width]
        )
        self._safe_addstr(
            stdscr, 10, 2, f"GPU POWER ({snap.gpu_power_w:5.1f}W) : {p_bar}"[:width]
        )
        self._safe_addstr(
            stdscr, 11, 2, f"HEAT DUTY ({snap.duty_cycle*100:5.1f}%) : {d_bar}"[:width]
        )

    def _draw_charts(
        self, stdscr: curses.window, snap: TelemetrySnapshot, width: int
    ) -> None:
        """Draws rolling ASCII sparklines for Instant Temp, EMA Temp, and Power."""
        chart_w = max(10, width - 22)
        inst_spark = render_sparkline_row(
            self._orch.temp_history.recent_values(chart_w), chart_w, 30.0, 85.0
        )
        ema_spark = render_sparkline_row(
            self._orch.ema_history.recent_values(chart_w), chart_w, 30.0, 85.0
        )
        pwr_spark = render_sparkline_row(
            self._orch.power_history.recent_values(chart_w),
            chart_w,
            0.0,
            max(60.0, snap.gpu_power_limit_w),
        )
        self._safe_addstr(stdscr, 13, 2, "-" * width)
        self._safe_addstr(stdscr, 14, 2, f"INSTANT TEMP (30-85C): {inst_spark}"[:width])
        self._safe_addstr(stdscr, 15, 2, f"EMA TEMP     (30-85C): {ema_spark}"[:width])
        self._safe_addstr(stdscr, 16, 2, f"GPU POWER    (Watts) : {pwr_spark}"[:width])

    def _draw_footer(self, stdscr: curses.window, max_y: int, width: int) -> None:
        """Draws keyboard shortcut legend at the bottom of the screen."""
        row = min(18, max_y - 3)
        self._safe_addstr(stdscr, row, 2, "-" * width)
        help1 = (
            "KEYS: [Up/Down/j/k] Target +/-1C | [Left/Right] Target +/-5C | "
            "[Space] Pause/Resume"
        )
        help2 = (
            "      [m] Mode (Auto/Manual/Pause) | [ [ / ] ] Manual Duty +/-5% | "
            "[s] Sensor (GPU/CPU) | [g/c] Toggle GPU/CPU | [q] Quit"
        )
        self._safe_addstr(stdscr, row + 1, 2, help1[:width], curses.A_DIM)
        self._safe_addstr(stdscr, row + 2, 2, help2[:width], curses.A_DIM)

    def _safe_addstr(
        self,
        stdscr: curses.window,
        y: int,
        x: int,
        text: str,
        attr: int = 0,
    ) -> None:
        """Writes text to curses window, ignoring bounds errors on tiny terminals."""
        max_y, max_x = stdscr.getmaxyx()
        if 0 <= y < max_y and 0 <= x < max_x:
            try:
                stdscr.addstr(y, x, text[: max_x - x - 1], attr)
            except curses.error:
                pass
