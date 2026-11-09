"""Unit tests for EMA filter, PID regulator, orchestrator, and TUI helpers."""

import unittest

from heater.cli import parse_cli_arguments
from heater.ema_tracker import EmaFilter, RollingMetricBuffer
from heater.load_workers import compute_slice_timings
from heater.models import PidCoefficients, ThermalReading
from heater.orchestrator import ThermalOrchestrator
from heater.pid_regulator import PidThermalRegulator, clamp_duty_cycle
from heater.tui_renderer import format_progress_bar, render_sparkline_row
from tests.fakes import FakeThermalLoadActuator, FakeThermalSensorProbe


class EmaFilterTests(unittest.TestCase):
    """Tests for EmaFilter and RollingMetricBuffer."""

    def test_ema_filter_initializes_and_smooths(self) -> None:
        tracker = EmaFilter(smoothing_alpha=0.5)
        self.assertEqual(tracker.current_value, 0.0)
        first = tracker.update_estimate(40.0)
        self.assertEqual(first, 40.0)
        second = tracker.update_estimate(60.0)
        self.assertEqual(second, 50.0)

    def test_ema_filter_rejects_invalid_alpha(self) -> None:
        with self.assertRaises(ValueError):
            EmaFilter(smoothing_alpha=0.0)
        with self.assertRaises(ValueError):
            EmaFilter(smoothing_alpha=1.5)

    def test_rolling_buffer_maintains_capacity(self) -> None:
        buf = RollingMetricBuffer(max_samples=3)
        for val in (10.0, 20.0, 30.0, 40.0):
            buf.append_sample(val)
        self.assertEqual(buf.recent_values(2), [30.0, 40.0])
        self.assertEqual(buf.recent_values(5), [20.0, 30.0, 40.0])
        self.assertEqual(buf.recent_values(0), [])


class PidRegulatorTests(unittest.TestCase):
    """Tests for PidThermalRegulator and duty clamping."""

    def test_pid_increases_duty_below_target(self) -> None:
        reg = PidThermalRegulator(PidCoefficients())
        duty = reg.compute_duty_cycle(50.0, 65.0, 0.25)
        self.assertGreater(duty, 0.5)
        self.assertLessEqual(duty, 1.0)

    def test_pid_cuts_off_at_hard_ceiling(self) -> None:
        reg = PidThermalRegulator(PidCoefficients(hard_ceiling_c=85.0))
        duty = reg.compute_duty_cycle(85.5, 90.0, 0.25)
        self.assertEqual(duty, 0.0)

    def test_pid_rejects_non_positive_dt(self) -> None:
        reg = PidThermalRegulator(PidCoefficients())
        with self.assertRaises(ValueError):
            reg.compute_duty_cycle(50.0, 60.0, 0.0)

    def test_clamp_duty_cycle_bounds(self) -> None:
        self.assertEqual(clamp_duty_cycle(-0.3), 0.0)
        self.assertEqual(clamp_duty_cycle(0.45), 0.45)
        self.assertEqual(clamp_duty_cycle(2.1), 1.0)


class OrchestratorTests(unittest.TestCase):
    """Tests for ThermalOrchestrator using FakeThermalSensorProbe and FakeThermalLoadActuator."""

    def setUp(self) -> None:
        reading = ThermalReading(
            gpu_temp_c=45.0,
            cpu_temp_c=52.0,
            gpu_power_w=12.5,
            gpu_power_limit_w=110.0,
            timestamp_sec=100.0,
        )
        self.probe = FakeThermalSensorProbe(reading)
        self.actuator = FakeThermalLoadActuator()
        self.regulator = PidThermalRegulator(PidCoefficients())
        self.orch = ThermalOrchestrator(
            sensor_probe=self.probe,
            regulator=self.regulator,
            actuator=self.actuator,
            initial_target_c=65.0,
            ema_alpha=0.5,
        )

    def test_step_cycle_updates_ema_and_actuates(self) -> None:
        snap = self.orch.step_cycle()
        self.assertEqual(snap.instant_gpu_temp_c, 45.0)
        self.assertEqual(snap.ema_gpu_temp_c, 45.0)
        self.assertGreater(snap.duty_cycle, 0.0)
        self.assertEqual(self.actuator.last_duty_cycle, snap.duty_cycle)
        self.assertTrue(self.actuator.last_gpu_enabled)
        self.assertTrue(self.actuator.last_cpu_enabled)

    def test_pause_and_manual_modes(self) -> None:
        self.orch.toggle_pause_mode()
        snap = self.orch.step_cycle()
        self.assertEqual(snap.mode, "paused")
        self.assertEqual(snap.duty_cycle, 0.0)
        self.orch.adjust_manual_duty(0.4)
        snap2 = self.orch.step_cycle()
        self.assertEqual(snap2.mode, "manual")
        self.assertAlmostEqual(snap2.duty_cycle, 0.4)

    def test_sensor_and_heater_toggles(self) -> None:
        self.assertEqual(self.orch.toggle_active_sensor(), "cpu")
        self.assertFalse(self.orch.toggle_gpu_heating())
        self.assertFalse(self.orch.toggle_cpu_heating())
        self.assertEqual(self.orch.adjust_target_temperature(5.0), 70.0)
        self.orch.shutdown()
        self.assertTrue(self.probe.closed)
        self.assertTrue(self.actuator.was_shutdown)


class HelpersAndFormattingTests(unittest.TestCase):
    """Tests for slice timing, ASCII bars, sparklines, and CLI parsing."""

    def test_compute_slice_timings(self) -> None:
        busy, sleep = compute_slice_timings(0.25, 0.1)
        self.assertAlmostEqual(busy, 0.025)
        self.assertAlmostEqual(sleep, 0.075)
        with self.assertRaises(ValueError):
            compute_slice_timings(1.2, 0.1)

    def test_format_progress_bar_and_sparkline(self) -> None:
        bar = format_progress_bar(0.5, 10)
        self.assertEqual(bar, "[####----]")
        spark = render_sparkline_row([30.0, 55.0, 80.0], 5, 30.0, 80.0)
        self.assertEqual(len(spark), 5)

    def test_parse_cli_arguments(self) -> None:
        args = parse_cli_arguments(["--target", "72.5", "--alpha", "0.3"])
        self.assertEqual(args.target, 72.5)
        self.assertEqual(args.alpha, 0.3)


if __name__ == "__main__":
    unittest.main()
