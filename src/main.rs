//! CLI entrypoint and event loop for the Rust PC Thermal Regulator TUI.

use std::env;
use std::io;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

use heater::load_workers::{ThermalLoadActuator, ThreadedLoadActuator};
use heater::models::PidCoefficients;
use heater::orchestrator::ThermalOrchestrator;
use heater::pid_regulator::PidThermalRegulator;
use heater::thermal_sensors::{
    LinuxNvmlThermalProbe, NvmlLibraryWrapper, ThermalSensorProbe, discover_cpu_temp_sysfs_path,
};
use heater::tui_renderer::render_dashboard_frame;

/// Parsed command-line configuration for the thermal regulator.
#[derive(Debug, Clone, PartialEq)]
pub struct CliConfig {
    pub target_temp_c: f64,
    pub ema_alpha: f64,
    pub cpu_threads: Option<usize>,
}

/// Parses command-line arguments (`--target`, `--alpha`, `--cpu-workers`).
pub fn parse_cli_args(args: &[String]) -> Result<CliConfig, String> {
    let mut cfg = CliConfig {
        target_temp_c: 65.0,
        ema_alpha: 0.2,
        cpu_threads: None,
    };
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--target" if idx + 1 < args.len() => {
                cfg.target_temp_c = args[idx + 1]
                    .parse()
                    .map_err(|_| format!("Expected float for --target, got {:?}", args[idx + 1]))?;
                idx += 2;
            }
            "--alpha" if idx + 1 < args.len() => {
                cfg.ema_alpha = args[idx + 1]
                    .parse()
                    .map_err(|_| format!("Expected float for --alpha, got {:?}", args[idx + 1]))?;
                idx += 2;
            }
            "--cpu-workers" if idx + 1 < args.len() => {
                cfg.cpu_threads = Some(args[idx + 1].parse().map_err(|_| {
                    format!(
                        "Expected integer for --cpu-workers, got {:?}",
                        args[idx + 1]
                    )
                })?);
                idx += 2;
            }
            other => {
                return Err(format!(
                    "Expected [--target <f64>] [--alpha <f64>] [--cpu-workers <usize>], got {other:?}"
                ));
            }
        }
    }
    Ok(cfg)
}

fn handle_key_event<P: ThermalSensorProbe, A: ThermalLoadActuator>(
    orch: &mut ThermalOrchestrator<P, A>,
    code: KeyCode,
) -> bool {
    match code {
        KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => false,
        KeyCode::Up | KeyCode::Char('k') | KeyCode::Char('+') | KeyCode::Char('=') => {
            orch.adjust_target_temperature(1.0);
            true
        }
        KeyCode::Down | KeyCode::Char('j') | KeyCode::Char('-') | KeyCode::Char('_') => {
            orch.adjust_target_temperature(-1.0);
            true
        }
        KeyCode::Right => {
            orch.adjust_target_temperature(5.0);
            true
        }
        KeyCode::Left => {
            orch.adjust_target_temperature(-5.0);
            true
        }
        other => {
            handle_toggle_keys(orch, other);
            true
        }
    }
}

fn handle_toggle_keys<P: ThermalSensorProbe, A: ThermalLoadActuator>(
    orch: &mut ThermalOrchestrator<P, A>,
    code: KeyCode,
) {
    match code {
        KeyCode::Char(' ') => {
            orch.toggle_pause_mode();
        }
        KeyCode::Char('m') | KeyCode::Char('M') => {
            orch.cycle_control_mode();
        }
        KeyCode::Char('s') | KeyCode::Char('S') => {
            orch.toggle_active_sensor();
        }
        KeyCode::Char('g') | KeyCode::Char('G') => {
            orch.toggle_gpu_heating();
        }
        KeyCode::Char('c') | KeyCode::Char('C') => {
            orch.toggle_cpu_heating();
        }
        KeyCode::Char(']') => {
            orch.adjust_manual_duty(0.05);
        }
        KeyCode::Char('[') => {
            orch.adjust_manual_duty(-0.05);
        }
        _ => {}
    }
}

fn run_tui_loop<P: ThermalSensorProbe, A: ThermalLoadActuator>(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    orch: &mut ThermalOrchestrator<P, A>,
) -> Result<(), String> {
    let tick_rate = Duration::from_millis(250);
    let mut last_tick = Instant::now() - tick_rate;
    let mut snap = orch.step_cycle()?;
    loop {
        if last_tick.elapsed() >= tick_rate {
            snap = orch.step_cycle()?;
            last_tick = Instant::now();
        }
        terminal
            .draw(|frame| render_dashboard_frame(frame, orch, &snap))
            .map_err(|e| e.to_string())?;
        let timeout = tick_rate.saturating_sub(last_tick.elapsed());
        if event::poll(timeout).map_err(|e| e.to_string())?
            && let Event::Key(key) = event::read().map_err(|e| e.to_string())?
            && key.kind == KeyEventKind::Press
            && !handle_key_event(orch, key.code)
        {
            return Ok(());
        }
    }
}

fn main() -> Result<(), String> {
    let raw_args: Vec<String> = env::args().skip(1).collect();
    let cfg = parse_cli_args(&raw_args)?;
    let nvml = NvmlLibraryWrapper::open_default()?;
    let cpu_path = discover_cpu_temp_sysfs_path("/sys/class/thermal");
    let probe = LinuxNvmlThermalProbe::new(nvml, cpu_path);
    let regulator = PidThermalRegulator::new(PidCoefficients::default());
    let actuator = ThreadedLoadActuator::spawn_workers(cfg.cpu_threads);
    let mut orch =
        ThermalOrchestrator::new(probe, regulator, actuator, cfg.target_temp_c, cfg.ema_alpha)?;
    enable_raw_mode().map_err(|e| e.to_string())?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).map_err(|e| e.to_string())?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).map_err(|e| e.to_string())?;
    let res = run_tui_loop(&mut terminal, &mut orch);
    let _ = disable_raw_mode();
    let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
    let _ = terminal.show_cursor();
    orch.shutdown();
    res
}
