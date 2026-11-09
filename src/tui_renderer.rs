//! Ratatui terminal dashboard with Braille 2D line charts and gauges.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Axis, Block, Borders, Chart, Dataset, Gauge, GraphType, Paragraph};

use crate::load_workers::ThermalLoadActuator;
use crate::models::{SensorSource, TelemetrySnapshot};
use crate::orchestrator::ThermalOrchestrator;
use crate::thermal_sensors::ThermalSensorProbe;

/// Renders the complete TUI dashboard frame using `ratatui` widgets.
pub fn render_dashboard_frame<P: ThermalSensorProbe, A: ThermalLoadActuator>(
    frame: &mut Frame<'_>,
    orch: &ThermalOrchestrator<P, A>,
    snap: &TelemetrySnapshot,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(5),
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(8),
            Constraint::Length(3),
        ])
        .split(frame.area());
    render_summary_header(frame, chunks[0], snap);
    render_gauges_row(frame, chunks[1], snap);
    render_temperature_chart(frame, chunks[2], orch, snap);
    render_power_chart(frame, chunks[3], orch, snap);
    render_controls_footer(frame, chunks[4]);
}

fn render_summary_header(frame: &mut Frame<'_>, area: Rect, snap: &TelemetrySnapshot) {
    let gpu_status = if snap.gpu_worker_enabled {
        "ON "
    } else {
        "OFF"
    };
    let cpu_status = if snap.cpu_workers_enabled {
        "ON "
    } else {
        "OFF"
    };
    let lines = vec![
        Line::from(vec![Span::styled(
            format!(
                " MODE: {:<6} | PRIMARY SENSOR: {:<3} | TARGET TEMP: {:5.1} C | DUTY: {:5.1}% ",
                snap.mode.label(),
                snap.active_sensor.label(),
                snap.target_temp_c,
                snap.duty_cycle * 100.0
            ),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )]),
        Line::from(format!(
            " GPU -> Instant: {:5.1} C | EMA: {:5.2} C | Power: {:5.1} W / {:5.1} W | Heater: {}",
            snap.instant_gpu_temp_c,
            snap.ema_gpu_temp_c,
            snap.gpu_power_w,
            snap.gpu_power_limit_w,
            gpu_status
        )),
        Line::from(format!(
            " CPU -> Instant: {:5.1} C | EMA: {:5.2} C | Safety Ceiling: 85.0 C  | Heater: {}",
            snap.instant_cpu_temp_c, snap.ema_cpu_temp_c, cpu_status
        )),
    ];
    let block = Block::default()
        .title(" PC THERMAL REGULATOR ")
        .borders(Borders::ALL);
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

fn render_gauges_row(frame: &mut Frame<'_>, area: Rect, snap: &TelemetrySnapshot) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(34),
            Constraint::Percentage(33),
            Constraint::Percentage(33),
        ])
        .split(area);
    let primary_ema = match snap.active_sensor {
        SensorSource::Gpu => snap.ema_gpu_temp_c,
        SensorSource::Cpu => snap.ema_cpu_temp_c,
    };
    let temp_ratio = ((primary_ema - 25.0) / 60.0).clamp(0.0, 1.0);
    let power_ratio = (snap.gpu_power_w / snap.gpu_power_limit_w.max(1.0)).clamp(0.0, 1.0);
    render_single_gauge(
        frame,
        cols[0],
        " EMA Temp vs Ceiling (85C) ",
        temp_ratio,
        format!("{primary_ema:.1} C / {:.1} C Target", snap.target_temp_c),
        Color::Green,
    );
    render_single_gauge(
        frame,
        cols[1],
        " GPU Power Draw ",
        power_ratio,
        format!(
            "{:.1} W / {:.0} W",
            snap.gpu_power_w, snap.gpu_power_limit_w
        ),
        Color::Yellow,
    );
    render_single_gauge(
        frame,
        cols[2],
        " Heater Duty Cycle ",
        snap.duty_cycle.clamp(0.0, 1.0),
        format!("{:.1} %", snap.duty_cycle * 100.0),
        Color::LightRed,
    );
}

fn render_single_gauge(
    frame: &mut Frame<'_>,
    area: Rect,
    title: &str,
    ratio: f64,
    label: String,
    color: Color,
) {
    let text_span = Span::styled(
        format!(" {label} "),
        Style::default()
            .fg(Color::White)
            .bg(Color::Rgb(24, 26, 36))
            .add_modifier(Modifier::BOLD),
    );
    let gauge = Gauge::default()
        .block(Block::default().title(title).borders(Borders::ALL))
        .gauge_style(
            Style::default()
                .fg(color)
                .bg(Color::Rgb(35, 38, 52))
                .add_modifier(Modifier::BOLD),
        )
        .use_unicode(true)
        .ratio(ratio)
        .label(text_span);
    frame.render_widget(gauge, area);
}

fn render_temperature_chart<P: ThermalSensorProbe, A: ThermalLoadActuator>(
    frame: &mut Frame<'_>,
    area: Rect,
    orch: &ThermalOrchestrator<P, A>,
    snap: &TelemetrySnapshot,
) {
    let inst_pts = orch.instant_history.as_chart_points();
    let ema_pts = orch.ema_history.as_chart_points();
    let tgt_pts = orch.target_history.as_chart_points();
    let (min_t, max_t, _) = orch.instant_history.summary_stats();
    let datasets = vec![
        Dataset::default()
            .name("Target (C)")
            .marker(Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Red))
            .data(&tgt_pts),
        Dataset::default()
            .name("Instant Temp (C)")
            .marker(Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::DarkGray))
            .data(&inst_pts),
        Dataset::default()
            .name("EMA Temp (C)")
            .marker(Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Cyan))
            .data(&ema_pts),
    ];
    let title = format!(
        " Temperature History ({}) | Min: {min_t:.1}C  Max: {max_t:.1}C  EMA: {:.2}C  Target: {:.1}C ",
        snap.active_sensor.label(),
        orch.ema_history.summary_stats().2,
        snap.target_temp_c
    );
    let max_x = orch.instant_history.capacity() as f64;
    let chart = Chart::new(datasets)
        .block(Block::default().title(title).borders(Borders::ALL))
        .x_axis(
            Axis::default()
                .title("Time (-30s -> Now)")
                .bounds([0.0, max_x])
                .labels(vec![Span::raw("-30s"), Span::raw("-15s"), Span::raw("Now")]),
        )
        .y_axis(
            Axis::default()
                .title("Celsius")
                .bounds([25.0, 90.0])
                .labels(vec![Span::raw("25C"), Span::raw("55C"), Span::raw("85C")]),
        );
    frame.render_widget(chart, area);
}

fn render_power_chart<P: ThermalSensorProbe, A: ThermalLoadActuator>(
    frame: &mut Frame<'_>,
    area: Rect,
    orch: &ThermalOrchestrator<P, A>,
    snap: &TelemetrySnapshot,
) {
    let pwr_pts = orch.power_history.as_chart_points();
    let (min_p, max_p, cur_p) = orch.power_history.summary_stats();
    let datasets = vec![
        Dataset::default()
            .name("GPU Power (W)")
            .marker(Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Yellow))
            .data(&pwr_pts),
    ];
    let upper_w = snap.gpu_power_limit_w.max(60.0);
    let title =
        format!(" GPU Power History | Min: {min_p:.1}W  Max: {max_p:.1}W  Current: {cur_p:.1}W ");
    let max_x = orch.power_history.capacity() as f64;
    let chart = Chart::new(datasets)
        .block(Block::default().title(title).borders(Borders::ALL))
        .x_axis(Axis::default().bounds([0.0, max_x]))
        .y_axis(Axis::default().bounds([0.0, upper_w]).labels(vec![
            Span::raw("0W"),
            Span::raw(format!("{:.0}W", upper_w * 0.5)),
            Span::raw(format!("{upper_w:.0}W")),
        ]));
    frame.render_widget(chart, area);
}

fn render_controls_footer(frame: &mut Frame<'_>, area: Rect) {
    let text = Line::from(vec![Span::raw(
        " [Up/Down/j/k] Target +/-1C | [Left/Right] +/-5C | [Space] Pause | [m] Mode | [ [/] ] Manual Duty | [s] Sensor | [g/c] Toggle GPU/CPU | [q] Quit ",
    )]);
    let block = Block::default().title(" Controls ").borders(Borders::ALL);
    frame.render_widget(Paragraph::new(text).block(block), area);
}
