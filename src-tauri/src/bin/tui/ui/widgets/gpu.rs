//! Top GPU telemetry strip and host fallback card widget.

use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph, Sparkline},
};

use crate::app::{AppState, BAR_GPU_TEAL, BORDER_SOFT, LABEL_MUTED, TITLE_ACCENT};
use crate::ui::render_contrast_gauge;

pub fn render_top_gpu_strip(frame: &mut Frame, area: Rect, state: &AppState) {
    if state.gpus.is_empty() {
        render_host_card(frame, area, state);
        return;
    }

    let count = state.gpus.len();
    let constraints: Vec<Constraint> = (0..count)
        .map(|_| Constraint::Ratio(1, count as u32))
        .collect();
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(constraints)
        .split(area);

    for (idx, gpu) in state.gpus.iter().enumerate() {
        let card_area = chunks[idx];
        let power_str = gpu
            .power_watts
            .map(|w| format!("{w:.0}W"))
            .unwrap_or_else(|| "--W".to_string());
        let temp_str = gpu
            .temp_celsius
            .map(|t| format!("{t:.0}°C"))
            .unwrap_or_else(|| "--°C".to_string());

        let title = format!(
            " GPU {}: {} │ {} │ {} ",
            gpu.index, gpu.name, power_str, temp_str
        );
        let block = Block::default()
            .title(Span::styled(
                title,
                Style::default()
                    .fg(TITLE_ACCENT)
                    .add_modifier(Modifier::BOLD),
            ))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(BORDER_SOFT));

        let inner = block.inner(card_area);
        frame.render_widget(block, card_area);

        if inner.height == 0 || inner.width == 0 {
            continue;
        }

        let padded = Rect {
            x: inner.x + 1,
            y: inner.y,
            width: inner.width.saturating_sub(2),
            height: inner.height,
        };
        if padded.width == 0 {
            continue;
        }

        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(1)])
            .split(padded);

        let vram_ratio = if gpu.vram_total_gb > 0.0 {
            (gpu.vram_used_gb / gpu.vram_total_gb).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let vram_label = if gpu.vram_total_gb > 0.0 {
            format!(
                "VRAM {:.2} / {:.2} GiB ({:.0}%)",
                gpu.vram_used_gb,
                gpu.vram_total_gb,
                vram_ratio * 100.0
            )
        } else {
            format!("VRAM {:.2} GiB", gpu.vram_used_gb)
        };

        render_contrast_gauge(frame, rows[0], vram_ratio, &vram_label, BAR_GPU_TEAL);

        let spark_cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Length(13), Constraint::Min(4)])
            .split(rows[1]);

        let util_text = Paragraph::new(Line::from(vec![
            Span::styled("Util ", Style::default().fg(LABEL_MUTED)),
            Span::styled(
                format!("{:>5.1}%", gpu.util_percent),
                Style::default()
                    .fg(Color::LightGreen)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
        frame.render_widget(util_text, spark_cols[0]);

        let samples: Vec<u64> = gpu.util_history.iter().copied().collect();
        let sparkline = Sparkline::default()
            .data(&samples)
            .max(100)
            .style(Style::default().fg(Color::Rgb(106, 204, 142)));
        frame.render_widget(sparkline, spark_cols[1]);
    }
}

pub fn render_host_card(frame: &mut Frame, area: Rect, state: &AppState) {
    let title = format!(
        " Host Telemetry │ CPU {:.1}% │ RAM {:.1}/{:.1} GiB ",
        state.host_cpu_percent, state.host_ram_used_gb, state.host_ram_total_gb
    );
    let block = Block::default()
        .title(Span::styled(
            title,
            Style::default()
                .fg(TITLE_ACCENT)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(BORDER_SOFT));

    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height == 0 || inner.width <= 2 {
        return;
    }

    let padded = Rect {
        x: inner.x + 1,
        y: inner.y,
        width: inner.width.saturating_sub(2),
        height: inner.height,
    };

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(1)])
        .split(padded);

    let ram_ratio = if state.host_ram_total_gb > 0.0 {
        (state.host_ram_used_gb / state.host_ram_total_gb).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let ram_label = format!(
        "System RAM {:.1} / {:.1} GiB ({:.0}%)",
        state.host_ram_used_gb,
        state.host_ram_total_gb,
        ram_ratio * 100.0
    );
    render_contrast_gauge(frame, rows[0], ram_ratio, &ram_label, BAR_GPU_TEAL);

    let samples: Vec<u64> = state.host_cpu_history.iter().copied().collect();
    let spark = Sparkline::default()
        .data(&samples)
        .max(100)
        .style(Style::default().fg(Color::Rgb(106, 204, 142)));
    frame.render_widget(spark, rows[1]);
}
