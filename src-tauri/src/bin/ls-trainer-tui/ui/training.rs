//! Composes the main training TUI layout and active modal overlays.

use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
};

use crate::app::{AppState, OverlayMode};
use crate::ui::widgets::{
    config::{render_config_overlay, render_history_overlay},
    gpu::render_top_gpu_strip,
    help::render_help_overlay,
    log::render_bottom_log_and_footer,
    summary::render_run_summary,
    telemetry::render_center_chart,
};

pub fn render_training_screen(frame: &mut Frame, area: Rect, state: &AppState) {
    let total_h = area.height;
    let top_h = 5u16;
    let rem_h = total_h.saturating_sub(top_h);
    let min_center_h = 26u16.min(rem_h.saturating_sub(5));
    let prop_center_h = ((u32::from(rem_h) * 70) / 100) as u16;
    let center_h = prop_center_h.max(min_center_h);

    let root_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(top_h),
            Constraint::Length(center_h),
            Constraint::Min(5),
        ])
        .split(area);

    render_top_gpu_strip(frame, root_chunks[0], state);

    let center_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(root_chunks[1]);

    render_center_chart(frame, center_chunks[0], state);
    render_run_summary(frame, center_chunks[1], state);

    render_bottom_log_and_footer(frame, root_chunks[2], state);

    match state.overlay {
        OverlayMode::Help => render_help_overlay(frame, area),
        OverlayMode::Config => render_config_overlay(frame, area, state),
        OverlayMode::History => render_history_overlay(frame, area, state),
        OverlayMode::Browse | OverlayMode::None => {}
    }
}
