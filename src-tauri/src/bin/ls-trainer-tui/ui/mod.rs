//! UI module root and shared rendering helpers for `ls-trainer-tui`.

pub mod launcher;
pub mod splash;
pub mod training;
pub mod widgets;

use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
};

use crate::app::{
    AppState, BAR_TRACK_BG, GAUGE_TEXT_COVERED, GAUGE_TEXT_UNCOVERED, OverlayMode, UiMode,
};

pub fn draw_ui(frame: &mut Frame, state: &AppState) {
    if state.mode == UiMode::Setup {
        launcher::render_launcher_screen(frame, frame.area(), state);
        match state.overlay {
            OverlayMode::Help => widgets::help::render_help_overlay(frame, frame.area()),
            OverlayMode::Config => {
                widgets::config::render_config_overlay(frame, frame.area(), state)
            }
            OverlayMode::History => {
                widgets::config::render_history_overlay(frame, frame.area(), state)
            }
            OverlayMode::Browse => launcher::render_browse_overlay(frame, frame.area(), state),
            OverlayMode::None => {}
        }
        return;
    }

    training::render_training_screen(frame, frame.area(), state);
}

pub fn format_eta_seconds(eta_secs: f64) -> String {
    if !eta_secs.is_finite() || eta_secs <= 0.0 {
        return "--:--".to_string();
    }
    let total = eta_secs.round() as u64;
    let hours = total / 3600;
    let mins = (total % 3600) / 60;
    let secs = total % 60;
    if hours > 0 {
        format!("{hours:02}:{mins:02}:{secs:02}")
    } else {
        format!("{mins:02}:{secs:02}")
    }
}

pub fn compact_display_path(path: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let count = path.chars().count();
    if count <= max_chars {
        return path.to_string();
    }
    for marker in ["runs\\", "runs/"] {
        if let Some(pos) = path.find(marker) {
            let rel = &path[pos..];
            if rel.chars().count() <= max_chars {
                return rel.to_string();
            }
        }
    }
    let keep = max_chars.saturating_sub(1);
    let tail: String = path
        .chars()
        .rev()
        .take(keep)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("…{tail}")
}

/// Render a 1-row (or multi-row) progress bar where label characters covered by the bar
/// are drawn in dark grey/black (`GAUGE_TEXT_COVERED`) and uncovered characters are drawn
/// in crisp light text (`GAUGE_TEXT_UNCOVERED`).
pub fn render_contrast_gauge(
    frame: &mut Frame,
    area: Rect,
    ratio: f64,
    label: &str,
    bar_color: Color,
) {
    if area.is_empty() {
        return;
    }
    let clamped = ratio.clamp(0.0, 1.0);
    let filled_width = (f64::from(area.width) * clamped).round() as u16;
    let filled_end = area.left() + filled_width.min(area.width);

    let buf = frame.buffer_mut();
    for y in area.top()..area.bottom() {
        for x in area.left()..area.right() {
            let covered = x < filled_end;
            let bg = if covered { bar_color } else { BAR_TRACK_BG };
            let fg = if covered {
                GAUGE_TEXT_COVERED
            } else {
                GAUGE_TEXT_UNCOVERED
            };
            buf[(x, y)].set_symbol(" ").set_fg(fg).set_bg(bg);
        }
    }

    let label_chars: Vec<char> = label.chars().collect();
    let label_len = (label_chars.len() as u16).min(area.width);
    let label_col = area.left() + (area.width.saturating_sub(label_len)) / 2;
    let label_row = area.top() + area.height / 2;

    for (i, ch) in label_chars.into_iter().take(label_len as usize).enumerate() {
        let x = label_col + i as u16;
        if x >= area.right() {
            break;
        }
        let covered = x < filled_end;
        let bg = if covered { bar_color } else { BAR_TRACK_BG };
        let fg = if covered {
            GAUGE_TEXT_COVERED
        } else {
            GAUGE_TEXT_UNCOVERED
        };
        let mut utf8_buf = [0u8; 4];
        let sym = ch.encode_utf8(&mut utf8_buf);
        buf[(x, label_row)]
            .set_symbol(sym)
            .set_style(Style::default().fg(fg).bg(bg).add_modifier(Modifier::BOLD));
    }
}

pub fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let popup_vert = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_vert[1])[1]
}
