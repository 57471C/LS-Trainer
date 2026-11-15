use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};

use crate::app::{AppState, HealthLevel, BAR_TRACK_BG, GAUGE_TEXT_COVERED};

pub fn render_health_status_bar(frame: &mut Frame, area: Rect, state: &AppState) {
    let (badge_text, badge_fg, badge_bg, detail_text, detail_color) = match state.health_level {
        HealthLevel::Healthy => (
            " ● Healthy ",
            GAUGE_TEXT_COVERED,
            Color::LightGreen,
            format!(" Training proceeding normally  │  {}", state.status_line),
            Color::Rgb(176, 214, 188),
        ),
        HealthLevel::Warning => {
            let tags = if state.warning_tags.is_empty() {
                "non-fatal".to_string()
            } else {
                state.warning_tags.join(", ")
            };
            let supp = if state.suppressed_warning_count > 0 {
                format!(", {} suppressed in log", state.suppressed_warning_count)
            } else {
                String::new()
            };
            (
                " ▲ Warning ",
                GAUGE_TEXT_COVERED,
                Color::Yellow,
                format!(
                    " {} non-fatal warning(s) [{}{}]  │  {}",
                    state.warning_count, tags, supp, state.status_line
                ),
                Color::Yellow,
            )
        }
        HealthLevel::Error => {
            let err_detail = state
                .last_error_msg
                .as_deref()
                .unwrap_or(&state.status_line);
            (
                " ✖ Error ",
                Color::White,
                Color::Red,
                format!(" {err_detail}"),
                Color::LightRed,
            )
        }
    };

    let line = Line::from(vec![
        Span::styled(
            badge_text,
            Style::default()
                .fg(badge_fg)
                .bg(badge_bg)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(detail_text, Style::default().fg(detail_color)),
    ]);
    frame.render_widget(
        Paragraph::new(line).style(Style::default().bg(BAR_TRACK_BG)),
        area,
    );
}
