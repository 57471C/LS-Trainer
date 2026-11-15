use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Clear, Paragraph},
    Frame,
};

use crate::app::{OVERLAY_BG, TABLE_BORDER_SOFT, TITLE_ACCENT};
use crate::ui::centered_rect;

pub fn render_help_overlay(frame: &mut Frame, area: Rect) {
    let modal = centered_rect(72, 74, area);
    frame.render_widget(Clear, modal);

    let block = Block::default()
        .title(Span::styled(
            " Keyboard Shortcuts Help (Press Esc, F1, or ? to close) ",
            Style::default()
                .fg(TITLE_ACCENT)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(TITLE_ACCENT))
        .style(Style::default().bg(OVERLAY_BG));

    let inner = block.inner(modal);
    frame.render_widget(block, modal);
    if inner.height == 0 || inner.width <= 4 {
        return;
    }

    let padded = Rect {
        x: inner.x + 2,
        y: inner.y + 1,
        width: inner.width.saturating_sub(4),
        height: inner.height.saturating_sub(1),
    };

    let rows: [(&str, &str); 11] = [
        ("Key", "Action"),
        ("Space", "Pause / Resume active training process"),
        ("c", "Request immediate checkpoint save (CHECKPOINT)"),
        (
            "Up / Down",
            "Scroll live training log (scroll to bottom to auto-follow)",
        ),
        ("F1 / ?", "Toggle this Keyboard Shortcuts Help overlay"),
        (
            "F3",
            "Toggle Run Configuration Snapshot overlay (read-only)",
        ),
        (
            "F4 / h",
            "Toggle Recent Runs History overlay (view metrics & paths)",
        ),
        ("F12", "Copy live training log lines to system clipboard"),
        ("q / Ctrl+C", "Graceful stop, print exit summary, and quit"),
        (
            "Tab / Arrows",
            "Navigate fields and cycle presets in Launch Menu",
        ),
        ("F2 / Enter", "Start training from Launch Menu"),
    ];

    let mut lines: Vec<Line<'static>> = Vec::new();
    lines.push(Line::from(Span::styled(
        "╭──────────────────┬──────────────────────────────────────────────────────────╮",
        Style::default().fg(TABLE_BORDER_SOFT),
    )));
    for (idx, (k, desc)) in rows.iter().enumerate() {
        let key_style = if idx == 0 {
            Style::default()
                .fg(TITLE_ACCENT)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default()
                .fg(Color::LightGreen)
                .add_modifier(Modifier::BOLD)
        };
        let desc_style = if idx == 0 {
            Style::default()
                .fg(TITLE_ACCENT)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        };
        lines.push(Line::from(vec![
            Span::styled("│ ", Style::default().fg(TABLE_BORDER_SOFT)),
            Span::styled(format!("{k:<16} "), key_style),
            Span::styled("│ ", Style::default().fg(TABLE_BORDER_SOFT)),
            Span::styled(format!("{desc:<56} "), desc_style),
            Span::styled("│", Style::default().fg(TABLE_BORDER_SOFT)),
        ]));
        if idx == 0 {
            lines.push(Line::from(Span::styled(
                "├──────────────────┼──────────────────────────────────────────────────────────┤",
                Style::default().fg(TABLE_BORDER_SOFT),
            )));
        }
    }
    lines.push(Line::from(Span::styled(
        "╰──────────────────┴──────────────────────────────────────────────────────────╯",
        Style::default().fg(TABLE_BORDER_SOFT),
    )));

    frame.render_widget(Paragraph::new(lines), padded);
}
