//! Full-screen interactive Launch Menu (`ls-trainer-tui` started with no CLI flags).

use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph},
};

use crate::app::{
    AppState, BAR_GPU_TEAL, BAR_STEP_BLUE, BAR_TRACK_BG, BORDER_SOFT, GAUGE_TEXT_COVERED,
    LABEL_MUTED, TABLE_BORDER_SOFT, TITLE_ACCENT,
};
use crate::ui::widgets::gpu::render_top_gpu_strip;

pub fn render_launcher_screen(frame: &mut Frame, area: Rect, state: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),
            Constraint::Min(12),
            Constraint::Length(1),
        ])
        .split(area);

    render_top_gpu_strip(frame, chunks[0], state);

    let block = Block::default()
        .title(Span::styled(
            " LS-Trainer TUI — Launch Menu ",
            Style::default()
                .fg(TITLE_ACCENT)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(BORDER_SOFT));

    let inner = block.inner(chunks[1]);
    frame.render_widget(block, chunks[1]);

    if inner.height > 0 && inner.width > 2 {
        let padded = Rect {
            x: inner.x + 2,
            y: inner.y + 1,
            width: inner.width.saturating_sub(4),
            height: inner.height.saturating_sub(1),
        };

        let s = &state.setup;
        let ds_display = if s.dataset.is_empty() && !s.editing_field {
            "<enter dataset path>".to_string()
        } else {
            s.dataset.clone()
        };
        let workers_display = if s.workers_input.trim().is_empty() && !s.editing_field {
            "auto".to_string()
        } else {
            s.workers_input.clone()
        };

        let fields: [(&str, String, &str, bool); 6] = [
            (
                "Architecture",
                format!(
                    "◀ {} ({}) · {} ▶",
                    s.arch.as_family_str(),
                    s.arch.as_cli_str(),
                    s.model
                ),
                "Left/Right or Enter to switch yolo | rf-detr | rt-detr",
                false,
            ),
            (
                "Dataset Path",
                ds_display,
                "Enter to edit YOLO dataset folder/archive or COCO export path",
                true,
            ),
            (
                "Epochs",
                s.epochs.clone(),
                "Total training epochs (Enter to edit)",
                true,
            ),
            (
                "Batch Size",
                s.batch.clone(),
                "Batch size per training step (Enter to edit)",
                true,
            ),
            (
                "Device",
                s.device.clone(),
                "Left/Right to cycle (auto, xpu, xpu:0,1, cuda, cpu) or Enter to edit",
                true,
            ),
            (
                "Workers",
                workers_display,
                "Left/Right to cycle (auto, 0, 1, 2, 4, 8) or Enter to edit",
                true,
            ),
        ];

        let mut lines: Vec<Line<'static>> = Vec::new();
        lines.push(Line::from(vec![
            Span::styled(
                "Configure your training run below. Press ",
                Style::default().fg(LABEL_MUTED),
            ),
            Span::styled(
                "Enter",
                Style::default()
                    .fg(TITLE_ACCENT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" to edit a field, or ", Style::default().fg(LABEL_MUTED)),
            Span::styled(
                "F2",
                Style::default()
                    .fg(Color::LightGreen)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                " / Enter on Start to begin.",
                Style::default().fg(LABEL_MUTED),
            ),
        ]));
        lines.push(Line::from(""));

        for (idx, (label, val, hint, editable)) in fields.iter().enumerate() {
            let focused = s.focused_field == idx;
            let is_editing = focused && s.editing_field && *editable;
            let prefix = if focused { " ▶ " } else { "   " };
            let label_style = if focused {
                Style::default()
                    .fg(TITLE_ACCENT)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(LABEL_MUTED)
            };
            let val_style = if is_editing {
                Style::default()
                    .fg(GAUGE_TEXT_COVERED)
                    .bg(Color::LightYellow)
                    .add_modifier(Modifier::BOLD)
            } else if focused {
                Style::default()
                    .fg(GAUGE_TEXT_COVERED)
                    .bg(BAR_GPU_TEAL)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White).bg(BAR_TRACK_BG)
            };

            let cursor = if is_editing { "▏" } else { "" };
            let edit_tag = if is_editing { " [EDITING]" } else { "" };

            lines.push(Line::from(vec![
                Span::styled(prefix, label_style),
                Span::styled(format!("{label:<16} "), label_style),
                Span::styled(format!(" {val}{cursor} "), val_style),
                Span::styled(
                    edit_tag.to_string(),
                    Style::default()
                        .fg(Color::LightYellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(format!("   {hint}"), Style::default().fg(TABLE_BORDER_SOFT)),
            ]));
            lines.push(Line::from(""));
        }

        let start_focused = s.focused_field == 6;
        let start_prefix = if start_focused { " ▶ " } else { "   " };
        let start_btn_style = if start_focused {
            Style::default()
                .fg(GAUGE_TEXT_COVERED)
                .bg(Color::LightGreen)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default()
                .fg(Color::LightGreen)
                .bg(BAR_TRACK_BG)
                .add_modifier(Modifier::BOLD)
        };
        lines.push(Line::from(vec![
            Span::styled(
                start_prefix,
                Style::default()
                    .fg(Color::LightGreen)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("   [ Start Training (Enter / F2) ]   ", start_btn_style),
            Span::styled(
                "   Launch training and switch to live TUI",
                Style::default().fg(LABEL_MUTED),
            ),
        ]));

        if let Some(ref err) = s.error_banner {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                format!("   ⚠ {err}"),
                Style::default()
                    .fg(Color::LightRed)
                    .add_modifier(Modifier::BOLD),
            )));
        }

        frame.render_widget(Paragraph::new(lines), padded);
    }

    let footer = Line::from(vec![
        Span::styled(
            " [↑/↓/Tab] ",
            Style::default()
                .fg(GAUGE_TEXT_COVERED)
                .bg(BAR_GPU_TEAL)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Select Field  ", Style::default().fg(Color::White)),
        Span::styled(
            " [Enter] ",
            Style::default()
                .fg(GAUGE_TEXT_COVERED)
                .bg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Edit / Confirm  ", Style::default().fg(Color::White)),
        Span::styled(
            " [←/→] ",
            Style::default()
                .fg(GAUGE_TEXT_COVERED)
                .bg(BAR_STEP_BLUE)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Cycle Option  ", Style::default().fg(Color::White)),
        Span::styled(
            " [F2] ",
            Style::default()
                .fg(GAUGE_TEXT_COVERED)
                .bg(Color::LightGreen)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Start Training  ", Style::default().fg(Color::White)),
        Span::styled(
            " [Esc] ",
            Style::default()
                .fg(Color::White)
                .bg(Color::Red)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Quit ", Style::default().fg(Color::White)),
    ]);

    frame.render_widget(
        Paragraph::new(footer).alignment(Alignment::Center),
        chunks[2],
    );
}
