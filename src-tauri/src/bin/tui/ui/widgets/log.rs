use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph},
    Frame,
};

use super::status::render_health_status_bar;
use crate::app::{
    APP_VERSION_LABEL, AppState, BAR_GPU_TEAL, BAR_RAM_SLATE, BAR_STEP_BLUE, BORDER_SOFT,
    GAUGE_TEXT_COVERED, MAX_LOG_LINES, TABLE_BORDER_SOFT, TITLE_ACCENT,
};

/// Map heavy or double box-drawing characters to softer light/rounded equivalents.
fn soften_box_char(ch: char) -> Option<char> {
    match ch {
        '┏' | '╔' | '┌' | '╭' => Some('╭'),
        '┓' | '╗' | '┐' | '╮' => Some('╮'),
        '┗' | '╚' | '└' | '╰' => Some('╰'),
        '┛' | '╝' | '┘' | '╯' => Some('╯'),
        '━' | '═' | '─' | '╌' | '╍' => Some('─'),
        '┃' | '║' | '│' | '╎' | '╏' => Some('│'),
        '┳' | '┯' | '╦' | '╥' | '╤' | '┬' => Some('┬'),
        '┻' | '┷' | '╩' | '╨' | '╧' | '┴' => Some('┴'),
        '┣' | '┡' | '┢' | '┠' | '┝' | '╞' | '╟' | '╠' | '├' => Some('├'),
        '┫' | '┩' | '┪' | '┨' | '┥' | '╡' | '╢' | '╣' | '┤' => Some('┤'),
        '╋' | '╇' | '╈' | '╉' | '╊' | '╀' | '╁' | '╂' | '┿' | '┽' | '┾' | '┼' | '╬' | '╪' | '╫' => {
            Some('┼')
        }
        c if ('\u{2500}'..='\u{257F}').contains(&c) => Some(c),
        _ => None,
    }
}

fn push_softened_box_spans(
    spans: &mut Vec<Span<'static>>,
    segment: &str,
    text_style: Style,
    soft_pipe_separators: bool,
) {
    if segment.is_empty() {
        return;
    }
    let border_style = Style::default().fg(TABLE_BORDER_SOFT);
    let mut cur = String::new();
    let mut cur_is_border = false;

    for ch in segment.chars() {
        let mapped_border = soften_box_char(ch).or_else(|| {
            if soft_pipe_separators && ch == '|' {
                Some('│')
            } else {
                None
            }
        });
        let is_border = mapped_border.is_some();
        let out_ch = mapped_border.unwrap_or(ch);

        if !cur.is_empty() && cur_is_border != is_border {
            let st = if cur_is_border {
                border_style
            } else {
                text_style
            };
            spans.push(Span::styled(std::mem::take(&mut cur), st));
        }
        cur_is_border = is_border;
        cur.push(out_ch);
    }

    if !cur.is_empty() {
        let st = if cur_is_border {
            border_style
        } else {
            text_style
        };
        spans.push(Span::styled(cur, st));
    }
}

/// Parse a log line containing optional ANSI SGR escape sequences (`\x1b[...m`) into colored `Span`s.
pub fn parse_ansi_or_styled_log_line(raw: &str) -> Line<'static> {
    if raw.contains('\x1b') {
        let mut spans: Vec<Span<'static>> = Vec::new();
        let mut current_style = Style::default().fg(Color::Gray);
        let mut segment = String::new();
        let mut chars = raw.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '\x1b' && chars.peek() == Some(&'[') {
                if !segment.is_empty() {
                    let seg = std::mem::take(&mut segment);
                    push_softened_box_spans(&mut spans, &seg, current_style, false);
                }
                chars.next(); // consume '['
                let mut code_str = String::new();
                while let Some(&nc) = chars.peek() {
                    chars.next();
                    if ('@'..='~').contains(&nc) {
                        if nc == 'm' {
                            current_style = apply_sgr_codes(&code_str, current_style);
                        }
                        break;
                    }
                    code_str.push(nc);
                }
            } else {
                segment.push(ch);
            }
        }
        if !segment.is_empty() {
            push_softened_box_spans(&mut spans, &segment, current_style, false);
        }
        return Line::from(spans);
    }

    let is_metrics_line = raw.contains("[LS-Trainer Metrics]") || raw.contains("Final Metrics");
    let base_color = if raw.contains("Error]") || raw.contains("Traceback") {
        Color::LightRed
    } else if raw.contains("Warning]") {
        Color::Yellow
    } else if is_metrics_line {
        Color::Rgb(132, 214, 158)
    } else if raw.starts_with("[LS-Trainer") {
        TITLE_ACCENT
    } else if raw.starts_with("Epoch ") {
        Color::Rgb(130, 182, 232)
    } else {
        Color::Rgb(194, 204, 216)
    };

    let mut spans: Vec<Span<'static>> = Vec::new();
    push_softened_box_spans(
        &mut spans,
        raw,
        Style::default().fg(base_color),
        is_metrics_line || raw.starts_with("Epoch "),
    );
    Line::from(spans)
}

fn apply_sgr_codes(code_str: &str, mut style: Style) -> Style {
    if code_str.is_empty() {
        return Style::default().fg(Color::Gray);
    }
    for part in code_str.split(';') {
        match part.trim().parse::<u16>().unwrap_or(0) {
            0 => style = Style::default().fg(Color::Gray),
            1 => style = style.add_modifier(Modifier::BOLD),
            2 => style = style.add_modifier(Modifier::DIM),
            30 => style = style.fg(Color::DarkGray),
            31 | 91 => style = style.fg(Color::LightRed),
            32 | 92 => style = style.fg(Color::LightGreen),
            33 | 93 => style = style.fg(Color::Yellow),
            34 | 94 => style = style.fg(Color::LightBlue),
            35 | 95 => style = style.fg(Color::LightMagenta),
            36 | 96 => style = style.fg(Color::Cyan),
            37 | 97 => style = style.fg(Color::White),
            _ => {}
        }
    }
    style
}

pub fn render_bottom_log_and_footer(frame: &mut Frame, area: Rect, state: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(3),
            Constraint::Length(1),
        ])
        .split(area);

    render_health_status_bar(frame, chunks[0], state);

    let log_block = Block::default()
        .title(Span::styled(
            format!(
                " Live Training Log ({}/{} lines) ",
                state.logs.len(),
                MAX_LOG_LINES
            ),
            Style::default()
                .fg(TITLE_ACCENT)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(BORDER_SOFT));

    let inner = log_block.inner(chunks[1]);
    frame.render_widget(log_block, chunks[1]);

    let padded = Rect {
        x: inner.x + 1,
        y: inner.y,
        width: inner.width.saturating_sub(2),
        height: inner.height,
    };

    let visible_height = padded.height as usize;
    let total_lines = state.logs.len();
    let max_offset = total_lines.saturating_sub(visible_height);
    let offset_from_bottom = state.log_scroll_offset.min(max_offset);
    let start_idx = total_lines.saturating_sub(visible_height + offset_from_bottom);
    let end_idx = (start_idx + visible_height).min(total_lines);

    let visible_lines: Vec<Line<'static>> = state
        .logs
        .iter()
        .skip(start_idx)
        .take(end_idx.saturating_sub(start_idx))
        .map(|l| parse_ansi_or_styled_log_line(l))
        .collect();

    frame.render_widget(Paragraph::new(visible_lines), padded);

    let pause_label = if state.is_paused { "Resume" } else { "Pause" };
    let footer = Line::from(vec![
        Span::styled(
            " [Space] ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!(" {pause_label} "),
            Style::default().fg(Color::White),
        ),
        Span::styled(
            " [c] ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::LightGreen)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Checkpoint ", Style::default().fg(Color::White)),
        Span::styled(
            " [↑/↓] ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Scroll ", Style::default().fg(Color::White)),
        Span::styled(
            " [F1/?] ",
            Style::default()
                .fg(GAUGE_TEXT_COVERED)
                .bg(BAR_GPU_TEAL)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Help ", Style::default().fg(Color::White)),
        Span::styled(
            " [F3] ",
            Style::default()
                .fg(GAUGE_TEXT_COVERED)
                .bg(BAR_STEP_BLUE)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Config ", Style::default().fg(Color::White)),
        Span::styled(
            " [F4/h] ",
            Style::default()
                .fg(GAUGE_TEXT_COVERED)
                .bg(BAR_RAM_SLATE)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" History ", Style::default().fg(Color::White)),
        Span::styled(
            " [F12] ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Copy Log ", Style::default().fg(Color::White)),
        Span::styled(
            " [q] ",
            Style::default()
                .fg(Color::White)
                .bg(Color::Red)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Stop & Quit ", Style::default().fg(Color::White)),
    ]);

    let footer_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(APP_VERSION_LABEL.len() as u16 + 2),
            Constraint::Min(10),
        ])
        .split(chunks[2]);

    frame.render_widget(
        Paragraph::new(Span::styled(
            format!(" {APP_VERSION_LABEL}"),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )),
        footer_chunks[0],
    );
    frame.render_widget(
        Paragraph::new(footer).alignment(Alignment::Center),
        footer_chunks[1],
    );
}
