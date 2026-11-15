//! Center Braille telemetry curves chart (`train_loss`, `val_loss`, `mAP@50:95`).

use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Modifier, Style},
    symbols,
    text::Span,
    widgets::{Axis, Block, BorderType, Borders, Chart, Dataset, GraphType},
};

use crate::app::{AppState, BORDER_SOFT, LABEL_MUTED, TITLE_ACCENT};

pub fn render_center_chart(frame: &mut Frame, area: Rect, state: &AppState) {
    let train_pts: Vec<(f64, f64)> = state.train_loss_history.iter().copied().collect();
    let val_pts: Vec<(f64, f64)> = state.val_loss_history.iter().copied().collect();
    let map_pts: Vec<(f64, f64)> = state.map50_95_history.iter().copied().collect();

    let mut min_x = 1.0_f64;
    let mut max_x = 10.0_f64;
    let mut min_y = 0.0_f64;
    let mut max_y = 1.0_f64;

    let all_pts = train_pts.iter().chain(val_pts.iter()).chain(map_pts.iter());

    let mut first = true;
    for &(x, y) in all_pts {
        if first {
            min_x = x;
            max_x = x;
            min_y = y.min(0.0);
            max_y = y;
            first = false;
        } else {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
    }
    if (max_x - min_x).abs() < 1.0 {
        max_x = min_x + 10.0;
    }
    if (max_y - min_y).abs() < 1e-4 {
        max_y = min_y + 1.0;
    } else {
        max_y *= 1.08;
    }

    let datasets = vec![
        Dataset::default()
            .name("train_loss")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Cyan))
            .data(&train_pts),
        Dataset::default()
            .name("val_loss")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Yellow))
            .data(&val_pts),
        Dataset::default()
            .name("mAP@50:95")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::LightGreen))
            .data(&map_pts),
    ];

    let mid_x = (min_x + max_x) * 0.5;
    let mid_y = (min_y + max_y) * 0.5;

    let chart = Chart::new(datasets)
        .block(
            Block::default()
                .title(Span::styled(
                    " Telemetry Curves (train_loss · val_loss · mAP@50:95) ",
                    Style::default()
                        .fg(TITLE_ACCENT)
                        .add_modifier(Modifier::BOLD),
                ))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(BORDER_SOFT)),
        )
        .x_axis(
            Axis::default()
                .title(Span::styled("Step", Style::default().fg(LABEL_MUTED)))
                .style(Style::default().fg(BORDER_SOFT))
                .bounds([min_x, max_x])
                .labels(vec![
                    Span::raw(format!("{min_x:.0}")),
                    Span::raw(format!("{mid_x:.0}")),
                    Span::raw(format!("{max_x:.0}")),
                ]),
        )
        .y_axis(
            Axis::default()
                .title(Span::styled("Value", Style::default().fg(LABEL_MUTED)))
                .style(Style::default().fg(BORDER_SOFT))
                .bounds([min_y, max_y])
                .labels(vec![
                    Span::raw(format!("{min_y:.2}")),
                    Span::raw(format!("{mid_y:.2}")),
                    Span::raw(format!("{max_y:.2}")),
                ]),
        );

    frame.render_widget(chart, area);
}
