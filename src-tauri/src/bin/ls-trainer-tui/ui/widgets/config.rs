use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Clear, Paragraph},
    Frame,
};

use crate::app::{
    AppState, BAR_GPU_TEAL, GAUGE_TEXT_COVERED, LABEL_MUTED, OVERLAY_BG, TABLE_BORDER_SOFT,
    TITLE_ACCENT, display_family_name,
};
use crate::ui::{centered_rect, compact_display_path};

fn weights_with_resume(scale: &str, resume_from: &str) -> String {
    match crate::app::resume_file_name(resume_from) {
        Some(name) => format!("{scale} · ft {name}"),
        None => scale.to_string(),
    }
}

pub fn render_config_overlay(frame: &mut Frame, area: Rect, state: &AppState) {
    let modal = centered_rect(76, 76, area);
    frame.render_widget(Clear, modal);

    let block = Block::default()
        .title(Span::styled(
            " Run Configuration Snapshot — Read-Only (Press Esc or F3 to close) ",
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

    let (arch, task, model, dataset, epochs, batch, imgsz, device, workers, run_name, aug_summary) =
        if let Some(ref req) = state.active_request {
            let w_str = req
                .workers
                .map(|w| w.to_string())
                .unwrap_or_else(|| "auto".to_string());
            let aug = format!(
                "amp={} · freeze_backbone={} · degrees={:.0} · flipud={:.2} · scale={:.2} · mosaic={:.2} · close_mosaic={}",
                req.amp,
                req.freeze_backbone,
                req.degrees,
                req.flipud,
                req.scale,
                req.mosaic,
                req.close_mosaic
            );
            (
                display_family_name(&req.model_family, &req.model_choice),
                req.task.clone(),
                weights_with_resume(&req.model_choice, &req.resume_from),
                req.dataset_path.clone(),
                req.epochs.to_string(),
                req.batch.to_string(),
                req.imgsz.to_string(),
                req.device.clone(),
                w_str,
                req.run_name.clone(),
                aug,
            )
        } else {
            let s = &state.setup;
            let resolved_model = s.arch.resolve_model_or_scale(&s.model);
            let task = if resolved_model.to_ascii_lowercase().contains("-seg")
                || s.resume_from.to_ascii_lowercase().contains("-seg")
            {
                "segment".to_string()
            } else {
                "detect".to_string()
            };
            let aug = format!(
                "amp={} · freeze_backbone={} · degrees={} · flipud={} · scale={} · translate={} · mosaic={} · close_mosaic={} · mixup={} · copy_paste={}",
                s.amp,
                s.freeze_backbone,
                s.degrees,
                s.flipud,
                s.scale,
                s.translate,
                s.mosaic,
                s.close_mosaic,
                s.mixup,
                if s.copy_paste { "0.3" } else { "0.0" }
            );
            (
                s.arch.display_label().to_string(),
                task,
                weights_with_resume(&resolved_model, &s.resume_from),
                s.dataset.clone(),
                s.epochs.clone(),
                s.batch.clone(),
                s.resolve_imgsz().unwrap_or(s.imgsz).to_string(),
                s.device.clone(),
                s.workers_input.clone(),
                s.run_name.clone(),
                aug,
            )
        };

    let run_dir = state
        .results_dir
        .clone()
        .unwrap_or_else(|| format!("runs\\{task}\\{run_name}"));
    let best_wts = state
        .best_weights_path
        .clone()
        .unwrap_or_else(|| "pending".to_string());
    let onnx_file = state
        .onnx_path
        .clone()
        .unwrap_or_else(|| "pending (auto-export on completion)".to_string());
    let train_imgs = state
        .dataset_train_images
        .map(|n| format!("{n} images"))
        .unwrap_or_else(|| "auto-detected".to_string());

    let kv_rows: [(&str, String); 13] = [
        ("Architecture", arch),
        ("Task", task),
        ("Model / Weights", model),
        ("Dataset Path", dataset),
        ("Train Split Size", train_imgs),
        ("Epochs", epochs),
        ("Batch Size", batch),
        ("Image Size (imgsz)", imgsz),
        ("Device", device),
        ("DataLoader Workers", workers),
        ("Run Directory", run_dir),
        ("Best Checkpoint", best_wts),
        ("ONNX Export", onnx_file),
    ];

    let mut lines: Vec<Line<'static>> = Vec::new();
    for (k, v) in kv_rows {
        lines.push(Line::from(vec![
            Span::styled(
                format!("{k:<20} │ "),
                Style::default()
                    .fg(LABEL_MUTED)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(v, Style::default().fg(Color::White)),
        ]));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled(
            format!("{:<20} │ ", "Hyperparameters"),
            Style::default()
                .fg(LABEL_MUTED)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(aug_summary, Style::default().fg(TITLE_ACCENT)),
    ]));

    frame.render_widget(Paragraph::new(lines), padded);
}

pub fn render_history_overlay(frame: &mut Frame, area: Rect, state: &AppState) {
    let modal = centered_rect(82, 78, area);
    frame.render_widget(Clear, modal);

    let block = Block::default()
        .title(Span::styled(
            " Recent Training Runs (↑/↓ Select · Enter/y Copy Path · Esc/F4/h Close) ",
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

    if state.history_entries.is_empty() {
        let msg = Paragraph::new("No previous runs found in runs/detect or runs/segment.")
            .style(Style::default().fg(LABEL_MUTED));
        frame.render_widget(msg, padded);
        return;
    }

    let split = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(6), Constraint::Length(6)])
        .split(padded);

    let mut list_lines: Vec<Line<'static>> = Vec::new();
    list_lines.push(Line::from(vec![Span::styled(
        format!(
            "   {:<20} {:<24} {:<12} {:<12} {:<10}",
            "Run Name", "Architecture / Model", "mAP50", "mAP50-95", "Updated"
        ),
        Style::default()
            .fg(TITLE_ACCENT)
            .add_modifier(Modifier::BOLD),
    )]));

    let max_rows = split[0].height.saturating_sub(1) as usize;
    let sel = state
        .history_selected
        .min(state.history_entries.len().saturating_sub(1));
    let start_idx = if sel >= max_rows && max_rows > 0 {
        sel + 1 - max_rows
    } else {
        0
    };

    for (idx, entry) in state
        .history_entries
        .iter()
        .enumerate()
        .skip(start_idx)
        .take(max_rows.max(1))
    {
        let is_sel = idx == sel;
        let marker = if is_sel { " ▶ " } else { "   " };
        let m50_s = entry
            .map50
            .map(|v| format!("{:.2}%", v * 100.0))
            .unwrap_or_else(|| "--".to_string());
        let m95_s = entry
            .map50_95
            .map(|v| format!("{:.2}%", v * 100.0))
            .unwrap_or_else(|| "--".to_string());
        let row_text = format!(
            "{marker}{:<20} {:<24} {:<12} {:<12} {:<10}",
            compact_display_path(&entry.run_name, 19),
            compact_display_path(&entry.arch_model, 23),
            m50_s,
            m95_s,
            entry.timestamp_label
        );
        let style = if is_sel {
            Style::default()
                .fg(GAUGE_TEXT_COVERED)
                .bg(BAR_GPU_TEAL)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        };
        list_lines.push(Line::from(Span::styled(row_text, style)));
    }
    frame.render_widget(Paragraph::new(list_lines), split[0]);

    if let Some(entry) = state.history_entries.get(sel) {
        let detail_lines = vec![
            Line::from(Span::styled(
                "── Selected Run Summary ──────────────────────────────────────────────────",
                Style::default().fg(TABLE_BORDER_SOFT),
            )),
            Line::from(vec![
                Span::styled("Run Directory : ", Style::default().fg(LABEL_MUTED)),
                Span::styled(entry.run_dir.clone(), Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("Best Weights  : ", Style::default().fg(LABEL_MUTED)),
                Span::styled(
                    entry
                        .best_weights
                        .clone()
                        .unwrap_or_else(|| "Not saved".to_string()),
                    Style::default().fg(Color::LightGreen),
                ),
            ]),
            Line::from(vec![
                Span::styled("ONNX Model    : ", Style::default().fg(LABEL_MUTED)),
                Span::styled(
                    entry
                        .onnx_path
                        .clone()
                        .unwrap_or_else(|| "Not exported".to_string()),
                    Style::default().fg(TITLE_ACCENT),
                ),
            ]),
            Line::from(vec![
                Span::styled("Task / Age    : ", Style::default().fg(LABEL_MUTED)),
                Span::styled(
                    format!("{} · {}", entry.task, entry.timestamp_label),
                    Style::default().fg(Color::White),
                ),
            ]),
        ];
        frame.render_widget(Paragraph::new(detail_lines), split[1]);
    }
}
