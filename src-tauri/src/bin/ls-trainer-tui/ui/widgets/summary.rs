//! Host / CPU card and Run Summary panel (progress bars, metrics, new-best badge, live output paths).

use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph, Sparkline},
};

use crate::app::{
    AppState, BAR_EPOCH_TEAL, BAR_GPU_TEAL, BAR_OVERALL_CYAN, BAR_RAM_SLATE, BAR_STEP_BLUE,
    BORDER_SOFT, LABEL_MUTED, NEW_BEST_BADGE_BG, NEW_BEST_GOLD, TABLE_BORDER_SOFT, TITLE_ACCENT,
    display_family_name,
};
use crate::ui::{compact_display_path, format_eta_seconds, render_contrast_gauge};

pub fn render_run_summary(frame: &mut Frame, area: Rect, state: &AppState) {
    let host_h = if area.height >= 25 {
        area.height.saturating_sub(14).clamp(11, 13)
    } else if area.height >= 19 {
        area.height.saturating_sub(14).max(7)
    } else {
        4
    };
    let split = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(host_h), Constraint::Min(10)])
        .split(area);

    render_host_cpu_panel(frame, split[0], state);
    render_run_details_panel(frame, split[1], state);
}

pub fn render_host_cpu_panel(frame: &mut Frame, area: Rect, state: &AppState) {
    let block = Block::default()
        .title(Span::styled(
            " Host / CPU ",
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
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(1),
        ])
        .split(padded);

    let cpu_ratio = (state.host_cpu_percent / 100.0).clamp(0.0, 1.0);
    let cores_label = if state.host_core_count > 0 {
        format!(" ({} cores)", state.host_core_count)
    } else {
        String::new()
    };
    let cpu_label = format!("CPU {:.1}%{cores_label}", state.host_cpu_percent);
    render_contrast_gauge(frame, rows[0], cpu_ratio, &cpu_label, BAR_GPU_TEAL);

    let ram_ratio = if state.host_ram_total_gb > 0.0 {
        (state.host_ram_used_gb / state.host_ram_total_gb).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let ram_pct = ram_ratio * 100.0;
    let ram_label = format!(
        "RAM {:.1} / {:.1} GiB ({ram_pct:.0}%)",
        state.host_ram_used_gb, state.host_ram_total_gb
    );
    render_contrast_gauge(frame, rows[1], ram_ratio, &ram_label, BAR_RAM_SLATE);

    if rows[2].height > 0 {
        let bottom_cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Length(18), Constraint::Min(4)])
            .split(rows[2]);
        let (l1, l5, _l15) = state.host_load_avg;
        let load_line = Line::from(vec![
            Span::styled("Load ", Style::default().fg(LABEL_MUTED)),
            Span::styled(format!("{l1:.1} {l5:.1}"), Style::default().fg(Color::White)),
        ]);
        frame.render_widget(Paragraph::new(load_line), bottom_cols[0]);

        let cpu_samples: Vec<u64> = state.host_cpu_history.iter().copied().collect();
        let sparkline = Sparkline::default()
            .data(&cpu_samples)
            .max(100)
            .style(Style::default().fg(BAR_GPU_TEAL));
        frame.render_widget(sparkline, bottom_cols[1]);
    }
}

pub fn render_run_details_panel(frame: &mut Frame, area: Rect, state: &AppState) {
    let block = Block::default()
        .title(Span::styled(
            " Run Summary ",
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
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(4),
        ])
        .split(padded);

    let m = &state.latest_metrics;
    let total_ep = m.total_epochs.max(1);
    let ep_ratio = if m.total_epochs > 0 {
        (m.epoch as f64 / total_ep as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let ep_label = format!(
        "Epoch {} / {} ({:.0}%)",
        m.epoch,
        m.total_epochs,
        ep_ratio * 100.0
    );
    render_contrast_gauge(frame, rows[0], ep_ratio, &ep_label, BAR_EPOCH_TEAL);

    let step_ratio = if m.total_steps > 0 {
        (m.step as f64 / m.total_steps as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let step_label = format!(
        "Step {} / {} ({:.0}%)",
        m.step,
        m.total_steps,
        step_ratio * 100.0
    );
    render_contrast_gauge(frame, rows[1], step_ratio, &step_label, BAR_STEP_BLUE);

    let (arch_str, model_str, _dataset_str, dev_str, batch_val) =
        if let Some(ref req) = state.active_request {
            (
                display_family_name(&req.model_family, &req.model_choice),
                req.model_choice.as_str(),
                req.dataset_path.as_str(),
                req.device.as_str(),
                req.batch.max(1),
            )
        } else {
            let b = state.setup.batch.trim().parse::<u32>().unwrap_or(4).max(1);
            (
                state.setup.arch.display_label().to_string(),
                state.setup.model.as_str(),
                state.setup.dataset.as_str(),
                state.setup.device.as_str(),
                b,
            )
        };

    // Overall progress across epochs + steps + samples/images
    let overall_ratio = if !state.is_running && state.exit_code == Some(0) {
        1.0
    } else if m.total_epochs > 0 && m.total_steps > 0 && m.epoch > 0 {
        let done_steps = m
            .epoch
            .saturating_sub(1)
            .saturating_mul(m.total_steps)
            .saturating_add(m.step.min(m.total_steps));
        let all_steps = total_ep.saturating_mul(m.total_steps).max(1);
        (done_steps as f64 / all_steps as f64).clamp(0.0, 1.0)
    } else {
        ep_ratio
    };

    let imgs_per_epoch = state.dataset_train_images.or_else(|| {
        if m.total_steps > 0 {
            let dev_mult = state.gpus.len().max(1) as u32;
            Some(
                m.total_steps
                    .saturating_mul(batch_val)
                    .saturating_mul(dev_mult),
            )
        } else {
            None
        }
    });
    let samples_str = if let Some(per_ep) = imgs_per_epoch {
        let total_samples = per_ep.saturating_mul(total_ep);
        let done_samples =
            ((overall_ratio * f64::from(total_samples)).round() as u32).min(total_samples);
        format!("{done_samples}/{total_samples} imgs")
    } else {
        "-- imgs".to_string()
    };
    let overall_label = format!(
        "Overall {:.0}% · Ep {}/{} · Step {}/{} · {}",
        overall_ratio * 100.0,
        m.epoch,
        m.total_epochs,
        m.step,
        m.total_steps,
        samples_str
    );
    render_contrast_gauge(
        frame,
        rows[2],
        overall_ratio,
        &overall_label,
        BAR_OVERALL_CYAN,
    );

    let status_color = if state.is_paused {
        Color::Yellow
    } else if state.is_running {
        Color::LightGreen
    } else {
        TITLE_ACCENT
    };
    let spinner_frames = ["|", "/", "-", "\\"];
    let spinner_char = spinner_frames[state.spinner_tick % spinner_frames.len()];
    let status_badge = if state.is_paused {
        "PAUSED".to_string()
    } else if state.is_running {
        format!("TRAINING {spinner_char}")
    } else {
        state.status_line.clone()
    };

    let elapsed_str = match (state.training_started_at, state.training_ended_at) {
        (Some(start), Some(end)) => {
            format_eta_seconds(end.saturating_duration_since(start).as_secs_f64())
        }
        (Some(start), None) => format_eta_seconds(start.elapsed().as_secs_f64()),
        _ => "--:--".to_string(),
    };

    let map50_str = state
        .best_map50
        .map(|v| format!("{:.2}%", v * 100.0))
        .unwrap_or_else(|| "--".to_string());
    let map95_str = state
        .best_map50_95
        .map(|v| format!("{:.2}%", v * 100.0))
        .unwrap_or_else(|| "--".to_string());

    let new_best_active = state.is_new_best_active();
    let best_metric_color = if new_best_active {
        NEW_BEST_GOLD
    } else {
        Color::LightGreen
    };

    let mut best_map_spans = vec![
        Span::styled("Best mAP ", Style::default().fg(LABEL_MUTED)),
        Span::styled("50 ", Style::default().fg(LABEL_MUTED)),
        Span::styled(
            format!("{map50_str:<7}"),
            Style::default()
                .fg(best_metric_color)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" │ ", Style::default().fg(TABLE_BORDER_SOFT)),
        Span::styled("50:95 ", Style::default().fg(LABEL_MUTED)),
        Span::styled(
            map95_str,
            Style::default()
                .fg(best_metric_color)
                .add_modifier(Modifier::BOLD),
        ),
    ];
    if new_best_active {
        best_map_spans.push(Span::raw(" "));
        best_map_spans.push(Span::styled(
            " ★ New best ",
            Style::default()
                .fg(NEW_BEST_GOLD)
                .bg(NEW_BEST_BADGE_BG)
                .add_modifier(Modifier::BOLD),
        ));
    }

    let path_budget = padded.width.saturating_sub(9) as usize;
    let run_dir_disp = state
        .results_dir
        .as_deref()
        .map(|p| compact_display_path(p, path_budget))
        .unwrap_or_else(|| "pending...".to_string());
    let best_wts_disp = state
        .best_weights_path
        .as_deref()
        .map(|p| compact_display_path(p, path_budget))
        .unwrap_or_else(|| "pending (saving on epoch)...".to_string());
    let onnx_disp = state
        .onnx_path
        .as_deref()
        .map(|p| compact_display_path(p, path_budget))
        .unwrap_or_else(|| {
            if state.status_line.to_ascii_lowercase().contains("onnx") {
                "exporting best.onnx...".to_string()
            } else {
                "pending (exports on finish)".to_string()
            }
        });

    let sep_span = Span::styled("  │  ", Style::default().fg(TABLE_BORDER_SOFT));

    let lines = vec![
        Line::from(vec![
            Span::styled("State    ", Style::default().fg(LABEL_MUTED)),
            Span::styled(
                format!("{status_badge:<14}"),
                Style::default()
                    .fg(status_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("Dev ", Style::default().fg(LABEL_MUTED)),
            Span::styled(dev_str.to_string(), Style::default().fg(TITLE_ACCENT)),
        ]),
        Line::from(vec![
            Span::styled("Model    ", Style::default().fg(LABEL_MUTED)),
            Span::styled(
                format!("{arch_str} ({model_str})"),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled("Loss     ", Style::default().fg(LABEL_MUTED)),
            Span::styled(
                format!("{:<8.4}", m.train_loss),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            sep_span.clone(),
            Span::styled("LR ", Style::default().fg(LABEL_MUTED)),
            Span::styled(
                format!("{:.2e}", m.lr),
                Style::default()
                    .fg(Color::LightYellow)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(best_map_spans),
        Line::from(vec![
            Span::styled("Speed    ", Style::default().fg(LABEL_MUTED)),
            Span::styled(
                format!("{:>5.1} samples/s", m.samples_per_sec),
                Style::default()
                    .fg(Color::LightGreen)
                    .add_modifier(Modifier::BOLD),
            ),
            sep_span.clone(),
            Span::styled("Elapsed ", Style::default().fg(LABEL_MUTED)),
            Span::styled(elapsed_str, Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("ETA      ", Style::default().fg(LABEL_MUTED)),
            Span::styled(
                format!("{:<8}", format_eta_seconds(m.eta_secs)),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            sep_span,
            Span::styled("Samples ", Style::default().fg(LABEL_MUTED)),
            Span::styled(samples_str, Style::default().fg(TITLE_ACCENT)),
        ]),
        Line::from(vec![
            Span::styled("Run Dir  ", Style::default().fg(LABEL_MUTED)),
            Span::styled(
                run_dir_disp,
                Style::default().fg(Color::Rgb(196, 206, 218)),
            ),
        ]),
        Line::from(vec![
            Span::styled("Best Wts ", Style::default().fg(LABEL_MUTED)),
            Span::styled(
                best_wts_disp,
                Style::default().fg(if state.best_weights_path.is_some() {
                    Color::LightGreen
                } else {
                    LABEL_MUTED
                }),
            ),
        ]),
        Line::from(vec![
            Span::styled("ONNX     ", Style::default().fg(LABEL_MUTED)),
            Span::styled(
                onnx_disp,
                Style::default().fg(if state.onnx_path.is_some() {
                    TITLE_ACCENT
                } else {
                    LABEL_MUTED
                }),
            ),
        ]),
    ];

    frame.render_widget(Paragraph::new(lines), rows[3]);
}
