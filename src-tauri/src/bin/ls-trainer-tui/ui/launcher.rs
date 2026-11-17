//! Full-screen interactive Launch Menu (`ls-trainer-tui` started with no CLI flags).

use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph},
};

use crate::app::{
    APP_VERSION_LABEL, AppState, BAR_GPU_TEAL, BAR_STEP_BLUE, BAR_TRACK_BG, BORDER_SOFT,
    GAUGE_TEXT_COVERED, LABEL_MUTED, TABLE_BORDER_SOFT, TITLE_ACCENT,
};
use crate::ui::widgets::gpu::render_top_gpu_strip;

fn format_field_line(
    idx: usize,
    focused_field: usize,
    editing_field: bool,
    label: &str,
    val: &str,
    editable: bool,
    hint: Option<&str>,
    label_width: usize,
) -> Line<'static> {
    let focused = focused_field == idx;
    let is_editing = focused && editing_field && editable;
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
    let edit_tag = if is_editing { " [EDIT]" } else { "" };

    let mut spans = vec![
        Span::styled(prefix, label_style),
        Span::styled(format!("{label:<width$} ", width = label_width), label_style),
        Span::styled(format!(" {val}{cursor} "), val_style),
    ];
    if !edit_tag.is_empty() {
        spans.push(Span::styled(
            edit_tag.to_string(),
            Style::default()
                .fg(Color::LightYellow)
                .add_modifier(Modifier::BOLD),
        ));
    }
    if let Some(h) = hint {
        if !h.is_empty() {
            spans.push(Span::styled(
                format!("  {h}"),
                Style::default().fg(TABLE_BORDER_SOFT),
            ));
        }
    }
    Line::from(spans)
}

fn focused_field_help(state: &AppState) -> String {
    let s = &state.setup;
    match s.focused_field {
        0 => "Architecture family: Left/Right or Enter to switch YOLO11 | YOLO26 | RF-DETR | RT-DETR".to_string(),
        1 => {
            let family = s.arch.display_label();
            let cycle = match s.arch {
                crate::app::ArchChoice::Yolo | crate::app::ArchChoice::Yolo26 => {
                    "Left/Right cycles n/s/m/l/x"
                }
                _ => "Left/Right to cycle scales, Enter to type a scale or custom weights",
            };
            if let Some(opt) = s.arch.find_option(&s.model) {
                format!(
                    "{family}: {} ({}) — {} | {cycle}",
                    opt.label, opt.id, opt.desc
                )
            } else {
                format!(
                    "{family}: custom model '{}' — {cycle}",
                    s.model
                )
            }
        }
        2 => "Dataset Path: YOLO folder/archive (.tar/.zip) or RF-DETR COCO dataset directory (Enter to edit)".to_string(),
        3 => "Epochs: Total training passes over the dataset (Webview default: 60; Left/Right or Enter)".to_string(),
        4 => "Batch Size: Images processed per step (Webview default: 4; Left/Right or Enter)".to_string(),
        5 => "Resolution (imgsz): Input image size — 640, 960 (default), or 1280 (Left/Right or Enter)".to_string(),
        6 => "Workers: PyTorch DataLoader workers — auto (default), 0, 1, 2, 4, 6, 7, 8 (0 recommended on Windows Arc)".to_string(),
        7 => "Device: Compute target — auto (default), xpu, xpu:0, xpu:0,1 (dual-XPU), 0, cuda, cpu".to_string(),
        8 => "Freeze Backbone: Unfrozen (freeze=0, default) or Freeze first 10 backbone layers (Left/Right/Enter)".to_string(),
        9 => "Mixed Precision (AMP): Enabled FP16 (default) or Disabled FP32 (Left/Right/Enter)".to_string(),
        10 => "Run Name: Output directory under runs/<task>/<run_name> (Webview default: BB-Parts-v3)".to_string(),
        11 => "Augmentation Preset: Bench (default: deg=180, flipud=0.5, scale=0.05, close_mosaic=10, mosaic=0), Off, Light, Medium".to_string(),
        12 => "Rotation (degrees): Random rotation range 0–180° (Bench default: 180)".to_string(),
        13 => "Vertical Flip (flipud): Up-down flip probability 0.0–1.0 (Bench default: 0.5)".to_string(),
        14 => "Scale Gain (scale): Image scale jitter 0.0–1.0 (Bench default: 0.05)".to_string(),
        15 => "Translate: Random translation fraction 0.0–1.0 (Webview default: 0.1)".to_string(),
        16 => "Mosaic: 4-image mosaic probability 0.0–1.0 (Bench default: 0.0)".to_string(),
        17 => "Close Mosaic: Disable mosaic for final N epochs (Bench default: 10)".to_string(),
        18 => "MixUp: Image blend augmentation probability 0.0–1.0 (Webview default: 0.0)".to_string(),
        19 => "Copy-Paste: Instance segmentation copy-paste augmentation Off (0.0, default) / On (0.3)".to_string(),
        _ => "Press Enter or F2 to start training with the selected scale and hyperparameters".to_string(),
    }
}

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

    if inner.height > 0 && inner.width > 4 {
        let padded = Rect {
            x: inner.x + 1,
            y: inner.y,
            width: inner.width.saturating_sub(2),
            height: inner.height,
        };

        let s = &state.setup;
        let spacious = padded.height >= 20;
        let top_h = if spacious { 5 } else { 4 };
        let mid_h = if spacious { 11 } else { 10 };
        let bot_h = padded.height.saturating_sub(top_h + mid_h);

        let v_sections = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(top_h),
                Constraint::Length(mid_h),
                Constraint::Length(bot_h),
            ])
            .split(padded);

        // 1. Top full-width section: Header + Architecture (0) + Model Scale (1) + Dataset Path (2)
        let arch_val = format!(
            "◀ {} ({}) ▶",
            s.arch.display_label(),
            s.arch.as_cli_str()
        );
        let scale_val = if s.focused_field == 1 && s.editing_field {
            s.model.clone()
        } else if let Some(opt) = s.arch.find_option(&s.model) {
            format!("◀ {} · {} ({}) ▶", opt.scale, opt.id, opt.label)
        } else {
            format!("◀ custom · {} ▶", s.model)
        };
        let scale_list_hint = match s.arch {
            crate::app::ArchChoice::RfDetr => {
                "nano | small | medium (default) | large | base | seg-*"
            }
            crate::app::ArchChoice::Yolo | crate::app::ArchChoice::Yolo26 => {
                "n | s | m (default) | l | x | seg-*"
            }
            crate::app::ArchChoice::RtDetr => "r18 (default) | large | xlarge",
        };
        let ds_display = if s.dataset.is_empty() && !(s.focused_field == 2 && s.editing_field) {
            "<enter dataset path>".to_string()
        } else {
            s.dataset.clone()
        };

        let mut top_lines: Vec<Line<'static>> = Vec::with_capacity(top_h as usize);
        top_lines.push(Line::from(vec![
            Span::styled(
                "   Configure training run (Webview defaults). Press ",
                Style::default().fg(LABEL_MUTED),
            ),
            Span::styled(
                "←/→",
                Style::default()
                    .fg(TITLE_ACCENT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" to cycle, ", Style::default().fg(LABEL_MUTED)),
            Span::styled(
                "Enter",
                Style::default()
                    .fg(TITLE_ACCENT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" to edit, or ", Style::default().fg(LABEL_MUTED)),
            Span::styled(
                "F2",
                Style::default()
                    .fg(Color::LightGreen)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" to start.", Style::default().fg(LABEL_MUTED)),
        ]));
        top_lines.push(format_field_line(
            0,
            s.focused_field,
            s.editing_field,
            "Architecture",
            &arch_val,
            false,
            Some("yolo11 | yolo26 | rf-detr | rt-detr"),
            15,
        ));
        top_lines.push(format_field_line(
            1,
            s.focused_field,
            s.editing_field,
            "Model Scale",
            &scale_val,
            true,
            Some(scale_list_hint),
            15,
        ));
        top_lines.push(format_field_line(
            2,
            s.focused_field,
            s.editing_field,
            "Dataset Path",
            &ds_display,
            true,
            Some("Enter to edit dataset folder or archive path"),
            15,
        ));
        frame.render_widget(Paragraph::new(top_lines), v_sections[0]);

        // 2. Middle 2-column section: Hyperparameters (left, fields 3..=10) & Augmentation (right, fields 11..=19)
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(v_sections[1]);

        let imgsz_disp = if s.focused_field == 5 && s.editing_field {
            s.imgsz_input.clone()
        } else {
            let resolved = s.resolve_imgsz().unwrap_or(s.imgsz);
            format!("◀ {resolved} ▶")
        };
        let workers_disp = if s.focused_field == 6 && s.editing_field {
            s.workers_input.clone()
        } else if s.workers_input.trim().is_empty() {
            "◀ auto ▶".to_string()
        } else {
            format!("◀ {} ▶", s.workers_input.trim())
        };
        let device_disp = if s.focused_field == 7 && s.editing_field {
            s.device.clone()
        } else {
            format!("◀ {} ▶", s.device.trim())
        };
        let freeze_disp = if s.freeze_backbone {
            "◀ Freeze 10 layers ▶"
        } else {
            "◀ Unfrozen (freeze=0) ▶"
        };
        let amp_disp = if s.amp {
            "◀ Enabled (FP16) ▶"
        } else {
            "◀ Disabled (FP32) ▶"
        };
        let resolved_model = s.arch.resolve_model_or_scale(&s.model);
        let task_str = if resolved_model.to_ascii_lowercase().contains("-seg") {
            "segment"
        } else {
            "detect"
        };

        let mut left_lines: Vec<Line<'static>> = Vec::with_capacity(10);
        left_lines.push(Line::from(Span::styled(
            "   ── Hyperparameters & Device ──────────────────────",
            Style::default()
                .fg(TITLE_ACCENT)
                .add_modifier(Modifier::BOLD),
        )));
        left_lines.push(format_field_line(
            3,
            s.focused_field,
            s.editing_field,
            "Epochs",
            &s.epochs,
            true,
            Some("default: 60"),
            15,
        ));
        left_lines.push(format_field_line(
            4,
            s.focused_field,
            s.editing_field,
            "Batch Size",
            &s.batch,
            true,
            Some("default: 4"),
            15,
        ));
        left_lines.push(format_field_line(
            5,
            s.focused_field,
            s.editing_field,
            "Resolution",
            &imgsz_disp,
            true,
            Some("640 | 960 | 1280"),
            15,
        ));
        left_lines.push(format_field_line(
            6,
            s.focused_field,
            s.editing_field,
            "Workers",
            &workers_disp,
            true,
            Some("auto|0|1|2|4|6|7|8"),
            15,
        ));
        left_lines.push(format_field_line(
            7,
            s.focused_field,
            s.editing_field,
            "Device",
            &device_disp,
            true,
            Some("auto|xpu|xpu:0,1|cuda"),
            15,
        ));
        left_lines.push(format_field_line(
            8,
            s.focused_field,
            s.editing_field,
            "Freeze Backbone",
            freeze_disp,
            false,
            None,
            15,
        ));
        left_lines.push(format_field_line(
            9,
            s.focused_field,
            s.editing_field,
            "Mixed Prec (AMP)",
            amp_disp,
            false,
            None,
            15,
        ));
        left_lines.push(format_field_line(
            10,
            s.focused_field,
            s.editing_field,
            "Run Name",
            &s.run_name,
            true,
            Some(&format!("task: {task_str}")),
            15,
        ));
        frame.render_widget(Paragraph::new(left_lines), cols[0]);

        let active_preset = s.active_aug_preset();
        let preset_disp = format!("◀ {} ▶", active_preset.to_ascii_uppercase());
        let cp_disp = if s.copy_paste {
            "◀ On (0.3) ▶"
        } else {
            "◀ Off (0.0) ▶"
        };

        let mut right_lines: Vec<Line<'static>> = Vec::with_capacity(10);
        right_lines.push(Line::from(Span::styled(
            format!(
                "   ── Augmentation (Preset: {}) ──────────────────",
                active_preset.to_ascii_uppercase()
            ),
            Style::default()
                .fg(TITLE_ACCENT)
                .add_modifier(Modifier::BOLD),
        )));
        right_lines.push(format_field_line(
            11,
            s.focused_field,
            s.editing_field,
            "Aug Preset",
            &preset_disp,
            false,
            Some("bench|off|light|medium"),
            15,
        ));
        right_lines.push(format_field_line(
            12,
            s.focused_field,
            s.editing_field,
            "Degrees",
            &s.degrees,
            true,
            Some("0–180° (bench: 180)"),
            15,
        ));
        right_lines.push(format_field_line(
            13,
            s.focused_field,
            s.editing_field,
            "Flip UD",
            &s.flipud,
            true,
            Some("0.0–1.0 (bench: 0.5)"),
            15,
        ));
        right_lines.push(format_field_line(
            14,
            s.focused_field,
            s.editing_field,
            "Scale",
            &s.scale,
            true,
            Some("0.0–1.0 (bench: 0.05)"),
            15,
        ));
        right_lines.push(format_field_line(
            15,
            s.focused_field,
            s.editing_field,
            "Translate",
            &s.translate,
            true,
            Some("0.0–1.0 (default: 0.1)"),
            15,
        ));
        right_lines.push(format_field_line(
            16,
            s.focused_field,
            s.editing_field,
            "Mosaic",
            &s.mosaic,
            true,
            Some("0.0–1.0 (bench: 0.0)"),
            15,
        ));
        right_lines.push(format_field_line(
            17,
            s.focused_field,
            s.editing_field,
            "Close Mosaic",
            &s.close_mosaic,
            true,
            Some("epochs (bench: 10)"),
            15,
        ));
        right_lines.push(format_field_line(
            18,
            s.focused_field,
            s.editing_field,
            "MixUp",
            &s.mixup,
            true,
            Some("0.0–1.0 (default: 0.0)"),
            15,
        ));
        right_lines.push(format_field_line(
            19,
            s.focused_field,
            s.editing_field,
            "Copy-Paste",
            cp_disp,
            false,
            Some("seg copy-paste"),
            15,
        ));
        frame.render_widget(Paragraph::new(right_lines), cols[1]);

        // 3. Bottom full-width section: Start Training button (20) + active field help or error banner
        let start_focused = s.focused_field == 20;
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

        let mut bot_lines: Vec<Line<'static>> = Vec::new();
        bot_lines.push(Line::from(vec![
            Span::styled(
                start_prefix,
                Style::default()
                    .fg(Color::LightGreen)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("   [ Start Training (Enter / F2) ]   ", start_btn_style),
            Span::styled(
                format!(
                    "   Launch {} ({}) · {} ep · batch {} · {}px · dev {}",
                    s.arch.display_label(),
                    resolved_model,
                    s.epochs,
                    s.batch,
                    s.resolve_imgsz().unwrap_or(s.imgsz),
                    s.device
                ),
                Style::default().fg(LABEL_MUTED),
            ),
        ]));

        if let Some(ref err) = s.error_banner {
            bot_lines.push(Line::from(Span::styled(
                format!("   ⚠ {err}"),
                Style::default()
                    .fg(Color::LightRed)
                    .add_modifier(Modifier::BOLD),
            )));
        } else if bot_h >= 2 {
            bot_lines.push(Line::from(vec![
                Span::styled("   ℹ ", Style::default().fg(TITLE_ACCENT)),
                Span::styled(focused_field_help(state), Style::default().fg(LABEL_MUTED)),
            ]));
        }

        frame.render_widget(Paragraph::new(bot_lines), v_sections[2]);
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
            " [←/→] ",
            Style::default()
                .fg(GAUGE_TEXT_COVERED)
                .bg(BAR_STEP_BLUE)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Cycle Scale / Value  ", Style::default().fg(Color::White)),
        Span::styled(
            " [Enter] ",
            Style::default()
                .fg(GAUGE_TEXT_COVERED)
                .bg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Edit / Toggle  ", Style::default().fg(Color::White)),
        Span::styled(
            " [F2] ",
            Style::default()
                .fg(GAUGE_TEXT_COVERED)
                .bg(Color::LightGreen)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Start Training  ", Style::default().fg(Color::White)),
        Span::styled(
            " [F3] ",
            Style::default()
                .fg(GAUGE_TEXT_COVERED)
                .bg(BAR_GPU_TEAL)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Config  ", Style::default().fg(Color::White)),
        Span::styled(
            " [Esc/q] ",
            Style::default()
                .fg(Color::White)
                .bg(Color::Red)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Quit ", Style::default().fg(Color::White)),
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
