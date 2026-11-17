//! Full-screen interactive Launch Menu (`ls-trainer-tui` started with no CLI flags).

use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Clear, Paragraph},
};

use crate::app::{
    APP_VERSION_LABEL, AppState, BAR_GPU_TEAL, BAR_STEP_BLUE, BAR_TRACK_BG, BORDER_SOFT,
    FIELD_AMP, FIELD_ARCH, FIELD_AUG_PRESET, FIELD_BATCH, FIELD_CLOSE_MOSAIC, FIELD_COPY_PASTE,
    FIELD_DATASET, FIELD_DEGREES, FIELD_DEVICE, FIELD_EPOCHS, FIELD_FLIPUD, FIELD_FREEZE,
    FIELD_IMGSZ, FIELD_MIXUP, FIELD_MOSAIC, FIELD_RESUME, FIELD_RUN_NAME, FIELD_SCALE,
    FIELD_SCALE_GAIN, FIELD_START, FIELD_TRANSLATE, FIELD_WORKERS, GAUGE_TEXT_COVERED, LABEL_MUTED,
    OVERLAY_BG, TABLE_BORDER_SOFT, TITLE_ACCENT, dataset_looks_like_archive, resume_file_name,
    resume_status_text, BrowseEntryKind, BrowseKind, BROWSE_DIR_FG,
};
use crate::ui::centered_rect;
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
        FIELD_ARCH => "Architecture family: Left/Right or Enter to switch YOLO11 | YOLO26 | RF-DETR | RT-DETR".to_string(),
        FIELD_SCALE => {
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
        FIELD_DATASET => "A folder is used as-is. A .tar/.tar.gz/.tgz with data.yaml, labels, and images is unpacked into the dataset store. F6 opens folders; .. goes up.".to_string(),
        FIELD_RESUME => "Blank = fresh start. Type a .pt/.pth path or press F6 to browse. Gold folders and .. move between directories. Fine-tune adds -ft to the suggested run name.".to_string(),
        FIELD_EPOCHS => "Epochs: Total training passes over the dataset (Webview default: 60; Left/Right or Enter)".to_string(),
        FIELD_BATCH => "Batch Size: Images processed per step (Webview default: 4; Left/Right or Enter)".to_string(),
        FIELD_IMGSZ => "Resolution (imgsz): Input image size — 640, 960 (default), or 1280 (Left/Right or Enter)".to_string(),
        FIELD_WORKERS => "Workers: PyTorch DataLoader workers — auto (default), 0, 1, 2, 4, 6, 7, 8 (0 recommended on Windows Arc)".to_string(),
        FIELD_DEVICE => "Device: Compute target — auto (default), xpu, xpu:0, xpu:0,1 (dual-XPU), 0, cuda, cpu".to_string(),
        FIELD_FREEZE => "Freeze Backbone: Unfrozen (freeze=0, default) or Freeze first 10 backbone layers (Left/Right/Enter)".to_string(),
        FIELD_AMP => "Mixed Precision (AMP): Enabled FP16 (default) or Disabled FP32 (Left/Right/Enter)".to_string(),
        FIELD_RUN_NAME => "Run Name: suggested from the family, scale, epochs, and image size. Resume adds -ft. Enter to edit; your text is kept.".to_string(),
        FIELD_AUG_PRESET => "Augmentation Preset: Bench (default: deg=180, flipud=0.5, scale=0.05, close_mosaic=10, mosaic=0), Off, Light, Medium".to_string(),
        FIELD_DEGREES => "Rotation (degrees): Random rotation range 0–180° (Bench default: 180)".to_string(),
        FIELD_FLIPUD => "Vertical Flip (flipud): Up-down flip probability 0.0–1.0 (Bench default: 0.5)".to_string(),
        FIELD_SCALE_GAIN => "Scale Gain (scale): Image scale jitter 0.0–1.0 (Bench default: 0.05)".to_string(),
        FIELD_TRANSLATE => "Translate: Random translation fraction 0.0–1.0 (Webview default: 0.1)".to_string(),
        FIELD_MOSAIC => "Mosaic: 4-image mosaic probability 0.0–1.0 (Bench default: 0.0)".to_string(),
        FIELD_CLOSE_MOSAIC => "Close Mosaic: Disable mosaic for final N epochs (Bench default: 10)".to_string(),
        FIELD_MIXUP => "MixUp: Image blend augmentation probability 0.0–1.0 (Webview default: 0.0)".to_string(),
        FIELD_COPY_PASTE => "Copy-Paste: Instance segmentation copy-paste augmentation Off (0.0, default) / On (0.3)".to_string(),
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
        let spacious = padded.height >= 21;
        let top_h = if spacious { 6 } else { 5 };
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
        let scale_val = if s.focused_field == FIELD_SCALE && s.editing_field {
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
        let ds_display = if s.dataset.is_empty() && !(s.focused_field == FIELD_DATASET && s.editing_field)
        {
            "<enter dataset path>".to_string()
        } else {
            s.dataset.clone()
        };
        let ds_hint = if dataset_looks_like_archive(&s.dataset) {
            "archive · unpacks into the dataset store on start"
        } else {
            "folder or .tar · Enter to edit · F6 browse"
        };
        let resume_display =
            if s.resume_from.is_empty() && !(s.focused_field == FIELD_RESUME && s.editing_field) {
                "<fresh start>".to_string()
            } else {
                s.resume_from.clone()
            };
        let resume_hint = "blank=fresh · YOLO model= · DETR pretrain_weights";

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
            FIELD_ARCH,
            s.focused_field,
            s.editing_field,
            "Architecture",
            &arch_val,
            false,
            Some("yolo11 | yolo26 | rf-detr | rt-detr"),
            15,
        ));
        top_lines.push(format_field_line(
            FIELD_SCALE,
            s.focused_field,
            s.editing_field,
            "Model Scale",
            &scale_val,
            true,
            Some(scale_list_hint),
            15,
        ));
        top_lines.push(format_field_line(
            FIELD_DATASET,
            s.focused_field,
            s.editing_field,
            "Dataset Path",
            &ds_display,
            true,
            Some(ds_hint),
            15,
        ));
        top_lines.push(format_field_line(
            FIELD_RESUME,
            s.focused_field,
            s.editing_field,
            "Resume from",
            &resume_display,
            true,
            Some(resume_hint),
            15,
        ));
        frame.render_widget(Paragraph::new(top_lines), v_sections[0]);

        // 2. Middle 2-column section: Hyperparameters (left, fields 3..=10) & Augmentation (right, fields 11..=19)
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(v_sections[1]);

        let imgsz_disp = if s.focused_field == FIELD_IMGSZ && s.editing_field {
            s.imgsz_input.clone()
        } else {
            let resolved = s.resolve_imgsz().unwrap_or(s.imgsz);
            format!("◀ {resolved} ▶")
        };
        let workers_disp = if s.focused_field == FIELD_WORKERS && s.editing_field {
            s.workers_input.clone()
        } else if s.workers_input.trim().is_empty() {
            "◀ auto ▶".to_string()
        } else {
            format!("◀ {} ▶", s.workers_input.trim())
        };
        let device_disp = if s.focused_field == FIELD_DEVICE && s.editing_field {
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
            FIELD_EPOCHS,
            s.focused_field,
            s.editing_field,
            "Epochs",
            &s.epochs,
            true,
            Some("default: 60"),
            15,
        ));
        left_lines.push(format_field_line(
            FIELD_BATCH,
            s.focused_field,
            s.editing_field,
            "Batch Size",
            &s.batch,
            true,
            Some("default: 4"),
            15,
        ));
        left_lines.push(format_field_line(
            FIELD_IMGSZ,
            s.focused_field,
            s.editing_field,
            "Resolution",
            &imgsz_disp,
            true,
            Some("640 | 960 | 1280"),
            15,
        ));
        left_lines.push(format_field_line(
            FIELD_WORKERS,
            s.focused_field,
            s.editing_field,
            "Workers",
            &workers_disp,
            true,
            Some("auto|0|1|2|4|6|7|8"),
            15,
        ));
        left_lines.push(format_field_line(
            FIELD_DEVICE,
            s.focused_field,
            s.editing_field,
            "Device",
            &device_disp,
            true,
            Some("auto|xpu|xpu:0,1|cuda"),
            15,
        ));
        left_lines.push(format_field_line(
            FIELD_FREEZE,
            s.focused_field,
            s.editing_field,
            "Freeze Backbone",
            freeze_disp,
            false,
            None,
            15,
        ));
        left_lines.push(format_field_line(
            FIELD_AMP,
            s.focused_field,
            s.editing_field,
            "Mixed Prec (AMP)",
            amp_disp,
            false,
            None,
            15,
        ));
        left_lines.push(format_field_line(
            FIELD_RUN_NAME,
            s.focused_field,
            s.editing_field,
            "Run Name",
            &s.run_name,
            true,
            Some(&format!("suggested · {task_str}")),
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
            FIELD_AUG_PRESET,
            s.focused_field,
            s.editing_field,
            "Aug Preset",
            &preset_disp,
            false,
            Some("bench|off|light|medium"),
            15,
        ));
        right_lines.push(format_field_line(
            FIELD_DEGREES,
            s.focused_field,
            s.editing_field,
            "Degrees",
            &s.degrees,
            true,
            Some("0–180° (bench: 180)"),
            15,
        ));
        right_lines.push(format_field_line(
            FIELD_FLIPUD,
            s.focused_field,
            s.editing_field,
            "Flip UD",
            &s.flipud,
            true,
            Some("0.0–1.0 (bench: 0.5)"),
            15,
        ));
        right_lines.push(format_field_line(
            FIELD_SCALE_GAIN,
            s.focused_field,
            s.editing_field,
            "Scale",
            &s.scale,
            true,
            Some("0.0–1.0 (bench: 0.05)"),
            15,
        ));
        right_lines.push(format_field_line(
            FIELD_TRANSLATE,
            s.focused_field,
            s.editing_field,
            "Translate",
            &s.translate,
            true,
            Some("0.0–1.0 (default: 0.1)"),
            15,
        ));
        right_lines.push(format_field_line(
            FIELD_MOSAIC,
            s.focused_field,
            s.editing_field,
            "Mosaic",
            &s.mosaic,
            true,
            Some("0.0–1.0 (bench: 0.0)"),
            15,
        ));
        right_lines.push(format_field_line(
            FIELD_CLOSE_MOSAIC,
            s.focused_field,
            s.editing_field,
            "Close Mosaic",
            &s.close_mosaic,
            true,
            Some("epochs (bench: 10)"),
            15,
        ));
        right_lines.push(format_field_line(
            FIELD_MIXUP,
            s.focused_field,
            s.editing_field,
            "MixUp",
            &s.mixup,
            true,
            Some("0.0–1.0 (default: 0.0)"),
            15,
        ));
        right_lines.push(format_field_line(
            FIELD_COPY_PASTE,
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
        let start_focused = s.focused_field == FIELD_START;
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
                    "   Launch {} ({}) · {} ep · batch {} · {}px · dev {}{}",
                    s.arch.display_label(),
                    resolved_model,
                    s.epochs,
                    s.batch,
                    s.resolve_imgsz().unwrap_or(s.imgsz),
                    s.device,
                    resume_file_name(&s.resume_from)
                        .map(|name| format!(" · ft {name}"))
                        .unwrap_or_default()
                ),
                Style::default().fg(LABEL_MUTED),
            ),
        ]));

        if let Some(status) = resume_status_text(s.arch, &s.resume_from) {
            bot_lines.push(Line::from(Span::styled(
                format!("   {status}"),
                Style::default()
                    .fg(Color::LightYellow)
                    .add_modifier(Modifier::BOLD),
            )));
        }

        if let Some(ref err) = s.error_banner {
            bot_lines.push(Line::from(Span::styled(
                format!("   ⚠ {err}"),
                Style::default()
                    .fg(Color::LightRed)
                    .add_modifier(Modifier::BOLD),
            )));
        } else if (bot_h as usize) > bot_lines.len() {
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

fn clip_to_width(text: &str, width: usize) -> String {
    let width = width.max(1);
    let count = text.chars().count();
    if count <= width {
        return text.to_string();
    }
    let keep = width.saturating_sub(1);
    let mut clipped: String = text.chars().take(keep).collect();
    clipped.push('…');
    clipped
}

pub fn render_browse_overlay(frame: &mut Frame, area: Rect, state: &AppState) {
    let modal = centered_rect(78, 70, area);
    frame.render_widget(Clear, modal);

    let title = match state.browse_kind {
        BrowseKind::Weights => " Browse weights (.pt / .pth) — Enter opens or selects ",
        BrowseKind::Dataset => " Browse dataset — Enter opens a folder or selects a .tar ",
    };
    let block = Block::default()
        .title(Span::styled(
            title,
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

    let width = inner.width as usize;
    let mut lines: Vec<Line<'static>> = Vec::new();
    let show_header = inner.height >= 4;
    if show_header {
        let location = app_lib::to_display_path(&state.browse_dir);
        lines.push(Line::from(Span::styled(
            clip_to_width(&format!("  {location}"), width),
            Style::default().fg(LABEL_MUTED),
        )));
        lines.push(Line::from(Span::styled(
            clip_to_width("  gold folder · Enter opens · .. goes up · Esc closes", width),
            Style::default().fg(LABEL_MUTED),
        )));
    }

    let list_height = if show_header {
        inner.height.saturating_sub(2) as usize
    } else {
        inner.height as usize
    };
    if state.browse_entries.is_empty() {
        lines.push(Line::from(Span::styled(
            "  This folder cannot be read.",
            Style::default().fg(LABEL_MUTED),
        )));
    } else {
        let start = state
            .browse_cursor
            .saturating_sub(list_height / 2)
            .min(state.browse_entries.len().saturating_sub(list_height.max(1)));
        for (offset, entry) in state
            .browse_entries
            .iter()
            .enumerate()
            .skip(start)
            .take(list_height)
        {
            let selected = offset == state.browse_cursor;
            let navigable = entry.kind != BrowseEntryKind::File;
            let prefix = if selected { " ▶ " } else { "   " };
            let style = if selected && navigable {
                Style::default()
                    .fg(GAUGE_TEXT_COVERED)
                    .bg(BROWSE_DIR_FG)
                    .add_modifier(Modifier::BOLD)
            } else if selected {
                Style::default()
                    .fg(GAUGE_TEXT_COVERED)
                    .bg(BAR_GPU_TEAL)
                    .add_modifier(Modifier::BOLD)
            } else if navigable {
                Style::default()
                    .fg(BROWSE_DIR_FG)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            lines.push(Line::from(Span::styled(
                clip_to_width(&format!("{prefix}{}", entry.label), width),
                style,
            )));
        }
        let only_navigation = state
            .browse_entries
            .iter()
            .all(|entry| entry.kind != BrowseEntryKind::File);
        if only_navigation && lines.len() < inner.height as usize {
            lines.push(Line::from(Span::styled(
                "  No matching files in this folder.",
                Style::default().fg(LABEL_MUTED),
            )));
        }
    }

    frame.render_widget(Paragraph::new(lines), inner);
}
