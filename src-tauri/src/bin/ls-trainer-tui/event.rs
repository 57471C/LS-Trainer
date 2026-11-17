use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use app_lib::{
    active_training_control, force_kill_active_child, send_training_control, stop_training,
    ProcessControlCommand, TrainingRequest,
};
use arboard::Clipboard;
use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};

use crate::app::{
    launch_training_thread, scan_run_history, AppState, OverlayMode, UiMode, AUG_PRESET_NAMES,
    BATCH_PRESETS, CLOSE_MOSAIC_PRESETS, DEGREES_PRESETS, DEVICE_PRESETS, EPOCH_PRESETS,
    FIELD_AMP, FIELD_AUG_PRESET, FIELD_BATCH, FIELD_CLOSE_MOSAIC, FIELD_COPY_PASTE,
    FIELD_DATASET, FIELD_DEGREES, FIELD_DEVICE, FIELD_EPOCHS, FIELD_FLIPUD, FIELD_FREEZE,
    FIELD_IMGSZ, FIELD_MIXUP, FIELD_MOSAIC, FIELD_RESUME, FIELD_RUN_NAME, FIELD_SCALE,
    FIELD_SCALE_GAIN, FIELD_START, FIELD_TRANSLATE, FIELD_WORKERS, FLIPUD_PRESETS, IMGSZ_PRESETS,
    LAUNCHER_FIELD_COUNT, MIXUP_PRESETS, MOSAIC_PRESETS, SCALE_PRESETS, TRANSLATE_PRESETS,
    WORKER_PRESETS,
};

static CLIPBOARD_UNAVAILABLE: AtomicBool = AtomicBool::new(false);
static CLIPBOARD_FALLBACK_ANNOUNCED: AtomicBool = AtomicBool::new(false);

/// Headless machines have no X11 clipboard. F12 writes the same text here instead.
pub fn clipboard_fallback_path() -> PathBuf {
    if cfg!(unix) {
        PathBuf::from("/tmp/ls-trainer-log.txt")
    } else {
        std::env::temp_dir().join("ls-trainer-log.txt")
    }
}

enum CopiedTo {
    Clipboard,
    File(PathBuf),
}

pub enum EventOutcome {
    Continue,
    Quit { show_exit_summary: bool },
}

pub fn handle_event(
    state: &Arc<Mutex<AppState>>,
    ev: Event,
    clipboard: &mut Option<Clipboard>,
) -> EventOutcome {
    match ev {
        Event::Resize(_w, _h) => EventOutcome::Continue,
        Event::Key(key) => {
            if key.kind != KeyEventKind::Press {
                return EventOutcome::Continue;
            }

            if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
                graceful_stop_and_wait(state);
                return EventOutcome::Quit {
                    show_exit_summary: true,
                };
            }

            let mode = state.lock().map(|g| g.mode).unwrap_or(UiMode::Training);

            if mode == UiMode::Setup {
                let is_quit_q = matches!(key.code, KeyCode::Char('q') | KeyCode::Char('Q'))
                    && state
                        .lock()
                        .map(|g| !g.setup.editing_field && g.overlay != OverlayMode::Browse)
                        .unwrap_or(false);
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    && matches!(key.code, KeyCode::Char('o') | KeyCode::Char('O'))
                {
                    if let Ok(mut guard) = state.lock() {
                        guard.open_launcher_browse();
                    }
                    return EventOutcome::Continue;
                }
                if is_quit_q {
                    return EventOutcome::Quit {
                        show_exit_summary: true,
                    };
                }
                if handle_setup_key(state, key.code) {
                    return EventOutcome::Quit {
                        show_exit_summary: false,
                    };
                }
                return EventOutcome::Continue;
            }

            if handle_training_key(state, key.code, clipboard) {
                return EventOutcome::Quit {
                    show_exit_summary: true,
                };
            }
            EventOutcome::Continue
        }
        _ => EventOutcome::Continue,
    }
}

pub fn cycle_preset(current: &str, presets: &[&str], forward: bool) -> String {
    let cur_trimmed = current.trim();
    let idx = presets
        .iter()
        .position(|p| p.eq_ignore_ascii_case(cur_trimmed))
        .unwrap_or(0);
    let next_idx = if forward {
        (idx + 1) % presets.len()
    } else if idx == 0 {
        presets.len() - 1
    } else {
        idx - 1
    };
    presets[next_idx].to_string()
}

fn browse_overlay_key(state: &mut AppState, code: KeyCode) {
    match code {
        KeyCode::Esc | KeyCode::F(6) => {
            state.overlay = OverlayMode::None;
        }
        KeyCode::Up => {
            state.browse_cursor = state.browse_cursor.saturating_sub(1);
        }
        KeyCode::Down => {
            if !state.browse_entries.is_empty() {
                let max_idx = state.browse_entries.len() - 1;
                state.browse_cursor = (state.browse_cursor + 1).min(max_idx);
            }
        }
        KeyCode::Enter => {
            state.apply_browse_selection();
        }
        _ => {}
    }
}

pub fn handle_setup_key(state: &Arc<Mutex<AppState>>, code: KeyCode) -> bool {
    let mut launch_req: Option<TrainingRequest> = None;
    if let Ok(mut guard) = state.lock() {
        if guard.overlay == OverlayMode::Browse {
            browse_overlay_key(&mut guard, code);
            return false;
        }
        if guard.overlay != OverlayMode::None {
            match code {
                KeyCode::Esc
                | KeyCode::F(1)
                | KeyCode::F(3)
                | KeyCode::F(4)
                | KeyCode::Char('?') => {
                    guard.overlay = OverlayMode::None;
                }
                _ => {}
            }
            return false;
        }

        let focused = guard.setup.focused_field;
        let finish_edit = |s: &mut crate::app::CliConfig| {
            if s.editing_field {
                if s.focused_field == FIELD_SCALE {
                    s.model = s.arch.resolve_model_or_scale(&s.model);
                } else if s.focused_field == FIELD_IMGSZ {
                    if let Ok(parsed) = s.imgsz_input.trim().parse::<u32>() {
                        s.imgsz = parsed.max(32);
                    } else if s.imgsz_input.trim().is_empty() {
                        s.imgsz_input = s.imgsz.to_string();
                    }
                } else if s.focused_field == FIELD_WORKERS && s.workers_input.trim().is_empty() {
                    s.workers_input = "auto".to_string();
                } else if s.focused_field == FIELD_RUN_NAME {
                    s.finish_run_name_edit();
                }
                s.editing_field = false;
            }
        };

        match code {
            KeyCode::Esc => {
                if guard.setup.editing_field {
                    finish_edit(&mut guard.setup);
                } else {
                    return true;
                }
            }
            KeyCode::F(1) => {
                guard.overlay = OverlayMode::Help;
            }
            KeyCode::F(3) => {
                guard.overlay = OverlayMode::Config;
            }
            KeyCode::F(6) => {
                guard.open_launcher_browse();
            }
            KeyCode::F(2) => {
                finish_edit(&mut guard.setup);
                guard.setup.refresh_suggested_run_name();
                match guard.setup.to_training_request() {
                    Ok(req) => {
                        guard.setup.error_banner = None;
                        launch_req = Some(req);
                    }
                    Err(err) => {
                        guard.setup.error_banner = Some(err);
                    }
                }
            }
            KeyCode::Tab | KeyCode::Down => {
                finish_edit(&mut guard.setup);
                guard.setup.focused_field = (focused + 1) % LAUNCHER_FIELD_COUNT;
            }
            KeyCode::BackTab | KeyCode::Up => {
                finish_edit(&mut guard.setup);
                guard.setup.focused_field = if focused == 0 {
                    LAUNCHER_FIELD_COUNT - 1
                } else {
                    focused - 1
                };
            }
            KeyCode::Left => match focused {
                0 => {
                    let prev_arch = guard.setup.arch.prev();
                    guard.setup.switch_arch(prev_arch);
                }
                1 if !guard.setup.editing_field => {
                    guard.setup.model = guard.setup.arch.cycle_model(&guard.setup.model, false);
                }
                FIELD_EPOCHS if !guard.setup.editing_field => {
                    guard.setup.epochs = cycle_preset(&guard.setup.epochs, &EPOCH_PRESETS, false);
                }
                FIELD_BATCH if !guard.setup.editing_field => {
                    guard.setup.batch = cycle_preset(&guard.setup.batch, &BATCH_PRESETS, false);
                }
                FIELD_IMGSZ if !guard.setup.editing_field => {
                    guard.setup.imgsz_input =
                        cycle_preset(&guard.setup.imgsz_input, &IMGSZ_PRESETS, false);
                    if let Ok(parsed) = guard.setup.imgsz_input.parse::<u32>() {
                        guard.setup.imgsz = parsed;
                    }
                }
                FIELD_WORKERS if !guard.setup.editing_field => {
                    guard.setup.workers_input =
                        cycle_preset(&guard.setup.workers_input, &WORKER_PRESETS, false);
                }
                FIELD_DEVICE if !guard.setup.editing_field => {
                    guard.setup.device = cycle_preset(&guard.setup.device, &DEVICE_PRESETS, false);
                }
                FIELD_FREEZE => {
                    guard.setup.freeze_backbone = !guard.setup.freeze_backbone;
                }
                FIELD_AMP => {
                    guard.setup.amp = !guard.setup.amp;
                }
                FIELD_AUG_PRESET => {
                    let cur = guard.setup.active_aug_preset();
                    let next_p = cycle_preset(cur, &AUG_PRESET_NAMES, false);
                    guard.setup.apply_aug_preset(&next_p);
                }
                FIELD_DEGREES if !guard.setup.editing_field => {
                    guard.setup.degrees =
                        cycle_preset(&guard.setup.degrees, &DEGREES_PRESETS, false);
                }
                FIELD_FLIPUD if !guard.setup.editing_field => {
                    guard.setup.flipud = cycle_preset(&guard.setup.flipud, &FLIPUD_PRESETS, false);
                }
                FIELD_SCALE_GAIN if !guard.setup.editing_field => {
                    guard.setup.scale = cycle_preset(&guard.setup.scale, &SCALE_PRESETS, false);
                }
                FIELD_TRANSLATE if !guard.setup.editing_field => {
                    guard.setup.translate =
                        cycle_preset(&guard.setup.translate, &TRANSLATE_PRESETS, false);
                }
                FIELD_MOSAIC if !guard.setup.editing_field => {
                    guard.setup.mosaic = cycle_preset(&guard.setup.mosaic, &MOSAIC_PRESETS, false);
                }
                FIELD_CLOSE_MOSAIC if !guard.setup.editing_field => {
                    guard.setup.close_mosaic =
                        cycle_preset(&guard.setup.close_mosaic, &CLOSE_MOSAIC_PRESETS, false);
                }
                FIELD_MIXUP if !guard.setup.editing_field => {
                    guard.setup.mixup = cycle_preset(&guard.setup.mixup, &MIXUP_PRESETS, false);
                }
                FIELD_COPY_PASTE => {
                    guard.setup.copy_paste = !guard.setup.copy_paste;
                }
                _ => {}
            },
            KeyCode::Right => match focused {
                0 => {
                    let next_arch = guard.setup.arch.next();
                    guard.setup.switch_arch(next_arch);
                }
                1 if !guard.setup.editing_field => {
                    guard.setup.model = guard.setup.arch.cycle_model(&guard.setup.model, true);
                }
                FIELD_EPOCHS if !guard.setup.editing_field => {
                    guard.setup.epochs = cycle_preset(&guard.setup.epochs, &EPOCH_PRESETS, true);
                }
                FIELD_BATCH if !guard.setup.editing_field => {
                    guard.setup.batch = cycle_preset(&guard.setup.batch, &BATCH_PRESETS, true);
                }
                FIELD_IMGSZ if !guard.setup.editing_field => {
                    guard.setup.imgsz_input =
                        cycle_preset(&guard.setup.imgsz_input, &IMGSZ_PRESETS, true);
                    if let Ok(parsed) = guard.setup.imgsz_input.parse::<u32>() {
                        guard.setup.imgsz = parsed;
                    }
                }
                FIELD_WORKERS if !guard.setup.editing_field => {
                    guard.setup.workers_input =
                        cycle_preset(&guard.setup.workers_input, &WORKER_PRESETS, true);
                }
                FIELD_DEVICE if !guard.setup.editing_field => {
                    guard.setup.device = cycle_preset(&guard.setup.device, &DEVICE_PRESETS, true);
                }
                FIELD_FREEZE => {
                    guard.setup.freeze_backbone = !guard.setup.freeze_backbone;
                }
                FIELD_AMP => {
                    guard.setup.amp = !guard.setup.amp;
                }
                FIELD_AUG_PRESET => {
                    let cur = guard.setup.active_aug_preset();
                    let next_p = cycle_preset(cur, &AUG_PRESET_NAMES, true);
                    guard.setup.apply_aug_preset(&next_p);
                }
                FIELD_DEGREES if !guard.setup.editing_field => {
                    guard.setup.degrees =
                        cycle_preset(&guard.setup.degrees, &DEGREES_PRESETS, true);
                }
                FIELD_FLIPUD if !guard.setup.editing_field => {
                    guard.setup.flipud = cycle_preset(&guard.setup.flipud, &FLIPUD_PRESETS, true);
                }
                FIELD_SCALE_GAIN if !guard.setup.editing_field => {
                    guard.setup.scale = cycle_preset(&guard.setup.scale, &SCALE_PRESETS, true);
                }
                FIELD_TRANSLATE if !guard.setup.editing_field => {
                    guard.setup.translate =
                        cycle_preset(&guard.setup.translate, &TRANSLATE_PRESETS, true);
                }
                FIELD_MOSAIC if !guard.setup.editing_field => {
                    guard.setup.mosaic = cycle_preset(&guard.setup.mosaic, &MOSAIC_PRESETS, true);
                }
                FIELD_CLOSE_MOSAIC if !guard.setup.editing_field => {
                    guard.setup.close_mosaic =
                        cycle_preset(&guard.setup.close_mosaic, &CLOSE_MOSAIC_PRESETS, true);
                }
                FIELD_MIXUP if !guard.setup.editing_field => {
                    guard.setup.mixup = cycle_preset(&guard.setup.mixup, &MIXUP_PRESETS, true);
                }
                FIELD_COPY_PASTE => {
                    guard.setup.copy_paste = !guard.setup.copy_paste;
                }
                _ => {}
            },
            KeyCode::Backspace => match focused {
                1 => {
                    guard.setup.editing_field = true;
                    guard.setup.model.pop();
                }
                FIELD_DATASET => {
                    guard.setup.editing_field = true;
                    guard.setup.dataset.pop();
                }
                FIELD_RESUME => {
                    guard.setup.editing_field = true;
                    guard.setup.resume_from.pop();
                }
                FIELD_EPOCHS => {
                    guard.setup.editing_field = true;
                    guard.setup.epochs.pop();
                }
                FIELD_BATCH => {
                    guard.setup.editing_field = true;
                    guard.setup.batch.pop();
                }
                FIELD_IMGSZ => {
                    guard.setup.editing_field = true;
                    guard.setup.imgsz_input.pop();
                    if let Ok(parsed) = guard.setup.imgsz_input.parse::<u32>() {
                        guard.setup.imgsz = parsed;
                    }
                }
                FIELD_WORKERS => {
                    guard.setup.editing_field = true;
                    if guard.setup.workers_input.eq_ignore_ascii_case("auto") {
                        guard.setup.workers_input.clear();
                    } else {
                        guard.setup.workers_input.pop();
                    }
                }
                FIELD_DEVICE => {
                    guard.setup.editing_field = true;
                    guard.setup.device.pop();
                }
                FIELD_RUN_NAME => {
                    guard.setup.editing_field = true;
                    guard.setup.run_name_custom = true;
                    guard.setup.run_name.pop();
                }
                FIELD_DEGREES => {
                    guard.setup.editing_field = true;
                    guard.setup.degrees.pop();
                }
                FIELD_FLIPUD => {
                    guard.setup.editing_field = true;
                    guard.setup.flipud.pop();
                }
                FIELD_SCALE_GAIN => {
                    guard.setup.editing_field = true;
                    guard.setup.scale.pop();
                }
                FIELD_TRANSLATE => {
                    guard.setup.editing_field = true;
                    guard.setup.translate.pop();
                }
                FIELD_MOSAIC => {
                    guard.setup.editing_field = true;
                    guard.setup.mosaic.pop();
                }
                FIELD_CLOSE_MOSAIC => {
                    guard.setup.editing_field = true;
                    guard.setup.close_mosaic.pop();
                }
                FIELD_MIXUP => {
                    guard.setup.editing_field = true;
                    guard.setup.mixup.pop();
                }
                _ => {}
            },
            KeyCode::Char(c) => match focused {
                0 if c == ' ' => {
                    let next_arch = guard.setup.arch.next();
                    guard.setup.switch_arch(next_arch);
                }
                1 => {
                    guard.setup.editing_field = true;
                    guard.setup.model.push(c);
                }
                FIELD_DATASET => {
                    guard.setup.editing_field = true;
                    guard.setup.dataset.push(c);
                }
                FIELD_RESUME => {
                    guard.setup.editing_field = true;
                    guard.setup.resume_from.push(c);
                }
                FIELD_EPOCHS if c.is_ascii_digit() => {
                    guard.setup.editing_field = true;
                    guard.setup.epochs.push(c);
                }
                FIELD_BATCH if c.is_ascii_digit() => {
                    guard.setup.editing_field = true;
                    guard.setup.batch.push(c);
                }
                FIELD_IMGSZ if c.is_ascii_digit() => {
                    guard.setup.editing_field = true;
                    guard.setup.imgsz_input.push(c);
                    if let Ok(parsed) = guard.setup.imgsz_input.parse::<u32>() {
                        guard.setup.imgsz = parsed;
                    }
                }
                FIELD_WORKERS => {
                    guard.setup.editing_field = true;
                    if c.is_ascii_digit() {
                        if guard.setup.workers_input.eq_ignore_ascii_case("auto") {
                            guard.setup.workers_input.clear();
                        }
                        guard.setup.workers_input.push(c);
                    } else if c.eq_ignore_ascii_case(&'a') {
                        guard.setup.workers_input = "auto".to_string();
                    }
                }
                FIELD_DEVICE => {
                    guard.setup.editing_field = true;
                    guard.setup.device.push(c);
                }
                FIELD_FREEZE if c == ' ' => {
                    guard.setup.freeze_backbone = !guard.setup.freeze_backbone;
                }
                FIELD_AMP if c == ' ' => {
                    guard.setup.amp = !guard.setup.amp;
                }
                FIELD_RUN_NAME => {
                    guard.setup.editing_field = true;
                    guard.setup.run_name_custom = true;
                    guard.setup.run_name.push(c);
                }
                FIELD_AUG_PRESET if c == ' ' => {
                    let cur = guard.setup.active_aug_preset();
                    let next_p = cycle_preset(cur, &AUG_PRESET_NAMES, true);
                    guard.setup.apply_aug_preset(&next_p);
                }
                FIELD_DEGREES if c.is_ascii_digit() || c == '.' => {
                    guard.setup.editing_field = true;
                    guard.setup.degrees.push(c);
                }
                FIELD_FLIPUD if c.is_ascii_digit() || c == '.' => {
                    guard.setup.editing_field = true;
                    guard.setup.flipud.push(c);
                }
                FIELD_SCALE_GAIN if c.is_ascii_digit() || c == '.' => {
                    guard.setup.editing_field = true;
                    guard.setup.scale.push(c);
                }
                FIELD_TRANSLATE if c.is_ascii_digit() || c == '.' => {
                    guard.setup.editing_field = true;
                    guard.setup.translate.push(c);
                }
                FIELD_MOSAIC if c.is_ascii_digit() || c == '.' => {
                    guard.setup.editing_field = true;
                    guard.setup.mosaic.push(c);
                }
                FIELD_CLOSE_MOSAIC if c.is_ascii_digit() => {
                    guard.setup.editing_field = true;
                    guard.setup.close_mosaic.push(c);
                }
                FIELD_MIXUP if c.is_ascii_digit() || c == '.' => {
                    guard.setup.editing_field = true;
                    guard.setup.mixup.push(c);
                }
                FIELD_COPY_PASTE if c == ' ' => {
                    guard.setup.copy_paste = !guard.setup.copy_paste;
                }
                _ => {}
            },
            KeyCode::Enter => {
                if focused == FIELD_START {
                    finish_edit(&mut guard.setup);
                    guard.setup.refresh_suggested_run_name();
                    match guard.setup.to_training_request() {
                        Ok(req) => {
                            guard.setup.error_banner = None;
                            launch_req = Some(req);
                        }
                        Err(err) => {
                            guard.setup.error_banner = Some(err);
                        }
                    }
                } else if focused == 0 {
                    let next_arch = guard.setup.arch.next();
                    guard.setup.switch_arch(next_arch);
                } else if focused == FIELD_FREEZE {
                    guard.setup.freeze_backbone = !guard.setup.freeze_backbone;
                } else if focused == FIELD_AMP {
                    guard.setup.amp = !guard.setup.amp;
                } else if focused == FIELD_AUG_PRESET {
                    let cur = guard.setup.active_aug_preset();
                    let next_p = cycle_preset(cur, &AUG_PRESET_NAMES, true);
                    guard.setup.apply_aug_preset(&next_p);
                } else if focused == FIELD_COPY_PASTE {
                    guard.setup.copy_paste = !guard.setup.copy_paste;
                } else if guard.setup.editing_field {
                    finish_edit(&mut guard.setup);
                } else {
                    guard.setup.editing_field = true;
                }
            }
            _ => {}
        }
        let editing_run_name = guard.setup.focused_field == FIELD_RUN_NAME && guard.setup.editing_field;
        if !editing_run_name {
            guard.setup.refresh_suggested_run_name();
        }
    }

    if let Some(req) = launch_req {
        launch_training_thread(Arc::clone(state), req);
    }
    false
}

pub fn graceful_stop_and_wait(state: &Arc<Mutex<AppState>>) {
    if let Ok(mut guard) = state.lock() {
        guard.stop_initiated = true;
        guard.status_line = "Stopping training gracefully...".to_string();
    }

    if let Some(handle) = active_training_control() {
        let _ = handle.send_command(ProcessControlCommand::Stop);
    } else {
        let _ = send_training_control("STOP\n");
    }
    let _ = tauri::async_runtime::block_on(stop_training());

    let deadline = Instant::now() + Duration::from_millis(2500);
    while Instant::now() < deadline {
        let still_running = state.lock().map(|g| g.is_running).unwrap_or(false);
        if !still_running {
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }

    // Ensure no orphaned child process remains when the TUI exits after Ctrl+C / [q].
    force_kill_active_child();
}

pub fn handle_training_key(
    state: &Arc<Mutex<AppState>>,
    code: KeyCode,
    clipboard: &mut Option<Clipboard>,
) -> bool {
    let active_overlay = state.lock().map(|g| g.overlay).unwrap_or(OverlayMode::None);
    if active_overlay != OverlayMode::None {
        match active_overlay {
            OverlayMode::Help => {
                if matches!(code, KeyCode::Esc | KeyCode::F(1) | KeyCode::Char('?')) {
                    if let Ok(mut guard) = state.lock() {
                        guard.overlay = OverlayMode::None;
                    }
                }
            }
            OverlayMode::Config => {
                if matches!(code, KeyCode::Esc | KeyCode::F(3)) {
                    if let Ok(mut guard) = state.lock() {
                        guard.overlay = OverlayMode::None;
                    }
                }
            }
            OverlayMode::History => match code {
                KeyCode::Esc | KeyCode::F(4) | KeyCode::Char('h') | KeyCode::Char('H') => {
                    if let Ok(mut guard) = state.lock() {
                        guard.overlay = OverlayMode::None;
                    }
                }
                KeyCode::Up => {
                    if let Ok(mut guard) = state.lock() {
                        guard.history_selected = guard.history_selected.saturating_sub(1);
                    }
                }
                KeyCode::Down => {
                    if let Ok(mut guard) = state.lock() {
                        if !guard.history_entries.is_empty() {
                            let max_idx = guard.history_entries.len() - 1;
                            guard.history_selected = (guard.history_selected + 1).min(max_idx);
                        }
                    }
                }
                KeyCode::Enter | KeyCode::Char('y') | KeyCode::Char('Y') => {
                    let selected_path = state.lock().ok().and_then(|g| {
                        g.history_entries
                            .get(g.history_selected)
                            .map(|e| e.run_dir.clone())
                    });
                    if let Some(path) = selected_path {
                        match copy_text_with_fallback(clipboard, &path) {
                            Ok(CopiedTo::Clipboard) => {
                                if let Ok(mut guard) = state.lock() {
                                    guard.status_line =
                                        format!("Copied run path to clipboard: {path}");
                                }
                            }
                            Ok(CopiedTo::File(file_path)) => {
                                let shown = file_path.display().to_string();
                                announce_clipboard_fallback(state, &shown);
                                if let Ok(mut guard) = state.lock() {
                                    guard.status_line = format!(
                                        "Clipboard unavailable; wrote selection to {shown}"
                                    );
                                }
                            }
                            Err(err) => {
                                if let Ok(mut guard) = state.lock() {
                                    guard.status_line = format!("Clipboard copy failed: {err}");
                                }
                            }
                        }
                    }
                }
                _ => {}
            },
            OverlayMode::Browse | OverlayMode::None => {}
        }
        return false;
    }

    match code {
        KeyCode::F(1) | KeyCode::Char('?') => {
            if let Ok(mut guard) = state.lock() {
                guard.overlay = OverlayMode::Help;
            }
        }
        KeyCode::F(3) => {
            if let Ok(mut guard) = state.lock() {
                guard.overlay = OverlayMode::Config;
            }
        }
        KeyCode::F(4) | KeyCode::Char('h') | KeyCode::Char('H') => {
            let entries = scan_run_history();
            if let Ok(mut guard) = state.lock() {
                guard.history_entries = entries;
                guard.history_selected = 0;
                guard.overlay = OverlayMode::History;
            }
        }
        KeyCode::Char('q') | KeyCode::Char('Q') => {
            graceful_stop_and_wait(state);
            if let Ok(mut guard) = state.lock() {
                guard.refresh_run_artifacts();
            }
            return true;
        }
        KeyCode::Char(' ') => {
            let currently_paused = state.lock().map(|g| g.is_paused).unwrap_or(false);
            let cmd = if currently_paused {
                ProcessControlCommand::Resume
            } else {
                ProcessControlCommand::Pause
            };
            let sent = if let Some(handle) = active_training_control() {
                handle.send_command(cmd).is_ok()
            } else {
                send_training_control(cmd.as_wire_line()).unwrap_or(false)
            };
            if let Ok(mut guard) = state.lock() {
                if sent || guard.is_running {
                    guard.is_paused = !currently_paused;
                    guard.status_line = if guard.is_paused {
                        "Paused".to_string()
                    } else {
                        "Training in progress...".to_string()
                    };
                }
            }
        }
        KeyCode::Char('c') | KeyCode::Char('C') => {
            let sent = if let Some(handle) = active_training_control() {
                handle
                    .send_command(ProcessControlCommand::Checkpoint)
                    .is_ok()
            } else {
                send_training_control("CHECKPOINT\n").unwrap_or(false)
            };
            if let Ok(mut guard) = state.lock() {
                if sent {
                    guard.push_log(
                        "[LS-Trainer] Requested immediate checkpoint save (CHECKPOINT)."
                            .to_string(),
                    );
                }
            }
        }
        KeyCode::Up => {
            if let Ok(mut guard) = state.lock() {
                guard.follow_logs = false;
                guard.log_scroll_offset = guard.log_scroll_offset.saturating_add(1);
            }
        }
        KeyCode::Down => {
            if let Ok(mut guard) = state.lock() {
                guard.log_scroll_offset = guard.log_scroll_offset.saturating_sub(1);
                if guard.log_scroll_offset == 0 {
                    guard.follow_logs = true;
                }
            }
        }
        KeyCode::F(12) => {
            copy_logs_with_feedback(state, clipboard);
        }
        _ => {}
    }
    false
}

pub fn copy_logs_with_feedback(state: &Arc<Mutex<AppState>>, clipboard: &mut Option<Clipboard>) {
    let (logs_text, line_count) = {
        let Ok(guard) = state.lock() else {
            return;
        };
        let count = guard.logs.len();
        let text = guard.logs.iter().cloned().collect::<Vec<_>>().join("\n");
        (text, count)
    };

    if line_count == 0 {
        if let Ok(mut guard) = state.lock() {
            guard.status_line = "No logs to copy.".to_string();
        }
        return;
    }

    match copy_text_with_fallback(clipboard, &logs_text) {
        Ok(CopiedTo::Clipboard) => {
            if let Ok(mut guard) = state.lock() {
                guard.status_line = format!("Copied {line_count} log lines to clipboard (F12)");
                guard.push_log(format!(
                    "[LS-Trainer] Copied {line_count} log lines to clipboard."
                ));
            }
        }
        Ok(CopiedTo::File(path)) => {
            let shown = path.display().to_string();
            if let Ok(mut guard) = state.lock() {
                guard.status_line =
                    format!("Clipboard unavailable; wrote {line_count} log lines to {shown}");
            }
            announce_clipboard_fallback(state, &shown);
        }
        Err(err) => {
            if let Ok(mut guard) = state.lock() {
                guard.status_line = format!("Clipboard copy failed: {err}");
            }
        }
    }
}

fn announce_clipboard_fallback(state: &Arc<Mutex<AppState>>, path: &str) {
    if CLIPBOARD_FALLBACK_ANNOUNCED.swap(true, Ordering::Relaxed) {
        return;
    }
    if let Ok(mut guard) = state.lock() {
        guard.push_log(format!(
            "[LS-Trainer] Clipboard unavailable. Saving a copy to {path}."
        ));
    }
}

fn copy_text_with_fallback(
    clipboard: &mut Option<Clipboard>,
    text: &str,
) -> Result<CopiedTo, String> {
    match copy_text_to_clipboard(clipboard, text) {
        Ok(()) => Ok(CopiedTo::Clipboard),
        Err(err) => {
            let path = clipboard_fallback_path();
            std::fs::write(&path, text).map_err(|io_err| {
                format!("{err}; also failed to write {}: {io_err}", path.display())
            })?;
            Ok(CopiedTo::File(path))
        }
    }
}

pub fn copy_text_to_clipboard(clipboard: &mut Option<Clipboard>, text: &str) -> Result<(), String> {
    if CLIPBOARD_UNAVAILABLE.load(Ordering::Relaxed) {
        return Err("clipboard unavailable".to_string());
    }
    if clipboard.is_none() {
        match Clipboard::new() {
            Ok(cb) => *clipboard = Some(cb),
            Err(err) => {
                CLIPBOARD_UNAVAILABLE.store(true, Ordering::Relaxed);
                return Err(err.to_string());
            }
        }
    }
    let cb = clipboard
        .as_mut()
        .ok_or_else(|| "clipboard unavailable".to_string())?;
    match cb.set_text(text.to_owned()) {
        Ok(()) => Ok(()),
        Err(err) => {
            CLIPBOARD_UNAVAILABLE.store(true, Ordering::Relaxed);
            *clipboard = None;
            Err(err.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CLIPBOARD_FALLBACK_ANNOUNCED, CLIPBOARD_UNAVAILABLE, clipboard_fallback_path,
        handle_training_key,
    };
    use crate::app::{AppState, UiMode, parse_cli_args_from};
    use crossterm::event::KeyCode;
    use std::sync::atomic::Ordering;
    use std::sync::{Arc, Mutex};

    #[test]
    fn f12_on_headless_writes_log_file_and_does_not_repeat_the_notice() {
        CLIPBOARD_UNAVAILABLE.store(false, Ordering::Relaxed);
        CLIPBOARD_FALLBACK_ANNOUNCED.store(false, Ordering::Relaxed);
        let path = clipboard_fallback_path();
        let _ = std::fs::remove_file(&path);

        let cfg = parse_cli_args_from(Vec::<String>::new())
            .expect("parse")
            .expect("cfg");
        let state = Arc::new(Mutex::new(AppState::new(cfg)));
        {
            let mut guard = state.lock().expect("state");
            guard.mode = UiMode::Training;
            guard.logs.push_back("alpha log line".to_string());
            guard.logs.push_back("beta log line".to_string());
        }

        let mut clipboard = None;
        handle_training_key(&state, KeyCode::F(12), &mut clipboard);

        let status = state.lock().expect("state").status_line.clone();
        let shown = path.display().to_string();
        assert!(
            status.contains(&shown),
            "status should name the fallback file, got {status}"
        );
        let body = std::fs::read_to_string(&path).expect("fallback file written");
        assert_eq!(body, "alpha log line\nbeta log line");

        let notices = |logs: &std::collections::VecDeque<String>| {
            logs.iter()
                .filter(|line| line.contains("Clipboard unavailable"))
                .count()
        };
        assert_eq!(notices(&state.lock().expect("state").logs), 1);

        handle_training_key(&state, KeyCode::F(12), &mut clipboard);
        let guard = state.lock().expect("state");
        assert_eq!(
            notices(&guard.logs),
            1,
            "clipboard fallback must not warn on every F12"
        );
        assert!(guard.status_line.contains(&shown));
        assert!(!guard.status_line.to_ascii_lowercase().contains("failed"));
    }
}
