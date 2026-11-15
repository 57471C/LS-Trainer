use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use app_lib::{
    active_training_control, force_kill_active_child, send_training_control, stop_training,
    ProcessControlCommand, TrainingRequest,
};
use arboard::Clipboard;
use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};

use crate::app::{
    launch_training_thread, scan_run_history, AppState, OverlayMode, UiMode, DEVICE_PRESETS,
    LAUNCHER_FIELD_COUNT, WORKER_PRESETS,
};

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
                        .map(|g| !g.setup.editing_field)
                        .unwrap_or(false);
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

pub fn handle_setup_key(state: &Arc<Mutex<AppState>>, code: KeyCode) -> bool {
    let mut launch_req: Option<TrainingRequest> = None;
    if let Ok(mut guard) = state.lock() {
        let focused = guard.setup.focused_field;
        match code {
            KeyCode::Esc => {
                if guard.setup.editing_field {
                    guard.setup.editing_field = false;
                } else {
                    return true;
                }
            }
            KeyCode::F(2) => match guard.setup.to_training_request() {
                Ok(req) => {
                    guard.setup.editing_field = false;
                    guard.setup.error_banner = None;
                    launch_req = Some(req);
                }
                Err(err) => {
                    guard.setup.error_banner = Some(err);
                }
            },
            KeyCode::Tab | KeyCode::Down => {
                guard.setup.editing_field = false;
                guard.setup.focused_field = (focused + 1) % LAUNCHER_FIELD_COUNT;
            }
            KeyCode::BackTab | KeyCode::Up => {
                guard.setup.editing_field = false;
                guard.setup.focused_field = if focused == 0 {
                    LAUNCHER_FIELD_COUNT - 1
                } else {
                    focused - 1
                };
            }
            KeyCode::Left => match focused {
                0 => {
                    let prev_default = guard.setup.arch.default_model();
                    guard.setup.arch = guard.setup.arch.prev();
                    if guard.setup.model == prev_default {
                        guard.setup.model = guard.setup.arch.default_model().to_string();
                    }
                }
                4 if !guard.setup.editing_field => {
                    guard.setup.device = cycle_preset(&guard.setup.device, &DEVICE_PRESETS, false);
                }
                5 if !guard.setup.editing_field => {
                    guard.setup.workers_input =
                        cycle_preset(&guard.setup.workers_input, &WORKER_PRESETS, false);
                }
                _ => {}
            },
            KeyCode::Right => match focused {
                0 => {
                    let prev_default = guard.setup.arch.default_model();
                    guard.setup.arch = guard.setup.arch.next();
                    if guard.setup.model == prev_default {
                        guard.setup.model = guard.setup.arch.default_model().to_string();
                    }
                }
                4 if !guard.setup.editing_field => {
                    guard.setup.device = cycle_preset(&guard.setup.device, &DEVICE_PRESETS, true);
                }
                5 if !guard.setup.editing_field => {
                    guard.setup.workers_input =
                        cycle_preset(&guard.setup.workers_input, &WORKER_PRESETS, true);
                }
                _ => {}
            },
            KeyCode::Backspace => match focused {
                1 => {
                    guard.setup.editing_field = true;
                    guard.setup.dataset.pop();
                }
                2 => {
                    guard.setup.editing_field = true;
                    guard.setup.epochs.pop();
                }
                3 => {
                    guard.setup.editing_field = true;
                    guard.setup.batch.pop();
                }
                4 => {
                    guard.setup.editing_field = true;
                    guard.setup.device.pop();
                }
                5 => {
                    guard.setup.editing_field = true;
                    if guard.setup.workers_input.eq_ignore_ascii_case("auto") {
                        guard.setup.workers_input.clear();
                    } else {
                        guard.setup.workers_input.pop();
                    }
                }
                _ => {}
            },
            KeyCode::Char(c) => match focused {
                0 if c == ' ' => {
                    let prev_default = guard.setup.arch.default_model();
                    guard.setup.arch = guard.setup.arch.next();
                    if guard.setup.model == prev_default {
                        guard.setup.model = guard.setup.arch.default_model().to_string();
                    }
                }
                1 => {
                    guard.setup.editing_field = true;
                    guard.setup.dataset.push(c);
                }
                2 if c.is_ascii_digit() => {
                    guard.setup.editing_field = true;
                    guard.setup.epochs.push(c);
                }
                3 if c.is_ascii_digit() => {
                    guard.setup.editing_field = true;
                    guard.setup.batch.push(c);
                }
                4 => {
                    guard.setup.editing_field = true;
                    guard.setup.device.push(c);
                }
                5 => {
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
                _ => {}
            },
            KeyCode::Enter => {
                if focused == 6 {
                    match guard.setup.to_training_request() {
                        Ok(req) => {
                            guard.setup.editing_field = false;
                            guard.setup.error_banner = None;
                            launch_req = Some(req);
                        }
                        Err(err) => {
                            guard.setup.error_banner = Some(err);
                        }
                    }
                } else if focused == 0 {
                    let prev_default = guard.setup.arch.default_model();
                    guard.setup.arch = guard.setup.arch.next();
                    if guard.setup.model == prev_default {
                        guard.setup.model = guard.setup.arch.default_model().to_string();
                    }
                } else {
                    if guard.setup.editing_field
                        && focused == 5
                        && guard.setup.workers_input.trim().is_empty()
                    {
                        guard.setup.workers_input = "auto".to_string();
                    }
                    guard.setup.editing_field = !guard.setup.editing_field;
                }
            }
            _ => {}
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
                        let copied = copy_text_to_clipboard(clipboard, &path).is_ok();
                        if let Ok(mut guard) = state.lock() {
                            guard.status_line = if copied {
                                format!("Copied run path to clipboard: {path}")
                            } else {
                                format!("Selected run path: {path}")
                            };
                        }
                    }
                }
                _ => {}
            },
            OverlayMode::None => {}
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

    match copy_text_to_clipboard(clipboard, &logs_text) {
        Ok(()) => {
            if let Ok(mut guard) = state.lock() {
                guard.status_line = format!("Copied {line_count} log lines to clipboard (F12)");
                guard.push_log(format!(
                    "[LS-Trainer] Copied {line_count} log lines to clipboard."
                ));
            }
        }
        Err(err) => {
            if let Ok(mut guard) = state.lock() {
                guard.status_line = format!("Clipboard copy failed: {err}");
                guard.push_log(format!("[LS-Trainer Warning] Clipboard copy failed: {err}"));
            }
        }
    }
}

pub fn copy_text_to_clipboard(clipboard: &mut Option<Clipboard>, text: &str) -> Result<(), String> {
    if clipboard.is_none() {
        *clipboard = Some(Clipboard::new().map_err(|e| e.to_string())?);
    }
    let cb = clipboard
        .as_mut()
        .ok_or_else(|| "clipboard unavailable".to_string())?;
    cb.set_text(text.to_owned()).map_err(|e| e.to_string())
}
