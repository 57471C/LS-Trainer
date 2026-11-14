//! `start_training` entry point plus run metadata and metrics shared by both backends.

pub(crate) mod rfdetr;
pub(crate) mod rtdetr;
pub(crate) mod yolo;

use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use tauri::ipc::Channel;

use crate::models::{TrainingRequest, TrainingResponse};
use crate::process::{
    ProcessControlHandle, TauriChannelSink, TrainingEventSink, get_project_root,
    hide_console_window, send_break_signal, to_display_path,
};
use crate::rtdetr::is_rtdetr_request;
use crate::settings::{
    describe_write_access, ensure_data_directory_writable, nearest_existing_dir, runs_base_dir,
};
use crate::training::rfdetr::start_rfdetr_training;
use crate::training::rtdetr::start_rtdetr_training;
use crate::training::yolo::start_yolo_training;

pub struct ActiveTrainingState {
    pub is_running: AtomicBool,
    pub stop_requested: AtomicBool,
    pub results_dir: Mutex<Option<PathBuf>>,
    pub child_pid: Mutex<Option<u32>>,
    pub child_control: Mutex<Option<ProcessControlHandle>>,
}

pub static ACTIVE_TRAINING: ActiveTrainingState = ActiveTrainingState {
    is_running: AtomicBool::new(false),
    stop_requested: AtomicBool::new(false),
    results_dir: Mutex::new(None),
    child_pid: Mutex::new(None),
    child_control: Mutex::new(None),
};

pub(crate) fn register_active_training(results_dir: PathBuf) {
    ACTIVE_TRAINING.is_running.store(true, Ordering::SeqCst);
    ACTIVE_TRAINING
        .stop_requested
        .store(false, Ordering::SeqCst);
    if let Ok(mut lock) = ACTIVE_TRAINING.results_dir.lock() {
        let stop_file = results_dir.join(".stop_requested");
        let _ = std::fs::remove_file(stop_file);
        *lock = Some(results_dir);
    }
}

pub(crate) fn register_child_pid(pid: u32) {
    if let Ok(mut lock) = ACTIVE_TRAINING.child_pid.lock() {
        *lock = Some(pid);
    }
}

pub(crate) fn register_child_process(child: &mut Child) -> ProcessControlHandle {
    register_child_pid(child.id());
    let handle = ProcessControlHandle::from_child(child);
    if let Ok(mut lock) = ACTIVE_TRAINING.child_control.lock() {
        *lock = Some(handle.clone());
    }
    handle
}

pub fn active_training_control() -> Option<ProcessControlHandle> {
    ACTIVE_TRAINING
        .child_control
        .lock()
        .ok()
        .and_then(|g| g.clone())
}

pub fn send_training_control(command: &str) -> Result<bool, String> {
    if !ACTIVE_TRAINING.is_running.load(Ordering::SeqCst) {
        return Ok(false);
    }
    if let Some(handle) = active_training_control() {
        handle.write_raw(command)?;
        return Ok(true);
    }
    Ok(false)
}

pub(crate) fn unregister_child_pid() {
    if let Ok(mut lock) = ACTIVE_TRAINING.child_pid.lock() {
        *lock = None;
    }
    if let Ok(mut lock) = ACTIVE_TRAINING.child_control.lock() {
        if let Some(ref handle) = *lock {
            handle.clear();
        }
        *lock = None;
    }
}

pub(crate) fn clear_active_training() {
    ACTIVE_TRAINING.is_running.store(false, Ordering::SeqCst);
    if let Ok(mut lock) = ACTIVE_TRAINING.child_pid.lock() {
        *lock = None;
    }
    if let Ok(mut lock) = ACTIVE_TRAINING.child_control.lock() {
        if let Some(ref handle) = *lock {
            handle.clear();
        }
        *lock = None;
    }
    if let Ok(mut lock) = ACTIVE_TRAINING.results_dir.lock() {
        *lock = None;
    }
}

pub(crate) fn is_stop_requested() -> bool {
    ACTIVE_TRAINING.stop_requested.load(Ordering::SeqCst)
}

#[tauri::command]
pub async fn stop_training() -> Result<bool, String> {
    if !ACTIVE_TRAINING.is_running.load(Ordering::SeqCst) {
        return Ok(false);
    }

    ACTIVE_TRAINING.stop_requested.store(true, Ordering::SeqCst);

    // 1. Write .stop_requested file to active run folder for file-based pollers
    if let Ok(guard) = ACTIVE_TRAINING.results_dir.lock() {
        if let Some(ref dir) = *guard {
            let _ = std::fs::create_dir_all(dir);
            let _ = std::fs::write(dir.join(".stop_requested"), "stop");
        }
    }

    // 1b. Write STOP\n to piped child stdin if attached
    if let Some(handle) = active_training_control() {
        let _ = handle.stop();
    }

    // 2. Send break signal to active child process group
    let pid_opt = ACTIVE_TRAINING.child_pid.lock().ok().and_then(|g| *g);
    if let Some(pid) = pid_opt {
        send_break_signal(pid);

        // 3. Fallback watchdog: if training is still running after 6 seconds, terminate process
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(6));
            if ACTIVE_TRAINING.is_running.load(Ordering::SeqCst) {
                #[cfg(windows)]
                {
                    let mut cmd = Command::new("taskkill");
                    cmd.args(["/PID", &pid.to_string(), "/T", "/F"]);
                    hide_console_window(&mut cmd);
                    let _ = cmd.spawn();
                }
                #[cfg(not(windows))]
                {
                    let _ = Command::new("kill").args(["-9", &pid.to_string()]).spawn();
                }
            }
        });
    }

    Ok(true)
}

#[tauri::command]
pub async fn start_training(
    _app: tauri::AppHandle,
    on_log: Channel<String>,
    request: TrainingRequest,
) -> Result<TrainingResponse, String> {
    let sink: Arc<dyn TrainingEventSink> = Arc::new(TauriChannelSink::new(on_log));
    run_training(sink, request).await
}

pub async fn run_training(
    sink: Arc<dyn TrainingEventSink>,
    request: TrainingRequest,
) -> Result<TrainingResponse, String> {
    let run_name = if request.run_name.trim().is_empty() {
        "BB-Parts-v3".to_string()
    } else {
        request.run_name.trim().to_string()
    };

    let project_root = get_project_root();
    let yolo_task = resolve_train_task(&request);

    // Prefer configured data directory for runs; fall back to project root.
    let base = runs_base_dir();
    let project_runs_dir = base.join("runs").join(&yolo_task);
    let abs_results_dir = project_runs_dir.join(&run_name);
    let results_dir = to_display_path(&abs_results_dir);

    sink.on_log(format!(
        "[LS-Trainer] Initializing training run '{run_name}'..."
    ));
    sink.on_log(format!("[LS-Trainer] Results directory: {}", results_dir));

    // Do not create the data directory here. A missing `/mnt/datasets` is usually an
    // unmounted disk, and creating it would hide that mount.
    if let Err(err) = require_existing_directory(&base) {
        sink.on_log(format!("[LS-Trainer] Error: {err}"));
        return Err(err);
    }
    if let Err(err) = ensure_data_directory_writable(&base) {
        sink.on_log(format!("[LS-Trainer] Error: {err}"));
        return Err(err);
    }

    // Parent must exist before a backend writes its runner script into the run folder.
    // Creating the run folder itself is left to the backend: Ultralytics treats an
    // existing name as taken and increments it.
    if let Some(parent) = abs_results_dir.parent() {
        if let Err(err) = ensure_directory(parent) {
            sink.on_log(format!("[LS-Trainer] Error: {err}"));
            return Err(err);
        }
    }

    let dataset_input = PathBuf::from(&request.dataset_path);
    if !dataset_input.exists() {
        let msg = format!("Dataset path does not exist: {}", request.dataset_path);
        sink.on_log(format!("[LS-Trainer] Error: {msg}"));
        return Err(msg);
    }

    register_active_training(abs_results_dir.clone());

    let response = if is_rtdetr_request(&request) {
        start_rtdetr_training(
            project_root,
            run_name,
            yolo_task,
            abs_results_dir,
            results_dir,
            dataset_input,
            request,
            sink,
        )
        .await
    } else if is_rfdetr_request(&request) {
        start_rfdetr_training(
            project_root,
            run_name,
            yolo_task,
            abs_results_dir,
            results_dir,
            dataset_input,
            request,
            sink,
        )
        .await
    } else {
        start_yolo_training(
            project_root,
            run_name,
            yolo_task,
            abs_results_dir,
            results_dir,
            dataset_input,
            request,
            sink,
        )
        .await
    };

    clear_active_training();
    response
}

/// The configured data directory must already exist. Creating a missing mount
/// such as `/mnt/datasets` would hide the unmounted disk.
pub(crate) fn require_existing_directory(path: &Path) -> Result<(), String> {
    if path.as_os_str().is_empty() {
        return Err("Data directory path is empty".to_string());
    }

    match std::fs::symlink_metadata(path) {
        Ok(meta) => {
            if path.is_dir() {
                return Ok(());
            }
            if meta.file_type().is_symlink() {
                return Err(format!(
                    "Data directory '{}' is a broken symlink. Mount the target, or choose another folder in Settings.",
                    path.display()
                ));
            }
            Err(format!(
                "Data directory '{}' exists but is not a directory.",
                path.display()
            ))
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Err(format!(
            "Data directory '{}' does not exist or is not mounted. Mount it, or choose another folder in Settings, before training.",
            path.display()
        )),
        Err(err) => Err(format!(
            "Data directory '{}' is not accessible: {err}",
            path.display()
        )),
    }
}

/// Create `path` and its parents. The error names the directory so a missing
/// mount such as `/mnt/datasets` is not reported later as a missing script file.
pub(crate) fn ensure_directory(path: &Path) -> Result<(), String> {
    if path.as_os_str().is_empty() {
        return Err("Could not create results directory: path is empty".to_string());
    }
    if path.is_dir() {
        return Ok(());
    }

    let mut last_error: Option<String> = None;
    for attempt in 0..3 {
        match std::fs::create_dir_all(path) {
            Ok(()) if path.is_dir() => return Ok(()),
            Ok(()) => {
                last_error = Some(format!(
                    "Could not create results directory '{}': path is not a directory.",
                    path.display()
                ));
            }
            Err(err) => {
                last_error = Some(directory_error(path, &err));
                if err.kind() != std::io::ErrorKind::NotFound {
                    break;
                }
            }
        }
        if attempt < 2 {
            std::thread::sleep(std::time::Duration::from_millis(30));
        }
    }
    Err(last_error
        .unwrap_or_else(|| format!("Could not create results directory '{}'.", path.display())))
}

/// Write `contents` to `path`, creating the parent directory first.
/// A short retry covers a network mount that returns ENOENT just after mkdir.
pub(crate) fn write_text_file(path: &Path, contents: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            ensure_directory(parent)?;
        }
    }

    let mut last_error = String::new();
    for attempt in 0..3 {
        match std::fs::write(path, contents) {
            Ok(()) => return Ok(()),
            Err(err) => {
                last_error = err.to_string();
                let retry = err.kind() == std::io::ErrorKind::NotFound && attempt < 2;
                if !retry {
                    return Err(format!("'{}': {last_error}", path.display()));
                }
                if let Some(parent) = path.parent() {
                    if !parent.as_os_str().is_empty() {
                        let _ = ensure_directory(parent);
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(30));
            }
        }
    }
    Err(format!("'{}': {last_error}", path.display()))
}

fn directory_error(path: &Path, err: &std::io::Error) -> String {
    let mut message = format!(
        "Could not create results directory '{}': {err}.",
        path.display()
    );
    if err.kind() == std::io::ErrorKind::PermissionDenied {
        if let Some(existing) = nearest_existing_dir(path) {
            message.push_str(&describe_write_access(&existing));
        }
        message.push_str(" Training needs to create folders inside the data directory.");
        return message;
    }
    if let Some(missing) = first_missing_component(path) {
        if missing != path {
            message.push_str(&format!(" Missing '{}'.", missing.display()));
        }
    }
    message.push_str(" Check that this data directory exists, is mounted, and is writable.");
    message
}

/// Highest ancestor of `path` that is not present, so a missing `/mnt/datasets`
/// is named instead of the run folder underneath it.
fn first_missing_component(path: &Path) -> Option<PathBuf> {
    let mut current = path;
    let mut missing: Option<PathBuf> = None;
    loop {
        match std::fs::symlink_metadata(current) {
            Ok(meta) => {
                if meta.file_type().is_symlink() && !current.is_dir() {
                    return Some(current.to_path_buf());
                }
                break;
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                missing = Some(current.to_path_buf());
                match current.parent() {
                    Some(parent) if parent != current => current = parent,
                    _ => break,
                }
            }
            Err(_) => break,
        }
    }
    missing
}

pub(crate) fn resolve_train_task(request: &TrainingRequest) -> String {
    let model = request.model_choice.to_ascii_lowercase();
    if model.contains("-seg") {
        return "segment".to_string();
    }
    if request.task.eq_ignore_ascii_case("segment") {
        return "segment".to_string();
    }
    "detect".to_string()
}

pub(crate) fn save_run_config_metadata(
    abs_results_dir: &Path,
    request: &TrainingRequest,
    resolved_model: &str,
    effective_device: &str,
    effective_workers: u32,
    exact_command: &str,
    sink: &Arc<dyn TrainingEventSink>,
) {
    if std::fs::create_dir_all(abs_results_dir).is_err() {
        return;
    }

    let config_path = abs_results_dir.join("ls_trainer_config.json");
    let effective_freeze = if request.freeze_backbone {
        request.freeze_layers.max(1)
    } else {
        0
    };

    let payload = serde_json::json!({
        "run_name": request.run_name,
        "dataset_path": request.dataset_path,
        "model_family": request.model_family,
        "model_choice": request.model_choice,
        "resolved_model": resolved_model,
        "task": resolve_train_task(request),
        "freeze_backbone": request.freeze_backbone,
        "freeze_layers": effective_freeze,
        "epochs": request.epochs,
        "batch": request.batch,
        "imgsz": request.imgsz,
        "device": effective_device,
        "workers": effective_workers,
        "workers_override": request.workers,
        "amp": request.amp,
        "augmentation": {
            "mosaic": request.mosaic,
            "mixup": request.mixup,
            "copy_paste": request.copy_paste,
            "degrees": request.degrees,
            "translate": request.translate,
            "scale": request.scale,
            "flipud": request.flipud,
            "close_mosaic": request.close_mosaic,
        },
        "command": exact_command,
    });

    if let Ok(serialized) = serde_json::to_string_pretty(&payload) {
        if std::fs::write(&config_path, serialized).is_ok() {
            sink.on_log(format!(
                "[LS-Trainer] Saved run settings to: {}",
                config_path.display()
            ));
        }
    }
}

pub(crate) fn is_rfdetr_request(request: &TrainingRequest) -> bool {
    if is_rtdetr_request(request) {
        return false;
    }
    if request.model_family.eq_ignore_ascii_case("RF-DETR")
        || request.model_family.eq_ignore_ascii_case("RFDETR")
    {
        return true;
    }
    let choice = request.model_choice.trim().to_ascii_lowercase();
    choice.starts_with("rf-detr") || choice.starts_with("rfdetr") || choice.ends_with(".pth")
}

pub(crate) fn extract_final_metrics_from_results_dir(
    abs_results_dir: &Path,
) -> (Option<f64>, Option<f64>) {
    let metrics_json_path = abs_results_dir.join("ls_trainer_metrics.json");
    if metrics_json_path.is_file() {
        if let Ok(text) = std::fs::read_to_string(&metrics_json_path) {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&text) {
                let map50 = val.get("map50").and_then(|v| v.as_f64());
                let map50_95 = val.get("map50_95").and_then(|v| v.as_f64());
                if map50.is_some() || map50_95.is_some() {
                    return (map50, map50_95);
                }
            }
        }
    }

    // Fallback: parse Ultralytics results.csv or RF-DETR metrics.csv
    for csv_name in ["results.csv", "metrics.csv"] {
        let csv_path = abs_results_dir.join(csv_name);
        if !csv_path.is_file() {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(&csv_path) else {
            continue;
        };
        let mut lines = content.lines().filter(|l| !l.trim().is_empty());
        let Some(header_line) = lines.next() else {
            continue;
        };
        let headers: Vec<String> = header_line
            .split(',')
            .map(|h| h.trim().to_ascii_lowercase())
            .collect();

        let idx_map50 = headers.iter().position(|h| {
            h == "metrics/map50(b)"
                || h == "metrics/map50(m)"
                || h == "val/map_50"
                || h == "val/ema_map_50"
                || h == "val/segm_map_50"
        });
        let idx_map50_95 = headers.iter().position(|h| {
            h == "metrics/map50-95(b)"
                || h == "metrics/map50-95(m)"
                || h == "val/map_50_95"
                || h == "val/ema_map_50_95"
                || h == "val/segm_map_50_95"
        });

        let mut best_50: Option<f64> = None;
        let mut best_50_95: Option<f64> = None;
        for row in lines {
            let cols: Vec<&str> = row.split(',').map(|c| c.trim()).collect();
            if let Some(i50) = idx_map50 {
                if let Some(v) = cols.get(i50).and_then(|s| s.parse::<f64>().ok()) {
                    best_50 = Some(best_50.map_or(v, |prev| prev.max(v)));
                }
            }
            if let Some(i95) = idx_map50_95 {
                if let Some(v) = cols.get(i95).and_then(|s| s.parse::<f64>().ok()) {
                    best_50_95 = Some(best_50_95.map_or(v, |prev| prev.max(v)));
                }
            }
        }
        if best_50.is_some() || best_50_95.is_some() {
            return (best_50, best_50_95);
        }
    }

    (None, None)
}

/// Log the best mAP50 / mAP50-95 for a finished run when both are known.
pub(crate) fn log_final_metrics(
    sink: &Arc<dyn TrainingEventSink>,
    map50: Option<f64>,
    map50_95: Option<f64>,
) {
    if let (Some(m50), Some(m95)) = (map50, map50_95) {
        sink.on_log(format!(
            "[LS-Trainer] Final Metrics — mAP50: {:.1}% ({:.4}) | mAP50-95: {:.1}% ({:.4})",
            m50 * 100.0,
            m50,
            m95 * 100.0,
            m95
        ));
    }
}
