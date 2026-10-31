//! `start_training` entry point plus run metadata and metrics shared by both backends.

pub(crate) mod rfdetr;
pub(crate) mod yolo;

use std::path::{Path, PathBuf};
use tauri::ipc::Channel;

use crate::models::{TrainingRequest, TrainingResponse};
use crate::process::{get_project_root, to_display_path};
use crate::training::rfdetr::start_rfdetr_training;
use crate::training::yolo::start_yolo_training;

#[tauri::command]
pub async fn start_training(
    _app: tauri::AppHandle,
    on_log: Channel<String>,
    request: TrainingRequest,
) -> Result<TrainingResponse, String> {
    let run_name = if request.run_name.trim().is_empty() {
        "BB-Parts-v3".to_string()
    } else {
        request.run_name.trim().to_string()
    };

    let project_root = get_project_root();
    let yolo_task = resolve_train_task(&request);
    let project_runs_dir = project_root.join("runs").join(&yolo_task);
    let abs_results_dir = project_runs_dir.join(&run_name);
    let results_dir = to_display_path(&abs_results_dir);

    let _ = on_log.send(format!(
        "[LS-Trainer] Initializing training run '{run_name}'..."
    ));

    let dataset_input = PathBuf::from(&request.dataset_path);
    if !dataset_input.exists() {
        let msg = format!("Dataset path does not exist: {}", request.dataset_path);
        let _ = on_log.send(format!("[LS-Trainer] Error: {msg}"));
        return Err(msg);
    }

    if is_rfdetr_request(&request) {
        return start_rfdetr_training(
            project_root,
            run_name,
            yolo_task,
            abs_results_dir,
            results_dir,
            dataset_input,
            request,
            on_log,
        )
        .await;
    }

    start_yolo_training(
        project_root,
        run_name,
        yolo_task,
        abs_results_dir,
        results_dir,
        dataset_input,
        request,
        on_log,
    )
    .await
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
    on_log: &Channel<String>,
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
            let _ = on_log.send(format!(
                "[LS-Trainer] Saved run settings to: {}",
                config_path.display()
            ));
        }
    }
}

pub(crate) fn is_rfdetr_request(request: &TrainingRequest) -> bool {
    if request.model_family.eq_ignore_ascii_case("RF-DETR")
        || request.model_family.eq_ignore_ascii_case("RFDETR")
    {
        return true;
    }
    let choice = request.model_choice.trim().to_ascii_lowercase();
    choice.starts_with("rf-detr") || choice.starts_with("rfdetr") || choice.ends_with(".pth")
}

pub(crate) fn extract_final_metrics_from_results_dir(abs_results_dir: &Path) -> (Option<f64>, Option<f64>) {
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
pub(crate) fn log_final_metrics(on_log: &Channel<String>, map50: Option<f64>, map50_95: Option<f64>) {
    if let (Some(m50), Some(m95)) = (map50, map50_95) {
        let _ = on_log.send(format!(
            "[LS-Trainer] Final Metrics — mAP50: {:.1}% ({:.4}) | mAP50-95: {:.1}% ({:.4})",
            m50 * 100.0,
            m50,
            m95 * 100.0,
            m95
        ));
    }
}
