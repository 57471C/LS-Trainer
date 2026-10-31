//! RF-DETR ONNX export (runs `python/rfdetr_export.py`).

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use tauri::ipc::Channel;

use crate::hardware::is_cuda_available;
use crate::models::{ExportRequest, ExportResponse, default_export_imgsz};
use crate::process::{get_venv_python, hide_console_window, stream_child_output, to_display_path};
use crate::rfdetr::{ensure_rfdetr_installed, resolve_rfdetr_class_and_pretrain, snap_rfdetr_resolution};

pub(crate) async fn export_rfdetr_to_onnx(
    project_root: &Path,
    results_path: &Path,
    checkpoint_path: &Path,
    request: &ExportRequest,
    on_log: &Channel<String>,
) -> Result<ExportResponse, String> {
    ensure_rfdetr_installed(project_root, on_log);

    let output_dir = if results_path.is_dir() {
        results_path.to_path_buf()
    } else {
        checkpoint_path
            .parent()
            .unwrap_or(project_root)
            .to_path_buf()
    };

    // Determine RF-DETR model class and task from saved run metadata if available
    let mut model_choice = "rf-detr-medium".to_string();
    let mut task = "detect".to_string();
    let config_path = output_dir.join("ls_trainer_config.json");
    if config_path.is_file() {
        if let Ok(text) = std::fs::read_to_string(&config_path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                if let Some(mc) = json.get("model_choice").and_then(|v| v.as_str()) {
                    model_choice = mc.to_string();
                }
                if let Some(t) = json.get("task").and_then(|v| v.as_str()) {
                    task = t.to_string();
                }
            }
        }
    }

    let (rfdetr_class, _, block_size) = resolve_rfdetr_class_and_pretrain(&model_choice, &task);
    let raw_imgsz = if request.imgsz == 0 {
        default_export_imgsz()
    } else {
        request.imgsz
    };
    let export_res = snap_rfdetr_resolution(raw_imgsz, block_size);
    let device = if is_cuda_available(project_root) {
        "cuda"
    } else {
        "cpu"
    };

    let ckpt_display = checkpoint_path.to_string_lossy().replace('\\', "/");
    let _ = on_log.send(format!(
        "[LS-Trainer] Found RF-DETR checkpoint: {ckpt_display}"
    ));
    let _ = on_log.send(format!(
        "[LS-Trainer] Exporting {rfdetr_class} to ONNX (shape=({export_res}, {export_res}), dynamic={}, simplify={})...",
        request.dynamic, request.simplify
    ));

    let export_script_path = output_dir.join("ls_trainer_rfdetr_export.py");
    let export_script = include_str!("../python/rfdetr_export.py");

    std::fs::write(&export_script_path, export_script)
        .map_err(|e| format!("Failed to write RF-DETR ONNX export script: {e}"))?;

    let export_cfg_path = output_dir.join("ls_trainer_rfdetr_export_args.json");
    let export_cfg = serde_json::json!({
        "rfdetr_class": rfdetr_class,
        "checkpoint_path": checkpoint_path.to_string_lossy(),
        "output_dir": output_dir.to_string_lossy(),
        "resolution": export_res,
        "dynamic": request.dynamic,
        "simplify": request.simplify,
        "device": device,
        "task": task,
    });
    std::fs::write(
        &export_cfg_path,
        serde_json::to_string_pretty(&export_cfg).unwrap_or_default(),
    )
    .map_err(|e| format!("Failed to write RF-DETR export config: {e}"))?;

    let python_exe = get_venv_python(project_root).unwrap_or_else(|| PathBuf::from("python"));
    let mut cmd = Command::new(&python_exe);
    cmd.args([
        export_script_path.to_string_lossy().as_ref(),
        export_cfg_path.to_string_lossy().as_ref(),
    ])
    .current_dir(project_root)
    .env("PYTHONUNBUFFERED", "1")
    .env("PYTHONUTF8", "1")
    .env("PYTHONIOENCODING", "utf-8")
    .stdout(Stdio::piped())
    .stderr(Stdio::piped());

    hide_console_window(&mut cmd);

    match cmd.spawn() {
        Ok(mut child) => {
            stream_child_output(&mut child, &on_log);

            let status = child
                .wait()
                .map_err(|e| format!("Failed while waiting for RF-DETR ONNX export: {e}"))?;

            let expected_onnx = output_dir.join("best.onnx");
            let onnx_str = to_display_path(&expected_onnx);
            if status.success() && expected_onnx.is_file() {
                let _ = on_log.send(format!(
                    "[LS-Trainer] ONNX export completed successfully! File: {onnx_str}"
                ));
                let _ = on_log.send(
                    "[LS-Trainer] Note: This model is optimized and ready for inference in LS-Vision."
                        .to_string(),
                );
                Ok(ExportResponse {
                    success: true,
                    onnx_path: onnx_str,
                    message: "ONNX export completed successfully".to_string(),
                })
            } else {
                let err_msg = format!(
                    "RF-DETR ONNX export failed with status code: {:?}",
                    status.code()
                );
                let _ = on_log.send(format!("[LS-Trainer Error] {err_msg}"));
                Ok(ExportResponse {
                    success: false,
                    onnx_path: String::new(),
                    message: err_msg,
                })
            }
        }
        Err(err) => Err(format!("Failed to launch RF-DETR ONNX export: {err}")),
    }
}
