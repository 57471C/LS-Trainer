//! RT-DETR ONNX export (runs `python/rtdetr_export.py`).

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use tauri::ipc::Channel;

use crate::hardware::is_cuda_available;
use crate::models::{ExportRequest, ExportResponse, default_export_imgsz};
use crate::process::{get_venv_python, hide_console_window, stream_child_output, to_display_path};

pub(crate) async fn export_rtdetr_to_onnx(
    project_root: &Path,
    results_path: &Path,
    checkpoint_path: &Path,
    request: &ExportRequest,
    on_log: &Channel<String>,
) -> Result<ExportResponse, String> {
    let output_dir = if results_path.is_dir() {
        results_path.to_path_buf()
    } else {
        checkpoint_path
            .parent()
            .unwrap_or(project_root)
            .to_path_buf()
    };

    let mut task = "detect".to_string();
    let config_path = output_dir.join("ls_trainer_config.json");
    if config_path.is_file() {
        if let Ok(text) = std::fs::read_to_string(&config_path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                if let Some(t) = json.get("task").and_then(|v| v.as_str()) {
                    task = t.to_string();
                }
            }
        }
    }

    let export_res = if request.imgsz == 0 {
        default_export_imgsz()
    } else {
        request.imgsz
    };

    let device = if is_cuda_available(project_root) {
        "cuda"
    } else {
        "cpu"
    };

    let ckpt_display = checkpoint_path.to_string_lossy().replace('\\', "/");
    let _ = on_log.send(format!(
        "[LS-Trainer] Found RT-DETR checkpoint: {ckpt_display}"
    ));
    let _ = on_log.send(format!(
        "[LS-Trainer] Exporting RT-DETR v2-R18 to ONNX (shape=({export_res}, {export_res}), dynamic={}, simplify={})...",
        request.dynamic, request.simplify
    ));

    let export_script_path = output_dir.join("ls_trainer_rtdetr_export.py");
    let export_script = include_str!("../python/rtdetr_export.py");

    std::fs::write(&export_script_path, export_script)
        .map_err(|e| format!("Failed to write RT-DETR ONNX export script: {e}"))?;

    let export_cfg_path = output_dir.join("ls_trainer_rtdetr_export_args.json");
    let export_cfg = serde_json::json!({
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
    .map_err(|e| format!("Failed to write RT-DETR export config: {e}"))?;

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
                .map_err(|e| format!("Failed while waiting for RT-DETR ONNX export: {e}"))?;

            let expected_onnx = output_dir.join("best.onnx");
            let expected_weights_onnx = output_dir.join("weights").join("best.onnx");
            let final_onnx = if expected_onnx.is_file() {
                Some(expected_onnx)
            } else if expected_weights_onnx.is_file() {
                Some(expected_weights_onnx)
            } else {
                None
            };

            if status.success() && final_onnx.is_some() {
                let onnx_path_ref = final_onnx.as_ref().unwrap();
                let onnx_str = to_display_path(onnx_path_ref);
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
                    "RT-DETR ONNX export failed with status code: {:?}",
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
        Err(err) => Err(format!("Failed to launch RT-DETR ONNX export: {err}")),
    }
}

