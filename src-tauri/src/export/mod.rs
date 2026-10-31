//! `export_to_onnx` entry point and ONNX `ls_model_type` metadata embedding.

pub(crate) mod rfdetr;
pub(crate) mod yolo;

use std::path::{Path, PathBuf};
use std::process::Command;
use tauri::ipc::Channel;

use crate::export::rfdetr::export_rfdetr_to_onnx;
use crate::export::yolo::export_yolo_to_onnx;
use crate::models::{ExportRequest, ExportResponse};
use crate::process::{get_project_root, get_venv_python, hide_console_window};
use crate::rfdetr::find_rfdetr_checkpoint_in_dir;

#[tauri::command]
pub async fn export_to_onnx(
    _app: tauri::AppHandle,
    on_log: Channel<String>,
    request: ExportRequest,
) -> Result<ExportResponse, String> {
    let project_root = get_project_root();

    let mut results_path = PathBuf::from(&request.results_dir);
    if !results_path.is_absolute() {
        results_path = project_root.join(&results_path);
    }

    let _ = on_log.send(format!(
        "[LS-Trainer] Preparing ONNX export for results in '{}'...",
        results_path.display()
    ));

    if let Some(rfdetr_ckpt) = find_rfdetr_checkpoint_in_dir(&results_path) {
        return export_rfdetr_to_onnx(
            &project_root,
            &results_path,
            &rfdetr_ckpt,
            &request,
            &on_log,
        )
        .await;
    }

    export_yolo_to_onnx(project_root, results_path, request, on_log).await
}

pub(crate) fn embed_onnx_model_type_metadata(
    project_root: &Path,
    onnx_path: &Path,
    model_type: &str,
    model_family: &str,
    on_log: &Channel<String>,
) {
    if !onnx_path.is_file() {
        return;
    }
    let Some(python_exe) = get_venv_python(project_root) else {
        return;
    };

    let py_snippet = include_str!("../python/onnx_metadata.py");

    let mut cmd = Command::new(&python_exe);
    cmd.args([
        "-c",
        py_snippet,
        onnx_path.to_string_lossy().as_ref(),
        model_type,
        model_family,
    ]);
    hide_console_window(&mut cmd);
    if let Ok(status) = cmd.status() {
        if status.success() {
            let _ = on_log.send(format!(
                "[LS-Trainer] Embedded ONNX metadata: ls_model_type=\"{model_type}\" (ls_model_family=\"{model_family}\")"
            ));
        }
    }
}
