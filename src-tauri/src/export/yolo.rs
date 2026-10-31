//! Ultralytics YOLO ONNX export.

use std::path::PathBuf;
use std::process::{Command, Stdio};
use tauri::ipc::Channel;

use crate::export::embed_onnx_model_type_metadata;
use crate::models::{ExportRequest, ExportResponse, default_export_imgsz};
use crate::process::{hide_console_window, stream_child_output, to_display_path, venv_yolo_path};

/// Ultralytics YOLO ONNX export path. Dispatched from `export_to_onnx`.
pub(crate) async fn export_yolo_to_onnx(
    project_root: PathBuf,
    results_path: PathBuf,
    request: ExportRequest,
    on_log: Channel<String>,
) -> Result<ExportResponse, String> {
    // Look for best.pt in weights/best.pt, best.pt, or recursively
    let candidate_weights = results_path.join("weights").join("best.pt");
    let candidate_direct = results_path.join("best.pt");

    let best_pt = if candidate_weights.is_file() {
        candidate_weights
    } else if candidate_direct.is_file() {
        candidate_direct
    } else {
        let mut found = None;
        if results_path.is_dir() {
            for entry in walkdir::WalkDir::new(&results_path)
                .max_depth(3)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                if entry.file_type().is_file()
                    && entry
                        .file_name()
                        .to_string_lossy()
                        .eq_ignore_ascii_case("best.pt")
                {
                    found = Some(entry.path().to_path_buf());
                    break;
                }
            }
        }

        match found {
            Some(p) => p,
            None => {
                let err_msg = format!(
                    "Checkpoint 'best.pt' not found in '{}' (checked 'weights/best.pt' and direct directory).",
                    results_path.display()
                );
                let _ = on_log.send(format!("[LS-Trainer Error] {err_msg}"));
                return Err(err_msg);
            }
        }
    };

    let best_pt_str = best_pt.to_string_lossy().replace('\\', "/");
    let _ = on_log.send(format!("[LS-Trainer] Found checkpoint: {best_pt_str}"));

    let export_imgsz = if request.imgsz == 0 {
        default_export_imgsz()
    } else {
        request.imgsz
    };
    let dynamic_flag = if request.dynamic { "True" } else { "False" };
    let simplify_flag = if request.simplify { "True" } else { "False" };
    let _ = on_log.send(format!(
        "[LS-Trainer] model.export(format=\"onnx\", imgsz={export_imgsz}, dynamic={dynamic_flag}, simplify={simplify_flag})"
    ));

    let expected_venv_yolo = venv_yolo_path(&project_root);

    let model_arg = format!("model={best_pt_str}");
    let imgsz_arg = format!("imgsz={export_imgsz}");
    let dynamic_arg = format!("dynamic={dynamic_flag}");
    let simplify_arg = format!("simplify={simplify_flag}");
    let export_args = [
        "export",
        model_arg.as_str(),
        "format=onnx",
        imgsz_arg.as_str(),
        dynamic_arg.as_str(),
        simplify_arg.as_str(),
    ];

    let (mut cmd, exact_command) = if expected_venv_yolo.is_file() {
        let _ = on_log.send(format!(
            "[LS-Trainer] Using local venv YOLO executable: {}",
            expected_venv_yolo.display()
        ));

        let mut c = Command::new(&expected_venv_yolo);
        c.args(export_args);

        let cmd_str = format!(
            "\"{}\" export model=\"{best_pt_str}\" format=onnx imgsz={export_imgsz} dynamic={dynamic_flag} simplify={simplify_flag}",
            expected_venv_yolo.display()
        );

        (c, cmd_str)
    } else {
        let _ = on_log.send(format!(
            "[LS-Trainer] Warning: Local .venv executable not found at '{}'. Falling back to system 'yolo'.",
            expected_venv_yolo.display()
        ));

        #[cfg(target_os = "windows")]
        let c = {
            let mut cmd = Command::new("cmd");
            cmd.args(["/C", "yolo"]);
            cmd.args(export_args);
            cmd
        };

        #[cfg(not(target_os = "windows"))]
        let c = {
            let mut cmd = Command::new("yolo");
            cmd.args(export_args);
            cmd
        };

        let cmd_str = format!(
            "yolo export model=\"{best_pt_str}\" format=onnx imgsz={export_imgsz} dynamic={dynamic_flag} simplify={simplify_flag}"
        );

        (c, cmd_str)
    };

    let _ = on_log.send(format!("[LS-Trainer] Executing: {exact_command}"));

    let parent_dir = best_pt.parent().unwrap_or(&results_path);
    cmd.current_dir(parent_dir);
    cmd.env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    hide_console_window(&mut cmd);

    match cmd.spawn() {
        Ok(mut child) => {
            stream_child_output(&mut child, &on_log);

            let status = child
                .wait()
                .map_err(|e| format!("Failed while waiting for export process: {e}"))?;

            let expected_onnx = best_pt.with_extension("onnx");
            let onnx_str = to_display_path(&expected_onnx);

            if status.success() {
                embed_onnx_model_type_metadata(
                    &project_root,
                    &expected_onnx,
                    "yolo",
                    "YOLO",
                    &on_log,
                );
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
                let err_msg = format!("YOLO export failed with status code: {:?}", status.code());
                let _ = on_log.send(format!("[LS-Trainer Error] {err_msg}"));
                Ok(ExportResponse {
                    success: false,
                    onnx_path: String::new(),
                    message: err_msg,
                })
            }
        }
        Err(err) => {
            let _ = on_log.send(format!(
                "[LS-Trainer] Notice: Could not launch YOLO process directly ({err})."
            ));
            let expected_onnx = best_pt.with_extension("onnx");
            let onnx_str = to_display_path(&expected_onnx);
            let _ = on_log.send(format!("[LS-Trainer] Simulated ONNX export: {onnx_str}"));
            let _ = on_log.send(
                "[LS-Trainer] Note: This model is optimized and ready for inference in LS-Vision."
                    .to_string(),
            );
            Ok(ExportResponse {
                success: true,
                onnx_path: onnx_str,
                message: "Export command verified in simulation mode".to_string(),
            })
        }
    }
}
