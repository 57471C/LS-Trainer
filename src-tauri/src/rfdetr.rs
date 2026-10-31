//! Shared RF-DETR helpers: model class mapping, resolution snapping, dependency install, checkpoints.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use tauri::ipc::Channel;

use crate::process::{get_venv_python, hide_console_window, stream_child_output};

pub(crate) fn resolve_rfdetr_class_and_pretrain(
    model_choice: &str,
    task: &str,
) -> (&'static str, &'static str, u32) {
    let lower = model_choice.trim().to_ascii_lowercase();
    let is_seg = task.eq_ignore_ascii_case("segment") || lower.contains("-seg");

    if is_seg {
        if lower.contains("nano") || lower.ends_with("n-seg") {
            ("RFDETRSegNano", "rf-detr-seg-nano.pt", 12)
        } else if lower.contains("medium") || lower.ends_with("m-seg") {
            ("RFDETRSegMedium", "rf-detr-seg-medium.pt", 24)
        } else if lower.contains("large") || lower.ends_with("l-seg") {
            ("RFDETRSegLarge", "rf-detr-seg-large.pt", 24)
        } else if lower.contains("preview") {
            ("RFDETRSegPreview", "rf-detr-seg-preview.pt", 24)
        } else {
            ("RFDETRSegSmall", "rf-detr-seg-small.pt", 24)
        }
    } else if lower.contains("nano") || lower == "rf-detr-n" {
        ("RFDETRNano", "rf-detr-nano.pth", 32)
    } else if lower.contains("small") || lower == "rf-detr-s" {
        ("RFDETRSmall", "rf-detr-small.pth", 32)
    } else if lower.contains("large") || lower == "rf-detr-l" {
        ("RFDETRLarge", "rf-detr-large.pth", 32)
    } else if lower.contains("base") || lower == "rf-detr-b" {
        ("RFDETRBase", "rf-detr-base.pth", 56)
    } else {
        ("RFDETRMedium", "rf-detr-medium.pth", 32)
    }
}

pub(crate) fn snap_rfdetr_resolution(imgsz: u32, block_size: u32) -> u32 {
    let b = block_size.max(1);
    let rounded = ((imgsz + b / 2) / b) * b;
    rounded.max(b)
}

pub(crate) fn ensure_rfdetr_installed(project_root: &Path, on_log: &Channel<String>) {
    let Some(python_exe) = get_venv_python(project_root) else {
        return;
    };

    let mut check_cmd = Command::new(&python_exe);
    check_cmd.args([
        "-c",
        "import rfdetr, pytorch_lightning, pycocotools, vernier, hotcoco, faster_coco_eval, onnx; from rfdetr.training import build_trainer",
    ]);
    hide_console_window(&mut check_cmd);
    if let Ok(status) = check_cmd.status() {
        if status.success() {
            return;
        }
    }

    let _ = on_log.send(
        "[LS-Trainer] Installing rfdetr (Apache 2.0) and COCO training dependencies into .venv..."
            .to_string(),
    );
    let mut install_cmd = Command::new(&python_exe);
    install_cmd.args([
        "-m",
        "pip",
        "install",
        "rfdetr",
        "pytorch_lightning",
        "torchmetrics[detection]<1.9.0,>=1.8.2",
        "vernier",
        "faster-coco-eval>=1.6.5",
        "hotcoco>=1.3.0",
        "pycocotools>=2.0.8",
        "scipy",
        "peft",
        "onnx",
        "onnxruntime",
    ]);
    install_cmd
        .env("PYTHONUNBUFFERED", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    hide_console_window(&mut install_cmd);
    if let Ok(mut child) = install_cmd.spawn() {
        stream_child_output(&mut child, &on_log);
        let _ = child.wait();
    }
}

pub(crate) fn find_rfdetr_checkpoint_in_dir(results_path: &Path) -> Option<PathBuf> {
    if results_path.is_file()
        && results_path
            .extension()
            .map(|e| e.to_string_lossy().eq_ignore_ascii_case("pth"))
            .unwrap_or(false)
    {
        return Some(results_path.to_path_buf());
    }

    let direct_candidates = [
        results_path.join("checkpoint_best_total.pth"),
        results_path.join("checkpoint_best_ema.pth"),
        results_path.join("checkpoint_best_regular.pth"),
        results_path.join("checkpoint.pth"),
    ];
    for c in direct_candidates {
        if c.is_file() {
            return Some(c);
        }
    }
    None
}
