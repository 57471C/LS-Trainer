//! Native file pickers and "open in file manager".

use std::path::PathBuf;
use std::process::Command;

use crate::process::{get_project_root, to_display_path};

#[tauri::command]
pub async fn pick_dataset_file() -> Option<String> {
    let file = rfd::AsyncFileDialog::new()
        .add_filter(
            "Dataset Archive / Manifest (*.tar, *.tar.gz, *.tgz, *.json, *.yaml, *.yml)",
            &["tar", "tar.gz", "tgz", "json", "yaml", "yml"],
        )
        .set_title("Select Dataset Archive or COCO/YOLO Manifest")
        .pick_file()
        .await;
    file.map(|f| to_display_path(f.path()))
}

#[tauri::command]
pub async fn pick_dataset_folder() -> Option<String> {
    let folder = rfd::AsyncFileDialog::new()
        .set_title("Select Dataset Folder (COCO or YOLO)")
        .pick_folder()
        .await;
    folder.map(|f| to_display_path(f.path()))
}

#[tauri::command]
pub async fn pick_model_file() -> Option<String> {
    let file = rfd::AsyncFileDialog::new()
        .add_filter("Model Checkpoint (*.pt, *.pth)", &["pt", "pth"])
        .set_title("Select Model Checkpoint")
        .pick_file()
        .await;
    file.map(|f| to_display_path(f.path()))
}

#[tauri::command]
pub async fn open_folder(path: String) -> Result<(), String> {
    let project_root = get_project_root();
    let clean_path = path.trim().trim_matches('"');
    let mut target = PathBuf::from(clean_path);
    if !target.is_absolute() {
        target = project_root.join(&target);
    }

    // Use the same checkpoint / candidate path logic as ONNX export:
    // Check if target is a direct file, or contains weights/best.pt, best.onnx, best.pt, or RF-DETR checkpoint_best_total.pth
    let target_file = if target.is_file() {
        Some(target.clone())
    } else {
        let candidates = [
            target.join("weights").join("best.onnx"),
            target.join("weights").join("best.pt"),
            target.join("best.onnx"),
            target.join("best.pt"),
            target.join("checkpoint_best_total.pth"),
            target.join("checkpoint_best_ema.pth"),
            target.join("checkpoint_best_regular.pth"),
        ];
        candidates.into_iter().find(|p| p.is_file())
    };

    let dir_to_open = if let Some(ref file) = target_file {
        file.parent().unwrap_or(&target).to_path_buf()
    } else if target.is_dir() {
        target.clone()
    } else if let Some(parent) = target.parent() {
        if parent.is_dir() {
            parent.to_path_buf()
        } else {
            project_root.clone()
        }
    } else {
        project_root.clone()
    };

    #[cfg(target_os = "windows")]
    {
        // On Windows, explorer.exe requires native backslashes (\). Forward slashes (/) are parsed
        // by explorer as command-line switches, causing it to fall back to 'My Documents'.
        if let Some(file) = target_file {
            let win_file = file.to_string_lossy().replace('/', "\\");
            let arg = format!("/select,{win_file}");
            Command::new("explorer")
                .arg(arg)
                .spawn()
                .map_err(|e| format!("Failed to open folder: {e}"))?;
        } else {
            let win_dir = dir_to_open.to_string_lossy().replace('/', "\\");
            let clean_dir = if win_dir.ends_with('\\') && win_dir.len() > 3 {
                win_dir.trim_end_matches('\\').to_string()
            } else {
                win_dir
            };
            Command::new("explorer")
                .arg(clean_dir)
                .spawn()
                .map_err(|e| format!("Failed to open folder: {e}"))?;
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Some(file) = target_file {
            Command::new("open")
                .args(["-R", &file.to_string_lossy()])
                .spawn()
                .map_err(|e| format!("Failed to open folder: {e}"))?;
        } else {
            Command::new("open")
                .arg(&dir_to_open)
                .spawn()
                .map_err(|e| format!("Failed to open folder: {e}"))?;
        }
    }

    #[cfg(target_os = "linux")]
    {
        Command::new("xdg-open")
            .arg(&dir_to_open)
            .spawn()
            .map_err(|e| format!("Failed to open folder: {e}"))?;
    }

    Ok(())
}
