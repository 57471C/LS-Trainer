//! LS-Trainer Tauri backend.
//!
//! Module layout:
//! - `models`    – request/response types for the Tauri commands
//! - `process`   – project root, venv paths, child-process helpers, log streaming
//! - `hardware`  – GPU/CPU detection, PyTorch wheel selection, GPU repair
//! - `dataset`   – YOLO `data.yaml` handling, COCO prep for RF-DETR, label checks
//! - `rfdetr`    – shared RF-DETR helpers
//! - `training`  – `start_training` (YOLO and RF-DETR backends)
//! - `export`    – `export_to_onnx` (YOLO and RF-DETR backends)
//! - `dialogs`   – file pickers and open-folder
//!
//! Embedded Python runners live in `src/python/` and are compiled in with `include_str!`.

mod dataset;
mod dialogs;
mod export;
mod hardware;
mod models;
mod process;
mod rfdetr;
mod training;

#[cfg(test)]
mod tests;

pub use dataset::labels::LabelCheckResult;
pub use hardware::{GpuInfo, compute_recommended_workers};
pub use models::{ExportRequest, ExportResponse, TrainingRequest, TrainingResponse};
pub use process::to_display_path;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            dialogs::pick_dataset_file,
            dialogs::pick_dataset_folder,
            dialogs::pick_model_file,
            dataset::labels::inspect_dataset_labels,
            training::start_training,
            dialogs::open_folder,
            export::export_to_onnx,
            hardware::get_gpu_info,
            hardware::repair_cuda
        ])
        .run(tauri::generate_context!())
        .expect("error while building tauri application");
}
