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
//! - `settings`  – persistent data directory for runs / extraction
//!
//! Embedded Python runners live in `src/python/` and are compiled in with `include_str!`.

mod dataset;
mod dialogs;
mod export;
mod hardware;
mod models;
mod process;
mod rfdetr;
mod rtdetr;
mod settings;
mod training;

#[cfg(test)]
mod tests;

pub use dataset::labels::LabelCheckResult;
pub use hardware::{GpuInfo, compute_recommended_workers};
pub use models::{
    ExportRequest, ExportResponse, GpuTelemetry, TrainMetrics, TrainingRequest, TrainingResponse,
};
pub use process::{
    METRICS_PREFIX, ProcessControlCommand, ProcessControlHandle, TauriChannelSink,
    TrainingEventSink, to_display_path,
};
pub use training::{
    ACTIVE_TRAINING, active_training_control, force_kill_active_child, run_training,
    send_training_control, stop_training,
};

pub fn probe_local_gpu_info() -> GpuInfo {
    let root = process::get_project_root();
    hardware::detect_gpu_info(&root)
}

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
            training::stop_training,
            dialogs::open_folder,
            export::export_to_onnx,
            hardware::get_gpu_info,
            hardware::repair_cuda,
            settings::get_settings,
            settings::set_data_directory
        ])
        .run(tauri::generate_context!())
        .expect("error while building tauri application");
}
