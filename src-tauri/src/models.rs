//! Serde request/response types shared by the Tauri commands and the Svelte UI.

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
pub struct TrainingRequest {
    pub dataset_path: String,
    pub model_choice: String,
    pub run_name: String,
    pub epochs: u32,
    pub imgsz: u32,
    pub device: String,
    #[serde(default = "default_batch")]
    pub batch: u32,
    #[serde(default = "default_amp")]
    pub amp: bool,
    #[serde(default = "default_freeze_backbone")]
    pub freeze_backbone: bool,
    #[serde(default = "default_freeze_layers")]
    pub freeze_layers: u32,
    #[serde(default = "default_mosaic")]
    pub mosaic: f32,
    #[serde(default = "default_mixup")]
    pub mixup: f32,
    #[serde(default = "default_copy_paste")]
    pub copy_paste: f32,
    #[serde(default = "default_degrees")]
    pub degrees: f32,
    #[serde(default = "default_translate")]
    pub translate: f32,
    #[serde(default = "default_scale")]
    pub scale: f32,
    #[serde(default = "default_flipud")]
    pub flipud: f32,
    #[serde(default = "default_close_mosaic")]
    pub close_mosaic: u32,
    /// Ultralytics task: "detect" or "segment". A `-seg` weight forces "segment".
    #[serde(default = "default_task")]
    pub task: String,
    #[serde(default = "default_model_family")]
    pub model_family: String,
    /// Optional manual override for DataLoader `num_workers`. `None` = auto-detect from physical CPU cores.
    #[serde(default)]
    pub workers: Option<u32>,
}

pub(crate) fn default_model_family() -> String {
    "YOLO".to_string()
}

pub(crate) fn default_batch() -> u32 {
    4
}

pub(crate) fn default_amp() -> bool {
    true
}

pub(crate) fn default_freeze_backbone() -> bool {
    false
}

pub(crate) fn default_freeze_layers() -> u32 {
    10
}

pub(crate) fn default_mosaic() -> f32 {
    0.0
}

pub(crate) fn default_mixup() -> f32 {
    0.0
}

pub(crate) fn default_copy_paste() -> f32 {
    0.0
}

pub(crate) fn default_degrees() -> f32 {
    180.0
}

pub(crate) fn default_translate() -> f32 {
    0.1
}

pub(crate) fn default_scale() -> f32 {
    0.05
}

pub(crate) fn default_flipud() -> f32 {
    0.5
}

pub(crate) fn default_close_mosaic() -> u32 {
    10
}

pub(crate) fn default_task() -> String {
    "detect".to_string()
}

pub(crate) fn default_export_imgsz() -> u32 {
    640
}

pub(crate) fn default_export_dynamic() -> bool {
    false
}

pub(crate) fn default_export_simplify() -> bool {
    true
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
pub struct TrainingResponse {
    pub success: bool,
    pub results_dir: String,
    pub message: String,
    #[serde(default)]
    pub map50: Option<f64>,
    #[serde(default)]
    pub map50_95: Option<f64>,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
pub struct ExportRequest {
    pub results_dir: String,
    #[serde(default = "default_export_imgsz")]
    pub imgsz: u32,
    #[serde(default = "default_export_dynamic")]
    pub dynamic: bool,
    #[serde(default = "default_export_simplify")]
    pub simplify: bool,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
pub struct ExportResponse {
    pub success: bool,
    pub onnx_path: String,
    pub message: String,
}
