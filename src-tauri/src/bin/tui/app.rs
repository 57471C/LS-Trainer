//! Application state, configuration, modes, and background telemetry for `ls-trainer-tui`.

use std::collections::VecDeque;
use std::env;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use app_lib as ls_trainer_lib;
use ls_trainer_lib::{
    ExportRequest, GpuTelemetry, TrainMetrics, TrainingEventSink, TrainingRequest,
    probe_local_gpu_info, run_export_to_onnx, run_training, to_display_path,
};
use ratatui::style::Color;
use sysinfo::System;

pub const MAX_LOG_LINES: usize = 500;
pub const MAX_HISTORY_POINTS: usize = 300;
pub const MAX_SPARKLINE_SAMPLES: usize = 60;

pub const BAR_TRACK_BG: Color = Color::Rgb(26, 32, 44);
pub const BAR_GPU_TEAL: Color = Color::Rgb(68, 168, 186);
pub const BAR_EPOCH_TEAL: Color = Color::Rgb(62, 154, 174);
pub const BAR_STEP_BLUE: Color = Color::Rgb(72, 128, 184);
pub const BAR_OVERALL_CYAN: Color = Color::Rgb(56, 142, 168);
pub const BAR_RAM_SLATE: Color = Color::Rgb(92, 148, 188);
pub const GAUGE_TEXT_COVERED: Color = Color::Rgb(16, 20, 26);
pub const GAUGE_TEXT_UNCOVERED: Color = Color::Rgb(228, 234, 242);
pub const BORDER_SOFT: Color = Color::Rgb(64, 76, 94);
pub const TABLE_BORDER_SOFT: Color = Color::Rgb(74, 86, 104);
pub const TITLE_ACCENT: Color = Color::Rgb(116, 202, 218);
pub const LABEL_MUTED: Color = Color::Rgb(138, 150, 168);
pub const NEW_BEST_GOLD: Color = Color::Rgb(255, 216, 102);
pub const NEW_BEST_BADGE_BG: Color = Color::Rgb(42, 92, 66);
pub const OVERLAY_BG: Color = Color::Rgb(18, 22, 30);

pub const LAUNCHER_FIELD_COUNT: usize = 7;
pub const DEVICE_PRESETS: [&str; 6] = ["auto", "xpu", "xpu:0", "xpu:0,1", "cuda", "cpu"];
pub const WORKER_PRESETS: [&str; 6] = ["auto", "0", "1", "2", "4", "8"];

/// High-level screen/overlay mode enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum AppMode {
    Splash,
    Launcher,
    Training,
    Help,
    Config,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiMode {
    Setup,
    Training,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlayMode {
    None,
    Help,
    Config,
    History,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthLevel {
    Healthy,
    Warning,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchChoice {
    Yolo,
    RfDetr,
    RtDetr,
}

impl ArchChoice {
    pub fn from_str(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "yolo" | "ultralytics" => Some(Self::Yolo),
            "rf-detr" | "rfdetr" => Some(Self::RfDetr),
            "rt-detr" | "rtdetr" => Some(Self::RtDetr),
            _ => None,
        }
    }

    pub fn as_family_str(self) -> &'static str {
        match self {
            Self::Yolo => "YOLO",
            Self::RfDetr => "RF-DETR",
            Self::RtDetr => "RT-DETR",
        }
    }

    pub fn as_cli_str(self) -> &'static str {
        match self {
            Self::Yolo => "yolo",
            Self::RfDetr => "rf-detr",
            Self::RtDetr => "rt-detr",
        }
    }

    pub fn default_model(self) -> &'static str {
        match self {
            Self::Yolo => "yolo11s.pt",
            Self::RfDetr => "rf-detr-medium",
            Self::RtDetr => "rtdetr-v2-r18",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::Yolo => Self::RfDetr,
            Self::RfDetr => Self::RtDetr,
            Self::RtDetr => Self::Yolo,
        }
    }

    pub fn prev(self) -> Self {
        match self {
            Self::Yolo => Self::RtDetr,
            Self::RfDetr => Self::Yolo,
            Self::RtDetr => Self::RfDetr,
        }
    }
}

fn discover_recent_dataset_path() -> String {
    let candidates = [
        "runs/detect/BB-Parts-v3/ls_trainer_config.json",
        "runs/detect/test/ls_trainer_config.json",
        "runs/detect/gui-test/ls_trainer_config.json",
    ];
    for rel in candidates {
        if let Ok(raw) = std::fs::read_to_string(rel) {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&raw) {
                if let Some(ds) = val.get("dataset_path").and_then(|v| v.as_str()) {
                    let trimmed = ds.trim();
                    if !trimmed.is_empty() && Path::new(trimmed).exists() {
                        return trimmed.to_string();
                    }
                }
            }
        }
    }
    String::new()
}

#[derive(Debug, Clone)]
pub struct CliConfig {
    pub interactive: bool,
    pub arch: ArchChoice,
    pub dataset: String,
    pub model: String,
    pub epochs: String,
    pub batch: String,
    pub device: String,
    pub imgsz: u32,
    pub workers: Option<u32>,
    pub workers_input: String,
    pub run_name: String,
    pub focused_field: usize,
    pub editing_field: bool,
    pub error_banner: Option<String>,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            interactive: true,
            arch: ArchChoice::Yolo,
            dataset: discover_recent_dataset_path(),
            model: ArchChoice::Yolo.default_model().to_string(),
            epochs: "60".to_string(),
            batch: "4".to_string(),
            device: "auto".to_string(),
            imgsz: 960,
            workers: None,
            workers_input: "auto".to_string(),
            run_name: "BB-Parts-v3".to_string(),
            focused_field: 0,
            editing_field: false,
            error_banner: None,
        }
    }
}

impl CliConfig {
    pub fn parse_workers(&self) -> Result<Option<u32>, String> {
        let raw = self.workers_input.trim();
        if raw.is_empty() || raw.eq_ignore_ascii_case("auto") {
            return Ok(self.workers);
        }
        let parsed = raw
            .parse::<u32>()
            .map_err(|_| format!("Invalid workers value: '{raw}' (use a number or 'auto')"))?;
        Ok(Some(parsed))
    }

    pub fn to_training_request(&self) -> Result<TrainingRequest, String> {
        let dataset_trimmed = self.dataset.trim();
        if dataset_trimmed.is_empty() {
            return Err("Dataset path is required (--dataset <path>).".to_string());
        }
        let epochs_val = self
            .epochs
            .trim()
            .parse::<u32>()
            .map_err(|_| format!("Invalid epochs value: '{}'", self.epochs))?
            .max(1);
        let batch_val = self
            .batch
            .trim()
            .parse::<u32>()
            .map_err(|_| format!("Invalid batch value: '{}'", self.batch))?
            .max(1);
        let workers_val = self.parse_workers()?;
        let model_trimmed = if self.model.trim().is_empty() {
            self.arch.default_model().to_string()
        } else {
            self.model.trim().to_string()
        };
        let task = if model_trimmed.to_ascii_lowercase().contains("-seg") {
            "segment".to_string()
        } else {
            "detect".to_string()
        };

        Ok(TrainingRequest {
            dataset_path: dataset_trimmed.to_string(),
            model_choice: model_trimmed,
            run_name: if self.run_name.trim().is_empty() {
                "BB-Parts-v3".to_string()
            } else {
                self.run_name.trim().to_string()
            },
            epochs: epochs_val,
            imgsz: self.imgsz,
            device: if self.device.trim().is_empty() {
                "auto".to_string()
            } else {
                self.device.trim().to_string()
            },
            batch: batch_val,
            amp: true,
            freeze_backbone: false,
            freeze_layers: 10,
            mosaic: 0.0,
            mixup: 0.0,
            copy_paste: 0.0,
            degrees: 180.0,
            translate: 0.1,
            scale: 0.05,
            flipud: 0.5,
            close_mosaic: 10,
            task,
            model_family: self.arch.as_family_str().to_string(),
            workers: workers_val,
        })
    }
}

#[derive(Debug, Clone)]
pub struct GpuCardState {
    pub index: u32,
    pub name: String,
    pub vram_used_gb: f64,
    pub vram_total_gb: f64,
    pub util_percent: f64,
    pub power_watts: Option<f64>,
    pub temp_celsius: Option<f64>,
    pub util_history: VecDeque<u64>,
}

impl GpuCardState {
    pub fn new(index: u32, name: String) -> Self {
        let mut util_history = VecDeque::with_capacity(MAX_SPARKLINE_SAMPLES);
        util_history.push_back(0);
        Self {
            index,
            name,
            vram_used_gb: 0.0,
            vram_total_gb: 0.0,
            util_percent: 0.0,
            power_watts: None,
            temp_celsius: None,
            util_history,
        }
    }

    pub fn push_util(&mut self, util: f64) {
        self.util_percent = util.clamp(0.0, 100.0);
        if self.util_history.len() >= MAX_SPARKLINE_SAMPLES {
            self.util_history.pop_front();
        }
        self.util_history
            .push_back(self.util_percent.round() as u64);
    }
}

#[derive(Debug, Clone)]
pub struct RunHistoryEntry {
    pub run_name: String,
    pub task: String,
    pub run_dir: String,
    pub arch_model: String,
    pub map50: Option<f64>,
    pub map50_95: Option<f64>,
    pub best_weights: Option<String>,
    pub onnx_path: Option<String>,
    pub timestamp_label: String,
    pub modified_epoch_secs: u64,
}

pub struct AppState {
    pub mode: UiMode,
    pub overlay: OverlayMode,
    pub setup: CliConfig,
    pub active_request: Option<TrainingRequest>,
    pub is_running: bool,
    pub is_paused: bool,
    pub stop_initiated: bool,
    pub exit_code: Option<i32>,
    pub status_line: String,
    pub health_level: HealthLevel,
    pub warning_count: usize,
    pub suppressed_warning_count: usize,
    pub warning_tags: Vec<String>,
    pub last_error_msg: Option<String>,
    pub latest_metrics: TrainMetrics,
    pub step_counter: u64,
    pub train_loss_history: VecDeque<(f64, f64)>,
    pub val_loss_history: VecDeque<(f64, f64)>,
    pub map50_95_history: VecDeque<(f64, f64)>,
    pub best_map50: Option<f64>,
    pub best_map50_95: Option<f64>,
    pub best_loss: Option<f64>,
    pub best_ckpt_mtime: Option<std::time::SystemTime>,
    pub new_best_until: Option<Instant>,
    pub dataset_train_images: Option<u32>,
    pub gpus: Vec<GpuCardState>,
    pub host_cpu_percent: f64,
    pub host_ram_used_gb: f64,
    pub host_ram_total_gb: f64,
    pub host_cpu_history: VecDeque<u64>,
    pub host_core_count: usize,
    pub host_load_avg: (f64, f64, f64),
    pub logs: VecDeque<String>,
    pub log_scroll_offset: usize,
    pub follow_logs: bool,
    pub spinner_tick: usize,
    pub training_started_at: Option<Instant>,
    pub training_ended_at: Option<Instant>,
    pub results_dir: Option<String>,
    pub best_weights_path: Option<String>,
    pub onnx_path: Option<String>,
    pub history_entries: Vec<RunHistoryEntry>,
    pub history_selected: usize,
}

pub fn strip_ansi_codes(s: &str) -> String {
    if !s.contains('\x1b') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' && chars.peek() == Some(&'[') {
            chars.next();
            while let Some(&nc) = chars.peek() {
                chars.next();
                if ('@'..='~').contains(&nc) {
                    break;
                }
            }
        } else {
            out.push(ch);
        }
    }
    out
}

/// Classify noisy non-fatal framework warnings (triton, sync_dist, torch.jit.script, etc.)
/// that should update the Health/Warning status line while being suppressed from flooding the live log.
pub fn classify_noisy_warning(clean_lower: &str) -> Option<&'static str> {
    if clean_lower.contains("triton not found")
        || clean_lower.contains("flop counting will not work")
    {
        Some("triton")
    } else if clean_lower.contains("sync_dist") {
        Some("sync_dist")
    } else if clean_lower.contains("torch.jit.script") {
        Some("jit.script")
    } else if clean_lower.contains("accumulategrad")
        || clean_lower.contains("grad strides do not match")
    {
        Some("grad_stride")
    } else if clean_lower.contains("leafspec") {
        Some("LeafSpec")
    } else if clean_lower.contains("augmentation backend") {
        Some("aug_backend")
    } else if clean_lower.contains("possibleuserwarning:")
        || clean_lower.contains("futurewarning:")
        || clean_lower.contains("deprecationwarning:")
    {
        Some("deprecation")
    } else {
        None
    }
}

impl AppState {
    pub fn new(setup: CliConfig) -> Self {
        let mode = if setup.interactive {
            UiMode::Setup
        } else {
            UiMode::Training
        };
        let mut host_cpu_history = VecDeque::with_capacity(MAX_SPARKLINE_SAMPLES);
        host_cpu_history.push_back(0);
        Self {
            mode,
            overlay: OverlayMode::None,
            setup,
            active_request: None,
            is_running: false,
            is_paused: false,
            stop_initiated: false,
            exit_code: None,
            status_line: "Ready".to_string(),
            health_level: HealthLevel::Healthy,
            warning_count: 0,
            suppressed_warning_count: 0,
            warning_tags: Vec::new(),
            last_error_msg: None,
            latest_metrics: TrainMetrics::default(),
            step_counter: 0,
            train_loss_history: VecDeque::with_capacity(MAX_HISTORY_POINTS),
            val_loss_history: VecDeque::with_capacity(MAX_HISTORY_POINTS),
            map50_95_history: VecDeque::with_capacity(MAX_HISTORY_POINTS),
            best_map50: None,
            best_map50_95: None,
            best_loss: None,
            best_ckpt_mtime: None,
            new_best_until: None,
            dataset_train_images: None,
            gpus: Vec::new(),
            host_cpu_percent: 0.0,
            host_ram_used_gb: 0.0,
            host_ram_total_gb: 0.0,
            host_cpu_history,
            host_core_count: 0,
            host_load_avg: (0.0, 0.0, 0.0),
            logs: VecDeque::with_capacity(MAX_LOG_LINES),
            log_scroll_offset: 0,
            follow_logs: true,
            spinner_tick: 0,
            training_started_at: None,
            training_ended_at: None,
            results_dir: None,
            best_weights_path: None,
            onnx_path: None,
            history_entries: Vec::new(),
            history_selected: 0,
        }
    }

    #[allow(dead_code)]
    pub fn active_app_mode(&self) -> AppMode {
        match self.overlay {
            OverlayMode::Help => AppMode::Help,
            OverlayMode::Config | OverlayMode::History => AppMode::Config,
            OverlayMode::None => match self.mode {
                UiMode::Setup => AppMode::Launcher,
                UiMode::Training => AppMode::Training,
            },
        }
    }

    pub fn trigger_new_best(&mut self) {
        self.new_best_until = Some(Instant::now() + Duration::from_secs(6));
    }

    pub fn is_new_best_active(&self) -> bool {
        self.new_best_until
            .map_or(false, |until| Instant::now() < until)
    }

    pub fn record_warning_tag(&mut self, tag: &str) {
        if self.health_level != HealthLevel::Error {
            self.health_level = HealthLevel::Warning;
        }
        self.warning_count = self.warning_count.saturating_add(1);
        let clean_tag = tag.trim().to_string();
        if !clean_tag.is_empty() && !self.warning_tags.iter().any(|t| t == &clean_tag) {
            if self.warning_tags.len() < 4 {
                self.warning_tags.push(clean_tag);
            }
        }
    }

    pub fn record_error(&mut self, msg: &str) {
        self.health_level = HealthLevel::Error;
        let trimmed = msg.trim();
        if !trimmed.is_empty() {
            self.last_error_msg = Some(trimmed.to_string());
        }
    }

    pub fn refresh_run_artifacts(&mut self) {
        if self.results_dir.is_none() {
            if let Some(ref req) = self.active_request {
                let fallback_dir = PathBuf::from("runs").join(&req.task).join(&req.run_name);
                if fallback_dir.exists() {
                    self.results_dir = Some(to_display_path(&fallback_dir));
                }
            }
        }
        if let Some(ref dir_str) = self.results_dir {
            let dir_path = PathBuf::from(dir_str);
            if let Some(best_p) = find_best_checkpoint_in_dir(&dir_path) {
                let disp = to_display_path(&best_p);
                let mtime = std::fs::metadata(&best_p)
                    .ok()
                    .and_then(|m| m.modified().ok());
                let is_best_named = best_p
                    .file_name()
                    .map(|n| n.to_string_lossy().to_ascii_lowercase().contains("best"))
                    .unwrap_or(false);
                if is_best_named {
                    let changed = self.best_weights_path.as_ref() != Some(&disp)
                        || (mtime.is_some() && mtime != self.best_ckpt_mtime);
                    if changed && self.step_counter > 0 {
                        self.trigger_new_best();
                    }
                    self.best_ckpt_mtime = mtime;
                }
                self.best_weights_path = Some(disp);
            }
            if let Some(onnx_p) = find_onnx_in_dir(&dir_path) {
                self.onnx_path = Some(to_display_path(&onnx_p));
            }
        }
    }

    pub fn elapsed_seconds(&self) -> f64 {
        match (self.training_started_at, self.training_ended_at) {
            (Some(start), Some(end)) => end.saturating_duration_since(start).as_secs_f64(),
            (Some(start), None) => start.elapsed().as_secs_f64(),
            _ => 0.0,
        }
    }

    pub fn push_log(&mut self, line: String) {
        let clean = strip_ansi_codes(&line);
        let clean_trimmed = clean.trim();
        let clean_lower = clean_trimmed.to_ascii_lowercase();

        // Parse live paths & dataset info from log stream immediately
        if let Some(rest) = clean_trimmed.strip_prefix("[LS-Trainer] Results directory:") {
            let d = rest.trim();
            if !d.is_empty() {
                self.results_dir = Some(to_display_path(Path::new(d)));
                self.refresh_run_artifacts();
            }
        } else if let Some(idx) = clean_trimmed.find("Logging results to ") {
            let d = clean_trimmed[idx + "Logging results to ".len()..].trim();
            if !d.is_empty() && Path::new(d).is_dir() {
                self.results_dir = Some(to_display_path(Path::new(d)));
                self.refresh_run_artifacts();
            }
        } else if let Some(idx) = clean_trimmed.find("Results saved to: ") {
            let rest = clean_trimmed[idx + "Results saved to: ".len()..].trim();
            let d = rest.split(" (run ").next().unwrap_or(rest).trim();
            if !d.is_empty() && Path::new(d).is_dir() {
                self.results_dir = Some(to_display_path(Path::new(d)));
                self.refresh_run_artifacts();
            }
        }

        if let Some(idx) = clean_trimmed.find("Immediate checkpoint saved to:") {
            let p = clean_trimmed[idx + "Immediate checkpoint saved to:".len()..].trim();
            if !p.is_empty() {
                self.best_weights_path = Some(to_display_path(Path::new(p)));
                self.refresh_run_artifacts();
            }
        } else if let Some(idx) = clean_trimmed.find("Found RF-DETR checkpoint:") {
            let p = clean_trimmed[idx + "Found RF-DETR checkpoint:".len()..].trim();
            if !p.is_empty() {
                self.best_weights_path = Some(to_display_path(Path::new(p)));
            }
        } else if let Some(idx) = clean_trimmed.find("ONNX export completed successfully! File:") {
            let p =
                clean_trimmed[idx + "ONNX export completed successfully! File:".len()..].trim();
            if !p.is_empty() {
                self.onnx_path = Some(to_display_path(Path::new(p)));
            }
        }

        // Parse train image count if logged by COCO / YOLO prep
        if let Some(idx) = clean_trimmed.find("Prepared RF-DETR COCO splits:") {
            let tail = &clean_trimmed[idx + "Prepared RF-DETR COCO splits:".len()..];
            if let Some(num_str) = tail.split_whitespace().next() {
                if let Ok(n) = num_str.parse::<u32>() {
                    if n > 0 {
                        self.dataset_train_images = Some(n);
                    }
                }
            }
        }

        // 1. Suppress noisy framework warnings (triton, sync_dist, etc.) from flooding the live log
        if let Some(noisy_tag) = classify_noisy_warning(&clean_lower) {
            self.suppressed_warning_count = self.suppressed_warning_count.saturating_add(1);
            self.record_warning_tag(noisy_tag);
            return;
        }

        // 2. Track LS-Trainer / runtime warnings and errors in the status line
        if clean_trimmed.starts_with("[LS-Trainer Error]")
            || clean_trimmed.starts_with("[LS-Trainer] Error:")
            || clean_lower.starts_with("runtimeerror:")
            || clean_lower.contains("outofmemoryerror")
        {
            let msg = clean_trimmed
                .strip_prefix("[LS-Trainer Error]")
                .or_else(|| clean_trimmed.strip_prefix("[LS-Trainer] Error:"))
                .unwrap_or(clean_trimmed);
            self.record_error(msg);
        } else if clean_trimmed.starts_with("[LS-Trainer Warning]")
            || clean_trimmed.starts_with("[LS-Trainer] Warning:")
        {
            let msg = clean_trimmed
                .strip_prefix("[LS-Trainer Warning]")
                .or_else(|| clean_trimmed.strip_prefix("[LS-Trainer] Warning:"))
                .unwrap_or(clean_trimmed)
                .trim();
            let short_tag = if msg.to_ascii_lowercase().contains("xpu index") {
                "xpu_device"
            } else if msg.to_ascii_lowercase().contains("onnx") {
                "onnx_export"
            } else if msg.to_ascii_lowercase().contains("polygon")
                || msg.to_ascii_lowercase().contains("segmentation")
            {
                "seg_labels"
            } else {
                "trainer_warn"
            };
            self.record_warning_tag(short_tag);
            // Collapse consecutive identical warnings in the log
            if self.logs.back().map(|prev| prev == &line).unwrap_or(false) {
                self.suppressed_warning_count = self.suppressed_warning_count.saturating_add(1);
                return;
            }
        }

        if self.logs.len() >= MAX_LOG_LINES {
            self.logs.pop_front();
        }
        self.logs.push_back(line);
        if self.follow_logs {
            self.log_scroll_offset = 0;
        }
    }

    pub fn apply_metrics(&mut self, mut metrics: TrainMetrics) {
        if metrics.total_epochs > 0 && metrics.epoch > metrics.total_epochs {
            metrics.epoch = metrics.total_epochs;
        }
        if metrics.total_steps > 0 && metrics.step > metrics.total_steps {
            metrics.step = metrics.total_steps;
        }
        self.step_counter += 1;
        let x = self.step_counter as f64;
        let mut improved = false;

        if metrics.train_loss.is_finite() && metrics.train_loss > 0.0 {
            if self.train_loss_history.len() >= MAX_HISTORY_POINTS {
                self.train_loss_history.pop_front();
            }
            self.train_loss_history.push_back((x, metrics.train_loss));
            if let Some(prev_loss) = self.best_loss {
                if metrics.train_loss < prev_loss - 1e-5 {
                    self.best_loss = Some(metrics.train_loss);
                    if metrics.total_steps > 0 && metrics.step == metrics.total_steps {
                        improved = true;
                    }
                }
            } else {
                self.best_loss = Some(metrics.train_loss);
            }
        } else if metrics.train_loss.is_finite() && metrics.train_loss >= 0.0 {
            if self.train_loss_history.len() >= MAX_HISTORY_POINTS {
                self.train_loss_history.pop_front();
            }
            self.train_loss_history.push_back((x, metrics.train_loss));
        }

        if let Some(vl) = metrics.val_loss {
            if vl.is_finite() && vl >= 0.0 {
                let prev_min_vl = self
                    .val_loss_history
                    .iter()
                    .map(|&(_, y)| y)
                    .fold(None::<f64>, |acc, y| Some(acc.map_or(y, |m| m.min(y))));
                if self.val_loss_history.len() >= MAX_HISTORY_POINTS {
                    self.val_loss_history.pop_front();
                }
                self.val_loss_history.push_back((x, vl));
                if prev_min_vl.map_or(true, |prev| vl < prev - 1e-5) {
                    improved = true;
                }
            }
        }

        if let Some(m95) = metrics.map50_95 {
            if m95.is_finite() && m95 >= 0.0 {
                if self.map50_95_history.len() >= MAX_HISTORY_POINTS {
                    self.map50_95_history.pop_front();
                }
                self.map50_95_history.push_back((x, m95));
                let is_better = self
                    .best_map50_95
                    .map_or(m95 > 0.0, |prev| m95 > prev + 1e-6);
                if is_better {
                    improved = true;
                }
                self.best_map50_95 = Some(self.best_map50_95.map_or(m95, |prev| prev.max(m95)));
            }
        }

        if let Some(m50) = metrics.map50 {
            if m50.is_finite() && m50 >= 0.0 {
                let is_better = self.best_map50.map_or(m50 > 0.0, |prev| m50 > prev + 1e-6);
                if is_better {
                    improved = true;
                }
                self.best_map50 = Some(self.best_map50.map_or(m50, |prev| prev.max(m50)));
            }
        }

        if improved {
            self.trigger_new_best();
        }

        self.merge_metric_gpus(&metrics.gpus);
        self.latest_metrics = metrics;
        self.refresh_run_artifacts();
    }

    pub fn merge_metric_gpus(&mut self, incoming: &[GpuTelemetry]) {
        for g in incoming {
            if let Some(card) = self.gpus.iter_mut().find(|c| c.index == g.index) {
                if !g.name.is_empty() {
                    card.name = g.name.clone();
                }
                card.vram_used_gb = g.vram_used_gb;
                if g.vram_total_gb > 0.0 {
                    card.vram_total_gb = g.vram_total_gb;
                }
                card.push_util(g.util_percent);
            } else {
                let mut card = GpuCardState::new(g.index, g.name.clone());
                card.vram_used_gb = g.vram_used_gb;
                card.vram_total_gb = g.vram_total_gb;
                card.push_util(g.util_percent);
                self.gpus.push(card);
            }
        }
        self.gpus.sort_by_key(|c| c.index);
    }
}

/// `TrainingEventSink` implementation for `ls-trainer-tui`.
#[derive(Clone)]
pub struct TuiEventSink {
    state: Arc<Mutex<AppState>>,
}

impl TuiEventSink {
    pub fn new(state: Arc<Mutex<AppState>>) -> Self {
        Self { state }
    }
}

impl TrainingEventSink for TuiEventSink {
    fn on_log(&self, line: String) {
        if let Ok(mut guard) = self.state.lock() {
            if let Some(status) = line.strip_prefix("[LS-Trainer Status] ") {
                guard.status_line = status.trim().to_string();
            }
            guard.push_log(line);
        }
    }

    fn on_metrics(&self, metrics: TrainMetrics) {
        if let Ok(mut guard) = self.state.lock() {
            guard.apply_metrics(metrics);
        }
    }

    fn on_exit(&self, code: Option<i32>) {
        if let Ok(mut guard) = self.state.lock() {
            guard.refresh_run_artifacts();
            if code == Some(0) && !guard.stop_initiated {
                // Keep `is_running = true` while the post-training ONNX export executes.
                guard.status_line = "Training complete — exporting ONNX...".to_string();
            } else {
                guard.exit_code = code;
                guard.is_running = false;
                guard.is_paused = false;
                guard.training_ended_at = Some(Instant::now());
                if let Some(c) = code {
                    if c != 0 && !guard.stop_initiated {
                        guard.record_error(&format!("Process exited with code {c}"));
                    }
                }
                guard.status_line = match code {
                    Some(0) => "Completed".to_string(),
                    Some(c) if guard.stop_initiated => format!("Stopped (exit {c})"),
                    Some(c) => format!("Exited (code {c})"),
                    None if guard.stop_initiated => "Stopped".to_string(),
                    None => "Finished".to_string(),
                };
            }
        }
    }

    fn preserve_ansi(&self) -> bool {
        true
    }
}

#[cfg(target_os = "linux")]
mod linux_hwmon {
    use super::{AppState, GpuCardState};
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    #[derive(Debug, Clone)]
    struct HwmonSensorReading {
        index: u32,
        name: String,
        temp_celsius: Option<f64>,
        power_watts: Option<f64>,
    }

    fn read_trimmed_u64(path: &Path) -> Option<u64> {
        std::fs::read_to_string(path)
            .ok()?
            .trim()
            .parse::<u64>()
            .ok()
    }

    fn classify_pci_gpu_vendor(pci_dev_dir: &Path) -> Option<&'static str> {
        let class_raw = std::fs::read_to_string(pci_dev_dir.join("class")).ok()?;
        let class_trimmed = class_raw.trim().to_ascii_lowercase();
        if !class_trimmed.starts_with("0x03") {
            return None;
        }
        let vendor_raw = std::fs::read_to_string(pci_dev_dir.join("vendor")).ok()?;
        match vendor_raw.trim().to_ascii_lowercase().as_str() {
            "0x8086" => Some("Intel Arc / XPU"),
            "0x10de" => Some("NVIDIA GPU"),
            "0x1002" => Some("AMD GPU"),
            _ => Some("Discrete GPU"),
        }
    }

    fn scan_pci_hwmon_sensors(
        prev_energy: &mut HashMap<PathBuf, (u64, Instant)>,
    ) -> Vec<HwmonSensorReading> {
        let mut results = Vec::new();
        let Ok(pci_entries) = std::fs::read_dir("/sys/bus/pci/devices") else {
            return results;
        };

        let mut pci_dirs: Vec<PathBuf> = pci_entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| classify_pci_gpu_vendor(p).is_some())
            .collect();
        pci_dirs.sort();

        for (gpu_idx, pci_dir) in pci_dirs.into_iter().enumerate() {
            let vendor_label = classify_pci_gpu_vendor(&pci_dir).unwrap_or("GPU");
            let hwmon_root = pci_dir.join("hwmon");
            let Ok(hwmon_entries) = std::fs::read_dir(&hwmon_root) else {
                continue;
            };

            let mut temp_c: Option<f64> = None;
            let mut power_w: Option<f64> = None;
            let mut sensor_label = format!("{vendor_label} Card {gpu_idx}");

            for hw_entry in hwmon_entries.flatten() {
                let hw_dir = hw_entry.path();
                if let Ok(name_str) = std::fs::read_to_string(hw_dir.join("name")) {
                    let trimmed = name_str.trim();
                    if !trimmed.is_empty() {
                        sensor_label = format!("{vendor_label} ({trimmed}) #{gpu_idx}");
                    }
                }

                for temp_file in ["temp1_input", "temp2_input"] {
                    if let Some(milli_c) = read_trimmed_u64(&hw_dir.join(temp_file)) {
                        if milli_c > 0 && milli_c < 150_000 {
                            temp_c = Some(milli_c as f64 / 1000.0);
                            break;
                        }
                    }
                }

                for power_file in ["power1_input", "power1_average"] {
                    if let Some(micro_w) = read_trimmed_u64(&hw_dir.join(power_file)) {
                        if micro_w > 0 {
                            power_w = Some(micro_w as f64 / 1_000_000.0);
                            break;
                        }
                    }
                }

                if power_w.is_none() {
                    let energy_path = hw_dir.join("energy1_input");
                    if let Some(micro_j) = read_trimmed_u64(&energy_path) {
                        let now = Instant::now();
                        if let Some((prev_j, prev_t)) = prev_energy.get(&energy_path) {
                            let dt = now.duration_since(*prev_t).as_secs_f64();
                            if dt >= 0.1 && micro_j >= *prev_j {
                                let delta_j = (micro_j - *prev_j) as f64 / 1_000_000.0;
                                power_w = Some(delta_j / dt);
                            }
                        }
                        prev_energy.insert(energy_path, (micro_j, now));
                    }
                }
            }

            results.push(HwmonSensorReading {
                index: gpu_idx as u32,
                name: sensor_label,
                temp_celsius: temp_c,
                power_watts: power_w,
            });
        }

        results
    }

    pub fn spawn_linux_hwmon_poller(state: Arc<Mutex<AppState>>) {
        std::thread::spawn(move || {
            let mut prev_energy: HashMap<PathBuf, (u64, Instant)> = HashMap::new();
            loop {
                let readings = scan_pci_hwmon_sensors(&mut prev_energy);
                if let Ok(mut guard) = state.lock() {
                    for r in readings {
                        if let Some(card) = guard.gpus.iter_mut().find(|c| c.index == r.index) {
                            card.temp_celsius = r.temp_celsius;
                            card.power_watts = r.power_watts;
                        } else {
                            let mut card = GpuCardState::new(r.index, r.name);
                            card.temp_celsius = r.temp_celsius;
                            card.power_watts = r.power_watts;
                            guard.gpus.push(card);
                        }
                    }
                    guard.gpus.sort_by_key(|c| c.index);
                }
                std::thread::sleep(Duration::from_millis(1000));
            }
        });
    }
}

pub fn spawn_hardware_polling_tasks(state: Arc<Mutex<AppState>>) {
    let state_init = Arc::clone(&state);
    std::thread::spawn(move || {
        let info = probe_local_gpu_info();
        if let Some(dev_name) = info.device_name {
            if let Ok(mut guard) = state_init.lock() {
                if guard.gpus.is_empty() {
                    guard.gpus.push(GpuCardState::new(0, dev_name));
                }
            }
        }
    });

    #[cfg(target_os = "linux")]
    linux_hwmon::spawn_linux_hwmon_poller(Arc::clone(&state));

    std::thread::spawn(move || {
        let mut sys = System::new_all();
        loop {
            sys.refresh_cpu_usage();
            sys.refresh_memory();
            let cpu_usage = sys.global_cpu_usage() as f64;
            let core_count = sys.cpus().len();
            let load_avg = System::load_average();
            let used_ram_gb = sys.used_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
            let total_ram_gb = sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);

            if let Ok(mut guard) = state.lock() {
                guard.host_cpu_percent = cpu_usage.clamp(0.0, 100.0);
                guard.host_ram_used_gb = used_ram_gb;
                guard.host_ram_total_gb = total_ram_gb;
                guard.host_core_count = core_count;
                guard.host_load_avg = (load_avg.one, load_avg.five, load_avg.fifteen);
                let cpu_sample = guard.host_cpu_percent.round() as u64;
                if guard.host_cpu_history.len() >= MAX_SPARKLINE_SAMPLES {
                    guard.host_cpu_history.pop_front();
                }
                guard.host_cpu_history.push_back(cpu_sample);
            }

            std::thread::sleep(Duration::from_millis(1000));
        }
    });
}

pub fn find_best_checkpoint_in_dir(run_dir: &Path) -> Option<PathBuf> {
    let candidates = [
        run_dir.join("weights").join("best.pt"),
        run_dir.join("best.pt"),
        run_dir.join("checkpoint_best_total.pth"),
        run_dir.join("checkpoint_best_ema.pth"),
        run_dir.join("checkpoint_best_regular.pth"),
        run_dir.join("weights").join("best.pth"),
        run_dir.join("best.pth"),
        run_dir.join("weights").join("last.pt"),
        run_dir.join("last.pt"),
        run_dir.join("weights").join("last.pth"),
        run_dir.join("last.pth"),
        run_dir.join("checkpoint.pth"),
    ];
    for c in candidates {
        if c.is_file() {
            return Some(c);
        }
    }
    None
}

pub fn find_onnx_in_dir(run_dir: &Path) -> Option<PathBuf> {
    let candidates = [
        run_dir.join("best.onnx"),
        run_dir.join("weights").join("best.onnx"),
    ];
    for c in candidates {
        if c.is_file() {
            return Some(c);
        }
    }
    None
}

pub fn count_dataset_train_images(dataset_path: &str) -> Option<u32> {
    let p = Path::new(dataset_path.trim());
    if !p.exists() {
        return None;
    }
    let root = if p.is_file() { p.parent()? } else { p };

    for coco_rel in [
        PathBuf::from("train").join("_annotations.coco.json"),
        PathBuf::from("annotations").join("instances_train.json"),
    ] {
        let coco_file = root.join(coco_rel);
        if coco_file.is_file() {
            if let Ok(raw) = std::fs::read_to_string(&coco_file) {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&raw) {
                    if let Some(arr) = val.get("images").and_then(|v| v.as_array()) {
                        if !arr.is_empty() {
                            return Some(arr.len() as u32);
                        }
                    }
                }
            }
        }
    }

    for img_rel in [
        PathBuf::from("images").join("train"),
        PathBuf::from("train").join("images"),
    ] {
        let img_dir = root.join(img_rel);
        if img_dir.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&img_dir) {
                let count = entries
                    .flatten()
                    .filter(|e| {
                        e.path()
                            .extension()
                            .map(|ext| {
                                matches!(
                                    ext.to_string_lossy().to_ascii_lowercase().as_str(),
                                    "jpg" | "jpeg" | "png" | "bmp" | "webp" | "tif" | "tiff"
                                )
                            })
                            .unwrap_or(false)
                    })
                    .count();
                if count > 0 {
                    return Some(count as u32);
                }
            }
        }
    }
    None
}

fn format_relative_age(modified: std::time::SystemTime) -> (u64, String) {
    let epoch_secs = modified
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let age_secs = std::time::SystemTime::now()
        .duration_since(modified)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let label = if age_secs < 60 {
        format!("{age_secs}s ago")
    } else if age_secs < 3600 {
        format!("{}m ago", age_secs / 60)
    } else if age_secs < 86_400 {
        format!("{}h ago", age_secs / 3600)
    } else {
        format!("{}d ago", age_secs / 86_400)
    };
    (epoch_secs, label)
}

fn parse_metrics_for_run_dir(run_dir: &Path) -> (Option<f64>, Option<f64>) {
    let json_path = run_dir.join("ls_trainer_metrics.json");
    if json_path.is_file() {
        if let Ok(raw) = std::fs::read_to_string(&json_path) {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&raw) {
                let m50 = val.get("map50").and_then(|v| v.as_f64());
                let m95 = val.get("map50_95").and_then(|v| v.as_f64());
                if m50.is_some() || m95.is_some() {
                    return (m50, m95);
                }
            }
        }
    }
    let csv_path = run_dir.join("results.csv");
    if csv_path.is_file() {
        if let Ok(raw) = std::fs::read_to_string(&csv_path) {
            let mut lines = raw.lines();
            if let Some(header) = lines.next() {
                let cols: Vec<&str> = header.split(',').map(|s| s.trim()).collect();
                let idx_50 = cols.iter().position(|c| c.contains("mAP50("));
                let idx_95 = cols.iter().position(|c| c.contains("mAP50-95("));
                let mut best_50: Option<f64> = None;
                let mut best_95: Option<f64> = None;
                for row in lines {
                    let vals: Vec<&str> = row.split(',').map(|s| s.trim()).collect();
                    if let Some(i50) = idx_50 {
                        if let Some(v) = vals.get(i50).and_then(|s| s.parse::<f64>().ok()) {
                            if v.is_finite() {
                                best_50 = Some(best_50.map_or(v, |p| p.max(v)));
                            }
                        }
                    }
                    if let Some(i95) = idx_95 {
                        if let Some(v) = vals.get(i95).and_then(|s| s.parse::<f64>().ok()) {
                            if v.is_finite() {
                                best_95 = Some(best_95.map_or(v, |p| p.max(v)));
                            }
                        }
                    }
                }
                return (best_50, best_95);
            }
        }
    }
    (None, None)
}

pub fn scan_run_history() -> Vec<RunHistoryEntry> {
    let mut entries = Vec::new();
    for task in ["detect", "segment"] {
        let task_dir = PathBuf::from("runs").join(task);
        let Ok(read_dir) = std::fs::read_dir(&task_dir) else {
            continue;
        };
        for item in read_dir.flatten() {
            let path = item.path();
            if !path.is_dir() {
                continue;
            }
            let run_name = item.file_name().to_string_lossy().to_string();
            let best_weights = find_best_checkpoint_in_dir(&path).map(|p| to_display_path(&p));
            let onnx_path = find_onnx_in_dir(&path).map(|p| to_display_path(&p));
            let (map50, map50_95) = parse_metrics_for_run_dir(&path);

            let mut arch_model = String::new();
            let cfg_file = path.join("ls_trainer_config.json");
            if cfg_file.is_file() {
                if let Ok(raw) = std::fs::read_to_string(&cfg_file) {
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&raw) {
                        let fam = val
                            .get("model_family")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let mod_c = val
                            .get("model_choice")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        if !fam.is_empty() && !mod_c.is_empty() {
                            arch_model = format!("{fam} ({mod_c})");
                        } else if !mod_c.is_empty() {
                            arch_model = mod_c.to_string();
                        }
                    }
                }
            }
            if arch_model.is_empty() {
                if path.join("ls_trainer_rfdetr_args.json").is_file() {
                    arch_model = "RF-DETR".to_string();
                } else if path.join("args.yaml").is_file() {
                    arch_model = "YOLO".to_string();
                } else {
                    arch_model = task.to_string();
                }
            }

            let mtime = best_weights
                .as_ref()
                .and_then(|w| std::fs::metadata(w).ok())
                .or_else(|| std::fs::metadata(&cfg_file).ok())
                .or_else(|| std::fs::metadata(&path).ok())
                .and_then(|m| m.modified().ok())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            let (modified_epoch_secs, timestamp_label) = format_relative_age(mtime);

            entries.push(RunHistoryEntry {
                run_name,
                task: task.to_string(),
                run_dir: to_display_path(&path),
                arch_model,
                map50,
                map50_95,
                best_weights,
                onnx_path,
                timestamp_label,
                modified_epoch_secs,
            });
        }
    }
    entries.sort_by(|a, b| b.modified_epoch_secs.cmp(&a.modified_epoch_secs));
    entries.truncate(20);
    entries
}

pub fn launch_training_thread(state: Arc<Mutex<AppState>>, request: TrainingRequest) {
    let initial_dir = PathBuf::from("runs")
        .join(&request.task)
        .join(&request.run_name);
    let initial_train_imgs = count_dataset_train_images(&request.dataset_path);

    if let Ok(mut guard) = state.lock() {
        guard.mode = UiMode::Training;
        guard.overlay = OverlayMode::None;
        guard.active_request = Some(request.clone());
        guard.is_running = true;
        guard.is_paused = false;
        guard.stop_initiated = false;
        guard.exit_code = None;
        guard.health_level = HealthLevel::Healthy;
        guard.warning_count = 0;
        guard.suppressed_warning_count = 0;
        guard.warning_tags.clear();
        guard.last_error_msg = None;
        guard.results_dir = Some(to_display_path(&initial_dir));
        guard.best_weights_path = None;
        guard.onnx_path = None;
        guard.best_loss = None;
        guard.best_ckpt_mtime = None;
        guard.new_best_until = None;
        guard.dataset_train_images = initial_train_imgs;
        guard.training_started_at = Some(Instant::now());
        guard.training_ended_at = None;
        guard.status_line = "Initializing training...".to_string();
        guard.latest_metrics = TrainMetrics {
            epoch: 0,
            total_epochs: request.epochs,
            ..Default::default()
        };
        guard.refresh_run_artifacts();
    }

    let sink: Arc<dyn TrainingEventSink> = Arc::new(TuiEventSink::new(Arc::clone(&state)));
    tauri::async_runtime::spawn(async move {
        let result = run_training(sink.clone(), request.clone()).await;
        match result {
            Ok(train_resp) => {
                let actual_dir_path = PathBuf::from(&train_resp.results_dir);
                let best_ckpt =
                    find_best_checkpoint_in_dir(&actual_dir_path).map(|p| to_display_path(&p));
                if let Ok(mut guard) = state.lock() {
                    guard.results_dir = Some(train_resp.results_dir.clone());
                    if let Some(ref ckpt) = best_ckpt {
                        if guard.best_weights_path.as_ref() != Some(ckpt) {
                            guard.trigger_new_best();
                        }
                        guard.best_weights_path = Some(ckpt.clone());
                    }
                    if let Some(m50) = train_resp.map50 {
                        if guard.best_map50.map_or(m50 > 0.0, |prev| m50 > prev + 1e-6) {
                            guard.trigger_new_best();
                        }
                        guard.best_map50 =
                            Some(guard.best_map50.map_or(m50, |prev| prev.max(m50)));
                    }
                    if let Some(m95) = train_resp.map50_95 {
                        if guard
                            .best_map50_95
                            .map_or(m95 > 0.0, |prev| m95 > prev + 1e-6)
                        {
                            guard.trigger_new_best();
                        }
                        guard.best_map50_95 =
                            Some(guard.best_map50_95.map_or(m95, |prev| prev.max(m95)));
                    }
                }

                let stop_was_initiated = state.lock().map(|g| g.stop_initiated).unwrap_or(false);
                if train_resp.success
                    && train_resp.message.to_ascii_lowercase().contains("complete")
                    && !stop_was_initiated
                {
                    sink.on_log("[LS-Trainer Status] Exporting best model to ONNX...".to_string());
                    let export_req = ExportRequest {
                        results_dir: train_resp.results_dir.clone(),
                        imgsz: request.imgsz,
                        dynamic: false,
                        simplify: true,
                    };
                    match run_export_to_onnx(sink.clone(), export_req).await {
                        Ok(exp_resp) if exp_resp.success && !exp_resp.onnx_path.is_empty() => {
                            if let Ok(mut guard) = state.lock() {
                                guard.onnx_path = Some(exp_resp.onnx_path.clone());
                            }
                            let ckpt_disp = best_ckpt.as_deref().unwrap_or(&train_resp.results_dir);
                            sink.on_log(format!(
                                "[LS-Trainer] Training complete! Best weights: {} | ONNX: {}",
                                ckpt_disp, exp_resp.onnx_path
                            ));
                        }
                        Ok(exp_resp) => {
                            sink.on_log(format!(
                                "[LS-Trainer Warning] ONNX export did not complete: {}",
                                exp_resp.message
                            ));
                        }
                        Err(err) => {
                            sink.on_log(format!("[LS-Trainer Warning] ONNX export failed: {err}"));
                        }
                    }
                }

                if let Ok(mut guard) = state.lock() {
                    guard.refresh_run_artifacts();
                    guard.is_running = false;
                    guard.is_paused = false;
                    guard.training_ended_at = Some(Instant::now());
                    if train_resp.success {
                        guard.exit_code = Some(0);
                        guard.status_line = if let Some(ref onnx) = guard.onnx_path {
                            format!("Completed — {onnx}")
                        } else {
                            "Completed".to_string()
                        };
                    } else if guard.stop_initiated {
                        if guard.exit_code.is_none() {
                            guard.exit_code = Some(130);
                        }
                        guard.status_line = "Stopped by user".to_string();
                    } else if guard.exit_code.is_none() {
                        guard.exit_code = Some(1);
                        guard.record_error(&train_resp.message);
                        guard.status_line = train_resp.message;
                    }
                }
            }
            Err(err) => {
                sink.on_log(format!("[LS-Trainer Error] {err}"));
                if let Ok(mut guard) = state.lock() {
                    guard.refresh_run_artifacts();
                    guard.exit_code = Some(1);
                    guard.is_running = false;
                    guard.is_paused = false;
                    guard.training_ended_at = Some(Instant::now());
                    guard.record_error(&err);
                    guard.status_line = "Training failed".to_string();
                }
            }
        }
    });
}

pub fn parse_cli_args() -> Result<Option<CliConfig>, String> {
    parse_cli_args_from(env::args().skip(1))
}

pub fn parse_cli_args_from<I, S>(args_iter: I) -> Result<Option<CliConfig>, String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let args: Vec<String> = args_iter.into_iter().map(Into::into).collect();
    if args.is_empty() {
        return Ok(Some(CliConfig::default()));
    }

    let mut cfg = CliConfig {
        interactive: false,
        dataset: String::new(),
        ..CliConfig::default()
    };
    let mut explicit_model = false;

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        let (flag, inline_val) = if let Some((k, v)) = arg.split_once('=') {
            (k, Some(v.to_string()))
        } else {
            (arg.as_str(), None)
        };

        let mut next_val = || -> Result<String, String> {
            if let Some(ref v) = inline_val {
                return Ok(v.clone());
            }
            i += 1;
            args.get(i)
                .cloned()
                .ok_or_else(|| format!("Missing value for flag '{flag}'"))
        };

        match flag {
            "-h" | "--help" => {
                println!(
                    "ls-trainer-tui — btop-style terminal trainer for YOLO, RF-DETR, and RT-DETR\n\n\
                     USAGE:\n  \
                       ls-trainer-tui [OPTIONS]\n\n\
                     OPTIONS:\n  \
                       --arch <yolo|rf-detr|rt-detr>   Model architecture family\n  \
                       --dataset <path>                Dataset folder or archive path\n  \
                       --model <path>                  Model ID or weights path\n  \
                       --epochs <n>                    Number of training epochs (default: 60)\n  \
                       --batch <n>                     Batch size (default: 4)\n  \
                       --device <str>                  Device (auto, xpu, cuda, 0, cpu)\n  \
                       --imgsz <n>                     Image resolution (640, 960, 1280)\n  \
                       --workers <n>                   DataLoader workers (default: auto; 0 for Windows Arc)\n  \
                       --run-name <name>               Output run directory name\n"
                );
                return Ok(None);
            }
            "--arch" => {
                let raw = next_val()?;
                cfg.arch = ArchChoice::from_str(&raw).ok_or_else(|| {
                    format!("Unsupported --arch '{raw}'. Expected yolo, rf-detr, or rt-detr.")
                })?;
                if !explicit_model {
                    cfg.model = cfg.arch.default_model().to_string();
                }
            }
            "--dataset" => {
                cfg.dataset = next_val()?;
            }
            "--model" => {
                cfg.model = next_val()?;
                explicit_model = true;
            }
            "--epochs" => {
                cfg.epochs = next_val()?;
            }
            "--batch" => {
                cfg.batch = next_val()?;
            }
            "--device" => {
                cfg.device = next_val()?;
            }
            "--imgsz" => {
                let raw = next_val()?;
                cfg.imgsz = raw
                    .parse::<u32>()
                    .map_err(|_| format!("Invalid --imgsz value '{raw}'"))?;
            }
            "--workers" => {
                let raw = next_val()?;
                let parsed = raw
                    .parse::<u32>()
                    .map_err(|_| format!("Invalid --workers value '{raw}'"))?;
                cfg.workers = Some(parsed);
                cfg.workers_input = parsed.to_string();
            }
            "--run-name" => {
                cfg.run_name = next_val()?;
            }
            other => {
                return Err(format!(
                    "Unknown argument '{other}'. Run with --help for usage."
                ));
            }
        }
        i += 1;
    }

    Ok(Some(cfg))
}
