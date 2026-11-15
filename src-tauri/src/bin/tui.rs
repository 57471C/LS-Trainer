//! `ls-trainer-tui` — Cross-platform `btop`-style terminal UI for LS-Trainer.

use std::collections::VecDeque;
use std::env;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use app_lib as ls_trainer_lib;
use arboard::Clipboard;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    terminal::{disable_raw_mode, enable_raw_mode},
};
use ls_trainer_lib::{
    active_training_control, force_kill_active_child, probe_local_gpu_info, run_export_to_onnx,
    run_training, send_training_control, stop_training, to_display_path, ExportRequest,
    GpuTelemetry, ProcessControlCommand, TrainMetrics, TrainingEventSink, TrainingRequest,
};
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{
        Axis, Block, BorderType, Borders, Chart, Clear, Dataset, GraphType, Paragraph, Sparkline,
    },
    Frame,
};
use sysinfo::System;

const SPLASH_ASCII_LINES: [&str; 8] = [
    r" /$$        /$$$$$$       /$$$$$$$$                 /$$                              ",
    r"| $$       /$$__  $$     |__  $$__/                |__/                              ",
    r"| $$      | $$  \__/        | $$  /$$$$$$  /$$$$$$  /$$ /$$$$$$$   /$$$$$$   /$$$$$$ ",
    r"| $$      |  $$$$$$  /$$$$$$| $$ /$$__  $$|____  $$| $$| $$__  $$ /$$__  $$ /$$__  $$",
    r"| $$       \____  $$|______/| $$| $$  \__/ /$$$$$$$| $$| $$  \ $$| $$$$$$$$| $$  \__/",
    r"| $$       /$$  \ $$        | $$| $$      /$$__  $$| $$| $$  | $$| $$_____/| $$      ",
    r"| $$$$$$$$|  $$$$$$/        | $$| $$     |  $$$$$$$| $$| $$  | $$|  $$$$$$$| $$      ",
    r"|________/ \______/         |__/|__/      \_______/|__/|__/  |__/ \_______/|__/      ",
];

const MAX_LOG_LINES: usize = 500;
const MAX_HISTORY_POINTS: usize = 300;
const MAX_SPARKLINE_SAMPLES: usize = 60;

const BAR_TRACK_BG: Color = Color::Rgb(26, 32, 44);
const BAR_GPU_TEAL: Color = Color::Rgb(68, 168, 186);
const BAR_EPOCH_TEAL: Color = Color::Rgb(62, 154, 174);
const BAR_STEP_BLUE: Color = Color::Rgb(72, 128, 184);
const BAR_OVERALL_CYAN: Color = Color::Rgb(56, 142, 168);
const BAR_RAM_SLATE: Color = Color::Rgb(92, 148, 188);
const GAUGE_TEXT_COVERED: Color = Color::Rgb(16, 20, 26);
const GAUGE_TEXT_UNCOVERED: Color = Color::Rgb(228, 234, 242);
const BORDER_SOFT: Color = Color::Rgb(64, 76, 94);
const TABLE_BORDER_SOFT: Color = Color::Rgb(74, 86, 104);
const TITLE_ACCENT: Color = Color::Rgb(116, 202, 218);
const LABEL_MUTED: Color = Color::Rgb(138, 150, 168);
const NEW_BEST_GOLD: Color = Color::Rgb(255, 216, 102);
const NEW_BEST_BADGE_BG: Color = Color::Rgb(42, 92, 66);
const OVERLAY_BG: Color = Color::Rgb(18, 22, 30);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArchChoice {
    Yolo,
    RfDetr,
    RtDetr,
}

impl ArchChoice {
    fn from_str(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "yolo" | "ultralytics" => Some(Self::Yolo),
            "rf-detr" | "rfdetr" => Some(Self::RfDetr),
            "rt-detr" | "rtdetr" => Some(Self::RtDetr),
            _ => None,
        }
    }

    fn as_family_str(self) -> &'static str {
        match self {
            Self::Yolo => "YOLO",
            Self::RfDetr => "RF-DETR",
            Self::RtDetr => "RT-DETR",
        }
    }

    fn as_cli_str(self) -> &'static str {
        match self {
            Self::Yolo => "yolo",
            Self::RfDetr => "rf-detr",
            Self::RtDetr => "rt-detr",
        }
    }

    fn default_model(self) -> &'static str {
        match self {
            Self::Yolo => "yolo11s.pt",
            Self::RfDetr => "rf-detr-medium",
            Self::RtDetr => "rtdetr-v2-r18",
        }
    }

    fn next(self) -> Self {
        match self {
            Self::Yolo => Self::RfDetr,
            Self::RfDetr => Self::RtDetr,
            Self::RtDetr => Self::Yolo,
        }
    }

    fn prev(self) -> Self {
        match self {
            Self::Yolo => Self::RtDetr,
            Self::RfDetr => Self::Yolo,
            Self::RtDetr => Self::RfDetr,
        }
    }
}

const LAUNCHER_FIELD_COUNT: usize = 7;
const DEVICE_PRESETS: [&str; 6] = ["auto", "xpu", "xpu:0", "xpu:0,1", "cuda", "cpu"];
const WORKER_PRESETS: [&str; 6] = ["auto", "0", "1", "2", "4", "8"];

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
                    if !trimmed.is_empty() && std::path::Path::new(trimmed).exists() {
                        return trimmed.to_string();
                    }
                }
            }
        }
    }
    String::new()
}

#[derive(Debug, Clone)]
struct CliConfig {
    interactive: bool,
    arch: ArchChoice,
    dataset: String,
    model: String,
    epochs: String,
    batch: String,
    device: String,
    imgsz: u32,
    workers: Option<u32>,
    workers_input: String,
    run_name: String,
    focused_field: usize,
    editing_field: bool,
    error_banner: Option<String>,
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
    fn parse_workers(&self) -> Result<Option<u32>, String> {
        let raw = self.workers_input.trim();
        if raw.is_empty() || raw.eq_ignore_ascii_case("auto") {
            return Ok(self.workers);
        }
        let parsed = raw
            .parse::<u32>()
            .map_err(|_| format!("Invalid workers value: '{raw}' (use a number or 'auto')"))?;
        Ok(Some(parsed))
    }

    fn to_training_request(&self) -> Result<TrainingRequest, String> {
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
struct GpuCardState {
    index: u32,
    name: String,
    vram_used_gb: f64,
    vram_total_gb: f64,
    util_percent: f64,
    power_watts: Option<f64>,
    temp_celsius: Option<f64>,
    util_history: VecDeque<u64>,
}

impl GpuCardState {
    fn new(index: u32, name: String) -> Self {
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

    fn push_util(&mut self, util: f64) {
        self.util_percent = util.clamp(0.0, 100.0);
        if self.util_history.len() >= MAX_SPARKLINE_SAMPLES {
            self.util_history.pop_front();
        }
        self.util_history
            .push_back(self.util_percent.round() as u64);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UiMode {
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
    mode: UiMode,
    overlay: OverlayMode,
    setup: CliConfig,
    active_request: Option<TrainingRequest>,
    is_running: bool,
    is_paused: bool,
    stop_initiated: bool,
    exit_code: Option<i32>,
    status_line: String,
    health_level: HealthLevel,
    warning_count: usize,
    suppressed_warning_count: usize,
    warning_tags: Vec<String>,
    last_error_msg: Option<String>,
    latest_metrics: TrainMetrics,
    step_counter: u64,
    train_loss_history: VecDeque<(f64, f64)>,
    val_loss_history: VecDeque<(f64, f64)>,
    map50_95_history: VecDeque<(f64, f64)>,
    best_map50: Option<f64>,
    best_map50_95: Option<f64>,
    best_loss: Option<f64>,
    best_ckpt_mtime: Option<std::time::SystemTime>,
    new_best_until: Option<Instant>,
    dataset_train_images: Option<u32>,
    gpus: Vec<GpuCardState>,
    host_cpu_percent: f64,
    host_ram_used_gb: f64,
    host_ram_total_gb: f64,
    host_cpu_history: VecDeque<u64>,
    host_core_count: usize,
    host_load_avg: (f64, f64, f64),
    logs: VecDeque<String>,
    log_scroll_offset: usize,
    follow_logs: bool,
    spinner_tick: usize,
    training_started_at: Option<Instant>,
    training_ended_at: Option<Instant>,
    results_dir: Option<String>,
    best_weights_path: Option<String>,
    onnx_path: Option<String>,
    history_entries: Vec<RunHistoryEntry>,
    history_selected: usize,
}

fn strip_ansi_codes(s: &str) -> String {
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
fn classify_noisy_warning(clean_lower: &str) -> Option<&'static str> {
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
    fn new(setup: CliConfig) -> Self {
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

    fn trigger_new_best(&mut self) {
        self.new_best_until = Some(Instant::now() + Duration::from_secs(6));
    }

    fn is_new_best_active(&self) -> bool {
        self.new_best_until
            .map_or(false, |until| Instant::now() < until)
    }

    fn record_warning_tag(&mut self, tag: &str) {
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

    fn record_error(&mut self, msg: &str) {
        self.health_level = HealthLevel::Error;
        let trimmed = msg.trim();
        if !trimmed.is_empty() {
            self.last_error_msg = Some(trimmed.to_string());
        }
    }

    fn refresh_run_artifacts(&mut self) {
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

    fn elapsed_seconds(&self) -> f64 {
        match (self.training_started_at, self.training_ended_at) {
            (Some(start), Some(end)) => end.saturating_duration_since(start).as_secs_f64(),
            (Some(start), None) => start.elapsed().as_secs_f64(),
            _ => 0.0,
        }
    }

    fn push_log(&mut self, line: String) {
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
            let p = clean_trimmed[idx + "ONNX export completed successfully! File:".len()..].trim();
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

    fn apply_metrics(&mut self, mut metrics: TrainMetrics) {
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

    fn merge_metric_gpus(&mut self, incoming: &[GpuTelemetry]) {
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
        // PCI display controller classes: 0x0300 (VGA), 0x0302 (3D), 0x0380 (Display)
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

                // Temperature in millidegrees Celsius
                for temp_file in ["temp1_input", "temp2_input"] {
                    if let Some(milli_c) = read_trimmed_u64(&hw_dir.join(temp_file)) {
                        if milli_c > 0 && milli_c < 150_000 {
                            temp_c = Some(milli_c as f64 / 1000.0);
                            break;
                        }
                    }
                }

                // Direct power in microwatts
                for power_file in ["power1_input", "power1_average"] {
                    if let Some(micro_w) = read_trimmed_u64(&hw_dir.join(power_file)) {
                        if micro_w > 0 {
                            power_w = Some(micro_w as f64 / 1_000_000.0);
                            break;
                        }
                    }
                }

                // Fallback: energy1_input in microjoules (used by Intel xe / i915 hwmon)
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

fn spawn_hardware_polling_tasks(state: Arc<Mutex<AppState>>) {
    // Seed initial GPU info from LS-Trainer hardware probe
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

    // Cross-platform sysinfo CPU / RAM polling task (runs on both Windows and Linux)
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

                #[cfg(not(target_os = "linux"))]
                {
                    let _ = (&guard.host_cpu_percent, &guard.host_ram_used_gb);
                }
            }

            std::thread::sleep(Duration::from_millis(1000));
        }
    });
}

fn find_best_checkpoint_in_dir(run_dir: &Path) -> Option<PathBuf> {
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

fn find_onnx_in_dir(run_dir: &Path) -> Option<PathBuf> {
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

fn count_dataset_train_images(dataset_path: &str) -> Option<u32> {
    let p = Path::new(dataset_path.trim());
    if !p.exists() {
        return None;
    }
    let root = if p.is_file() { p.parent()? } else { p };

    // 1. Check COCO annotations JSON
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

    // 2. Check YOLO images/train directory
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

fn scan_run_history() -> Vec<RunHistoryEntry> {
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

fn launch_training_thread(state: Arc<Mutex<AppState>>, request: TrainingRequest) {
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
                        guard.best_map50 = Some(guard.best_map50.map_or(m50, |prev| prev.max(m50)));
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

fn format_eta_seconds(eta_secs: f64) -> String {
    if !eta_secs.is_finite() || eta_secs <= 0.0 {
        return "--:--".to_string();
    }
    let total = eta_secs.round() as u64;
    let hours = total / 3600;
    let mins = (total % 3600) / 60;
    let secs = total % 60;
    if hours > 0 {
        format!("{hours:02}:{mins:02}:{secs:02}")
    } else {
        format!("{mins:02}:{secs:02}")
    }
}

/// Map heavy or double box-drawing characters to softer light/rounded equivalents.
fn soften_box_char(ch: char) -> Option<char> {
    match ch {
        '┏' | '╔' | '┌' | '╭' => Some('╭'),
        '┓' | '╗' | '┐' | '╮' => Some('╮'),
        '┗' | '╚' | '└' | '╰' => Some('╰'),
        '┛' | '╝' | '┘' | '╯' => Some('╯'),
        '━' | '═' | '─' | '╌' | '╍' => Some('─'),
        '┃' | '║' | '│' | '╎' | '╏' => Some('│'),
        '┳' | '┯' | '╦' | '╥' | '╤' | '┬' => Some('┬'),
        '┻' | '┷' | '╩' | '╨' | '╧' | '┴' => Some('┴'),
        '┣' | '┡' | '┢' | '┠' | '┝' | '╞' | '╟' | '╠' | '├' => Some('├'),
        '┫' | '┩' | '┪' | '┨' | '┥' | '╡' | '╢' | '╣' | '┤' => Some('┤'),
        '╋' | '╇' | '╈' | '╉' | '╊' | '╀' | '╁' | '╂' | '┿' | '┽' | '┾' | '┼' | '╬' | '╪' | '╫' => {
            Some('┼')
        }
        c if ('\u{2500}'..='\u{257F}').contains(&c) => Some(c),
        _ => None,
    }
}

fn push_softened_box_spans(
    spans: &mut Vec<Span<'static>>,
    segment: &str,
    text_style: Style,
    soft_pipe_separators: bool,
) {
    if segment.is_empty() {
        return;
    }
    let border_style = Style::default().fg(TABLE_BORDER_SOFT);
    let mut cur = String::new();
    let mut cur_is_border = false;

    for ch in segment.chars() {
        let mapped_border = soften_box_char(ch).or_else(|| {
            if soft_pipe_separators && ch == '|' {
                Some('│')
            } else {
                None
            }
        });
        let is_border = mapped_border.is_some();
        let out_ch = mapped_border.unwrap_or(ch);

        if !cur.is_empty() && cur_is_border != is_border {
            let st = if cur_is_border {
                border_style
            } else {
                text_style
            };
            spans.push(Span::styled(std::mem::take(&mut cur), st));
        }
        cur_is_border = is_border;
        cur.push(out_ch);
    }

    if !cur.is_empty() {
        let st = if cur_is_border {
            border_style
        } else {
            text_style
        };
        spans.push(Span::styled(cur, st));
    }
}

/// Render a 1-row (or multi-row) progress bar where label characters covered by the bar
/// are drawn in dark grey/black (`GAUGE_TEXT_COVERED`) and uncovered characters are drawn
/// in crisp light text (`GAUGE_TEXT_UNCOVERED`).
fn render_contrast_gauge(frame: &mut Frame, area: Rect, ratio: f64, label: &str, bar_color: Color) {
    if area.is_empty() {
        return;
    }
    let clamped = ratio.clamp(0.0, 1.0);
    let filled_width = (f64::from(area.width) * clamped).round() as u16;
    let filled_end = area.left() + filled_width.min(area.width);

    let buf = frame.buffer_mut();
    for y in area.top()..area.bottom() {
        for x in area.left()..area.right() {
            let covered = x < filled_end;
            let bg = if covered { bar_color } else { BAR_TRACK_BG };
            let fg = if covered {
                GAUGE_TEXT_COVERED
            } else {
                GAUGE_TEXT_UNCOVERED
            };
            buf[(x, y)].set_symbol(" ").set_fg(fg).set_bg(bg);
        }
    }

    let label_chars: Vec<char> = label.chars().collect();
    let label_len = (label_chars.len() as u16).min(area.width);
    let label_col = area.left() + (area.width.saturating_sub(label_len)) / 2;
    let label_row = area.top() + area.height / 2;

    for (i, ch) in label_chars.into_iter().take(label_len as usize).enumerate() {
        let x = label_col + i as u16;
        if x >= area.right() {
            break;
        }
        let covered = x < filled_end;
        let bg = if covered { bar_color } else { BAR_TRACK_BG };
        let fg = if covered {
            GAUGE_TEXT_COVERED
        } else {
            GAUGE_TEXT_UNCOVERED
        };
        let mut utf8_buf = [0u8; 4];
        let sym = ch.encode_utf8(&mut utf8_buf);
        buf[(x, label_row)]
            .set_symbol(sym)
            .set_style(Style::default().fg(fg).bg(bg).add_modifier(Modifier::BOLD));
    }
}

/// Parse a log line containing optional ANSI SGR escape sequences (`\x1b[...m`) into colored `Span`s.
fn parse_ansi_or_styled_log_line(raw: &str) -> Line<'static> {
    if raw.contains('\x1b') {
        let mut spans: Vec<Span<'static>> = Vec::new();
        let mut current_style = Style::default().fg(Color::Gray);
        let mut segment = String::new();
        let mut chars = raw.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '\x1b' && chars.peek() == Some(&'[') {
                if !segment.is_empty() {
                    let seg = std::mem::take(&mut segment);
                    push_softened_box_spans(&mut spans, &seg, current_style, false);
                }
                chars.next(); // consume '['
                let mut code_str = String::new();
                while let Some(&nc) = chars.peek() {
                    chars.next();
                    if ('@'..='~').contains(&nc) {
                        if nc == 'm' {
                            current_style = apply_sgr_codes(&code_str, current_style);
                        }
                        break;
                    }
                    code_str.push(nc);
                }
            } else {
                segment.push(ch);
            }
        }
        if !segment.is_empty() {
            push_softened_box_spans(&mut spans, &segment, current_style, false);
        }
        return Line::from(spans);
    }

    let is_metrics_line = raw.contains("[LS-Trainer Metrics]") || raw.contains("Final Metrics");
    let base_color = if raw.contains("Error]") || raw.contains("Traceback") {
        Color::LightRed
    } else if raw.contains("Warning]") {
        Color::Yellow
    } else if is_metrics_line {
        Color::Rgb(132, 214, 158)
    } else if raw.starts_with("[LS-Trainer") {
        TITLE_ACCENT
    } else if raw.starts_with("Epoch ") {
        Color::Rgb(130, 182, 232)
    } else {
        Color::Rgb(194, 204, 216)
    };

    let mut spans: Vec<Span<'static>> = Vec::new();
    push_softened_box_spans(
        &mut spans,
        raw,
        Style::default().fg(base_color),
        is_metrics_line || raw.starts_with("Epoch "),
    );
    Line::from(spans)
}

fn apply_sgr_codes(code_str: &str, mut style: Style) -> Style {
    if code_str.is_empty() {
        return Style::default().fg(Color::Gray);
    }
    for part in code_str.split(';') {
        match part.trim().parse::<u16>().unwrap_or(0) {
            0 => style = Style::default().fg(Color::Gray),
            1 => style = style.add_modifier(Modifier::BOLD),
            2 => style = style.add_modifier(Modifier::DIM),
            30 => style = style.fg(Color::DarkGray),
            31 | 91 => style = style.fg(Color::LightRed),
            32 | 92 => style = style.fg(Color::LightGreen),
            33 | 93 => style = style.fg(Color::Yellow),
            34 | 94 => style = style.fg(Color::LightBlue),
            35 | 95 => style = style.fg(Color::LightMagenta),
            36 | 96 => style = style.fg(Color::Cyan),
            37 | 97 => style = style.fg(Color::White),
            _ => {}
        }
    }
    style
}

fn render_top_gpu_strip(frame: &mut Frame, area: Rect, state: &AppState) {
    if state.gpus.is_empty() {
        // Render Host CPU/RAM card when no discrete GPU is reported yet
        render_host_card(frame, area, state);
        return;
    }

    let count = state.gpus.len();
    let constraints: Vec<Constraint> = (0..count)
        .map(|_| Constraint::Ratio(1, count as u32))
        .collect();
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(constraints)
        .split(area);

    for (idx, gpu) in state.gpus.iter().enumerate() {
        let card_area = chunks[idx];
        let power_str = gpu
            .power_watts
            .map(|w| format!("{w:.0}W"))
            .unwrap_or_else(|| "--W".to_string());
        let temp_str = gpu
            .temp_celsius
            .map(|t| format!("{t:.0}°C"))
            .unwrap_or_else(|| "--°C".to_string());

        let title = format!(
            " GPU {}: {} │ {} │ {} ",
            gpu.index, gpu.name, power_str, temp_str
        );
        let block = Block::default()
            .title(Span::styled(
                title,
                Style::default()
                    .fg(TITLE_ACCENT)
                    .add_modifier(Modifier::BOLD),
            ))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(BORDER_SOFT));

        let inner = block.inner(card_area);
        frame.render_widget(block, card_area);

        if inner.height == 0 || inner.width == 0 {
            continue;
        }

        let padded = Rect {
            x: inner.x + 1,
            y: inner.y,
            width: inner.width.saturating_sub(2),
            height: inner.height,
        };
        if padded.width == 0 {
            continue;
        }

        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(1)])
            .split(padded);

        let vram_ratio = if gpu.vram_total_gb > 0.0 {
            (gpu.vram_used_gb / gpu.vram_total_gb).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let vram_label = if gpu.vram_total_gb > 0.0 {
            format!(
                "VRAM {:.2} / {:.2} GiB ({:.0}%)",
                gpu.vram_used_gb,
                gpu.vram_total_gb,
                vram_ratio * 100.0
            )
        } else {
            format!("VRAM {:.2} GiB", gpu.vram_used_gb)
        };

        render_contrast_gauge(frame, rows[0], vram_ratio, &vram_label, BAR_GPU_TEAL);

        let spark_cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Length(13), Constraint::Min(4)])
            .split(rows[1]);

        let util_text = Paragraph::new(Line::from(vec![
            Span::styled("Util ", Style::default().fg(LABEL_MUTED)),
            Span::styled(
                format!("{:>5.1}%", gpu.util_percent),
                Style::default()
                    .fg(Color::LightGreen)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
        frame.render_widget(util_text, spark_cols[0]);

        let samples: Vec<u64> = gpu.util_history.iter().copied().collect();
        let sparkline = Sparkline::default()
            .data(&samples)
            .max(100)
            .style(Style::default().fg(Color::Rgb(106, 204, 142)));
        frame.render_widget(sparkline, spark_cols[1]);
    }
}

fn render_host_card(frame: &mut Frame, area: Rect, state: &AppState) {
    let title = format!(
        " Host Telemetry │ CPU {:.1}% │ RAM {:.1}/{:.1} GiB ",
        state.host_cpu_percent, state.host_ram_used_gb, state.host_ram_total_gb
    );
    let block = Block::default()
        .title(Span::styled(
            title,
            Style::default()
                .fg(TITLE_ACCENT)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(BORDER_SOFT));

    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height == 0 || inner.width <= 2 {
        return;
    }

    let padded = Rect {
        x: inner.x + 1,
        y: inner.y,
        width: inner.width.saturating_sub(2),
        height: inner.height,
    };

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(1)])
        .split(padded);

    let ram_ratio = if state.host_ram_total_gb > 0.0 {
        (state.host_ram_used_gb / state.host_ram_total_gb).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let ram_label = format!(
        "System RAM {:.1} / {:.1} GiB ({:.0}%)",
        state.host_ram_used_gb,
        state.host_ram_total_gb,
        ram_ratio * 100.0
    );
    render_contrast_gauge(frame, rows[0], ram_ratio, &ram_label, BAR_GPU_TEAL);

    let samples: Vec<u64> = state.host_cpu_history.iter().copied().collect();
    let spark = Sparkline::default()
        .data(&samples)
        .max(100)
        .style(Style::default().fg(Color::Rgb(106, 204, 142)));
    frame.render_widget(spark, rows[1]);
}

fn render_center_chart(frame: &mut Frame, area: Rect, state: &AppState) {
    let train_pts: Vec<(f64, f64)> = state.train_loss_history.iter().copied().collect();
    let val_pts: Vec<(f64, f64)> = state.val_loss_history.iter().copied().collect();
    let map_pts: Vec<(f64, f64)> = state.map50_95_history.iter().copied().collect();

    let mut min_x = 1.0_f64;
    let mut max_x = 10.0_f64;
    let mut min_y = 0.0_f64;
    let mut max_y = 1.0_f64;

    let all_pts = train_pts.iter().chain(val_pts.iter()).chain(map_pts.iter());

    let mut first = true;
    for &(x, y) in all_pts {
        if first {
            min_x = x;
            max_x = x;
            min_y = y.min(0.0);
            max_y = y;
            first = false;
        } else {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
    }
    if (max_x - min_x).abs() < 1.0 {
        max_x = min_x + 10.0;
    }
    if (max_y - min_y).abs() < 1e-4 {
        max_y = min_y + 1.0;
    } else {
        max_y *= 1.08;
    }

    let datasets = vec![
        Dataset::default()
            .name("train_loss")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Cyan))
            .data(&train_pts),
        Dataset::default()
            .name("val_loss")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Yellow))
            .data(&val_pts),
        Dataset::default()
            .name("mAP@50:95")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::LightGreen))
            .data(&map_pts),
    ];

    let mid_x = (min_x + max_x) * 0.5;
    let mid_y = (min_y + max_y) * 0.5;

    let chart = Chart::new(datasets)
        .block(
            Block::default()
                .title(Span::styled(
                    " Telemetry Curves (train_loss · val_loss · mAP@50:95) ",
                    Style::default()
                        .fg(TITLE_ACCENT)
                        .add_modifier(Modifier::BOLD),
                ))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(BORDER_SOFT)),
        )
        .x_axis(
            Axis::default()
                .title(Span::styled("Step", Style::default().fg(LABEL_MUTED)))
                .style(Style::default().fg(BORDER_SOFT))
                .bounds([min_x, max_x])
                .labels(vec![
                    Span::raw(format!("{min_x:.0}")),
                    Span::raw(format!("{mid_x:.0}")),
                    Span::raw(format!("{max_x:.0}")),
                ]),
        )
        .y_axis(
            Axis::default()
                .title(Span::styled("Value", Style::default().fg(LABEL_MUTED)))
                .style(Style::default().fg(BORDER_SOFT))
                .bounds([min_y, max_y])
                .labels(vec![
                    Span::raw(format!("{min_y:.2}")),
                    Span::raw(format!("{mid_y:.2}")),
                    Span::raw(format!("{max_y:.2}")),
                ]),
        );

    frame.render_widget(chart, area);
}

fn compact_display_path(path: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let count = path.chars().count();
    if count <= max_chars {
        return path.to_string();
    }
    // Prefer preserving from `runs\` or `runs/` onwards if that fits
    for marker in ["runs\\", "runs/"] {
        if let Some(pos) = path.find(marker) {
            let rel = &path[pos..];
            if rel.chars().count() <= max_chars {
                return rel.to_string();
            }
        }
    }
    let keep = max_chars.saturating_sub(1);
    let tail: String = path
        .chars()
        .rev()
        .take(keep)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("…{tail}")
}

fn render_run_summary(frame: &mut Frame, area: Rect, state: &AppState) {
    let host_h = if area.height >= 19 { 5 } else { 4 };
    let split = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(host_h), Constraint::Min(10)])
        .split(area);

    render_host_cpu_panel(frame, split[0], state);
    render_run_details_panel(frame, split[1], state);
}

fn render_host_cpu_panel(frame: &mut Frame, area: Rect, state: &AppState) {
    let block = Block::default()
        .title(Span::styled(
            " Host / CPU ",
            Style::default()
                .fg(TITLE_ACCENT)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(BORDER_SOFT));

    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height == 0 || inner.width <= 2 {
        return;
    }

    let padded = Rect {
        x: inner.x + 1,
        y: inner.y,
        width: inner.width.saturating_sub(2),
        height: inner.height,
    };

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(1),
        ])
        .split(padded);

    let cpu_ratio = (state.host_cpu_percent / 100.0).clamp(0.0, 1.0);
    let cores_label = if state.host_core_count > 0 {
        format!(" ({} cores)", state.host_core_count)
    } else {
        String::new()
    };
    let cpu_label = format!("CPU {:.1}%{cores_label}", state.host_cpu_percent);
    render_contrast_gauge(frame, rows[0], cpu_ratio, &cpu_label, BAR_GPU_TEAL);

    let ram_ratio = if state.host_ram_total_gb > 0.0 {
        (state.host_ram_used_gb / state.host_ram_total_gb).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let ram_pct = ram_ratio * 100.0;
    let ram_label = format!(
        "RAM {:.1} / {:.1} GiB ({ram_pct:.0}%)",
        state.host_ram_used_gb, state.host_ram_total_gb
    );
    render_contrast_gauge(frame, rows[1], ram_ratio, &ram_label, BAR_RAM_SLATE);

    if rows[2].height > 0 {
        let bottom_cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Length(18), Constraint::Min(4)])
            .split(rows[2]);
        let (l1, l5, _l15) = state.host_load_avg;
        let load_line = Line::from(vec![
            Span::styled("Load ", Style::default().fg(LABEL_MUTED)),
            Span::styled(
                format!("{l1:.1} {l5:.1}"),
                Style::default().fg(Color::White),
            ),
        ]);
        frame.render_widget(Paragraph::new(load_line), bottom_cols[0]);

        let cpu_samples: Vec<u64> = state.host_cpu_history.iter().copied().collect();
        let sparkline = Sparkline::default()
            .data(&cpu_samples)
            .max(100)
            .style(Style::default().fg(BAR_GPU_TEAL));
        frame.render_widget(sparkline, bottom_cols[1]);
    }
}

fn render_run_details_panel(frame: &mut Frame, area: Rect, state: &AppState) {
    let block = Block::default()
        .title(Span::styled(
            " Run Summary ",
            Style::default()
                .fg(TITLE_ACCENT)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(BORDER_SOFT));

    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height == 0 || inner.width <= 2 {
        return;
    }

    let padded = Rect {
        x: inner.x + 1,
        y: inner.y,
        width: inner.width.saturating_sub(2),
        height: inner.height,
    };

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(4),
        ])
        .split(padded);

    let m = &state.latest_metrics;
    let total_ep = m.total_epochs.max(1);
    let ep_ratio = if m.total_epochs > 0 {
        (m.epoch as f64 / total_ep as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let ep_label = format!(
        "Epoch {} / {} ({:.0}%)",
        m.epoch,
        m.total_epochs,
        ep_ratio * 100.0
    );
    render_contrast_gauge(frame, rows[0], ep_ratio, &ep_label, BAR_EPOCH_TEAL);

    let step_ratio = if m.total_steps > 0 {
        (m.step as f64 / m.total_steps as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let step_label = format!(
        "Step {} / {} ({:.0}%)",
        m.step,
        m.total_steps,
        step_ratio * 100.0
    );
    render_contrast_gauge(frame, rows[1], step_ratio, &step_label, BAR_STEP_BLUE);

    let (arch_str, model_str, _dataset_str, dev_str, batch_val) =
        if let Some(ref req) = state.active_request {
            (
                req.model_family.as_str(),
                req.model_choice.as_str(),
                req.dataset_path.as_str(),
                req.device.as_str(),
                req.batch.max(1),
            )
        } else {
            let b = state.setup.batch.trim().parse::<u32>().unwrap_or(4).max(1);
            (
                state.setup.arch.as_family_str(),
                state.setup.model.as_str(),
                state.setup.dataset.as_str(),
                state.setup.device.as_str(),
                b,
            )
        };

    // Overall progress across epochs + steps + samples/images
    let overall_ratio = if !state.is_running && state.exit_code == Some(0) {
        1.0
    } else if m.total_epochs > 0 && m.total_steps > 0 && m.epoch > 0 {
        let done_steps = m
            .epoch
            .saturating_sub(1)
            .saturating_mul(m.total_steps)
            .saturating_add(m.step.min(m.total_steps));
        let all_steps = total_ep.saturating_mul(m.total_steps).max(1);
        (done_steps as f64 / all_steps as f64).clamp(0.0, 1.0)
    } else {
        ep_ratio
    };

    let imgs_per_epoch = state.dataset_train_images.or_else(|| {
        if m.total_steps > 0 {
            let dev_mult = state.gpus.len().max(1) as u32;
            Some(
                m.total_steps
                    .saturating_mul(batch_val)
                    .saturating_mul(dev_mult),
            )
        } else {
            None
        }
    });
    let samples_str = if let Some(per_ep) = imgs_per_epoch {
        let total_samples = per_ep.saturating_mul(total_ep);
        let done_samples =
            ((overall_ratio * f64::from(total_samples)).round() as u32).min(total_samples);
        format!("{done_samples}/{total_samples} imgs")
    } else {
        "-- imgs".to_string()
    };
    let overall_label = format!(
        "Overall {:.0}% · Ep {}/{} · Step {}/{} · {}",
        overall_ratio * 100.0,
        m.epoch,
        m.total_epochs,
        m.step,
        m.total_steps,
        samples_str
    );
    render_contrast_gauge(
        frame,
        rows[2],
        overall_ratio,
        &overall_label,
        BAR_OVERALL_CYAN,
    );

    let status_color = if state.is_paused {
        Color::Yellow
    } else if state.is_running {
        Color::LightGreen
    } else {
        TITLE_ACCENT
    };
    let spinner_frames = ["|", "/", "-", "\\"];
    let spinner_char = spinner_frames[state.spinner_tick % spinner_frames.len()];
    let status_badge = if state.is_paused {
        "PAUSED".to_string()
    } else if state.is_running {
        format!("TRAINING {spinner_char}")
    } else {
        state.status_line.clone()
    };

    let elapsed_str = match (state.training_started_at, state.training_ended_at) {
        (Some(start), Some(end)) => {
            format_eta_seconds(end.saturating_duration_since(start).as_secs_f64())
        }
        (Some(start), None) => format_eta_seconds(start.elapsed().as_secs_f64()),
        _ => "--:--".to_string(),
    };

    let map50_str = state
        .best_map50
        .map(|v| format!("{:.2}%", v * 100.0))
        .unwrap_or_else(|| "--".to_string());
    let map95_str = state
        .best_map50_95
        .map(|v| format!("{:.2}%", v * 100.0))
        .unwrap_or_else(|| "--".to_string());

    let new_best_active = state.is_new_best_active();
    let best_metric_color = if new_best_active {
        NEW_BEST_GOLD
    } else {
        Color::LightGreen
    };

    let mut best_map_spans = vec![
        Span::styled("Best mAP ", Style::default().fg(LABEL_MUTED)),
        Span::styled("50 ", Style::default().fg(LABEL_MUTED)),
        Span::styled(
            format!("{map50_str:<7}"),
            Style::default()
                .fg(best_metric_color)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" │ ", Style::default().fg(TABLE_BORDER_SOFT)),
        Span::styled("50:95 ", Style::default().fg(LABEL_MUTED)),
        Span::styled(
            map95_str,
            Style::default()
                .fg(best_metric_color)
                .add_modifier(Modifier::BOLD),
        ),
    ];
    if new_best_active {
        best_map_spans.push(Span::raw(" "));
        best_map_spans.push(Span::styled(
            " ★ New best ",
            Style::default()
                .fg(NEW_BEST_GOLD)
                .bg(NEW_BEST_BADGE_BG)
                .add_modifier(Modifier::BOLD),
        ));
    }

    let path_budget = padded.width.saturating_sub(9) as usize;
    let run_dir_disp = state
        .results_dir
        .as_deref()
        .map(|p| compact_display_path(p, path_budget))
        .unwrap_or_else(|| "pending...".to_string());
    let best_wts_disp = state
        .best_weights_path
        .as_deref()
        .map(|p| compact_display_path(p, path_budget))
        .unwrap_or_else(|| "pending (saving on epoch)...".to_string());
    let onnx_disp = state
        .onnx_path
        .as_deref()
        .map(|p| compact_display_path(p, path_budget))
        .unwrap_or_else(|| {
            if state.status_line.to_ascii_lowercase().contains("onnx") {
                "exporting best.onnx...".to_string()
            } else {
                "pending (exports on finish)".to_string()
            }
        });

    let sep_span = Span::styled("  │  ", Style::default().fg(TABLE_BORDER_SOFT));

    let lines = vec![
        Line::from(vec![
            Span::styled("State    ", Style::default().fg(LABEL_MUTED)),
            Span::styled(
                format!("{status_badge:<14}"),
                Style::default()
                    .fg(status_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("Dev ", Style::default().fg(LABEL_MUTED)),
            Span::styled(dev_str.to_string(), Style::default().fg(TITLE_ACCENT)),
        ]),
        Line::from(vec![
            Span::styled("Model    ", Style::default().fg(LABEL_MUTED)),
            Span::styled(
                format!("{arch_str} ({model_str})"),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled("Loss     ", Style::default().fg(LABEL_MUTED)),
            Span::styled(
                format!("{:<8.4}", m.train_loss),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            sep_span.clone(),
            Span::styled("LR ", Style::default().fg(LABEL_MUTED)),
            Span::styled(
                format!("{:.2e}", m.lr),
                Style::default()
                    .fg(Color::LightYellow)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(best_map_spans),
        Line::from(vec![
            Span::styled("Speed    ", Style::default().fg(LABEL_MUTED)),
            Span::styled(
                format!("{:>5.1} samples/s", m.samples_per_sec),
                Style::default()
                    .fg(Color::LightGreen)
                    .add_modifier(Modifier::BOLD),
            ),
            sep_span.clone(),
            Span::styled("Elapsed ", Style::default().fg(LABEL_MUTED)),
            Span::styled(elapsed_str, Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("ETA      ", Style::default().fg(LABEL_MUTED)),
            Span::styled(
                format!("{:<8}", format_eta_seconds(m.eta_secs)),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            sep_span,
            Span::styled("Samples ", Style::default().fg(LABEL_MUTED)),
            Span::styled(samples_str, Style::default().fg(TITLE_ACCENT)),
        ]),
        Line::from(vec![
            Span::styled("Run Dir  ", Style::default().fg(LABEL_MUTED)),
            Span::styled(run_dir_disp, Style::default().fg(Color::Rgb(196, 206, 218))),
        ]),
        Line::from(vec![
            Span::styled("Best Wts ", Style::default().fg(LABEL_MUTED)),
            Span::styled(
                best_wts_disp,
                Style::default().fg(if state.best_weights_path.is_some() {
                    Color::LightGreen
                } else {
                    LABEL_MUTED
                }),
            ),
        ]),
        Line::from(vec![
            Span::styled("ONNX     ", Style::default().fg(LABEL_MUTED)),
            Span::styled(
                onnx_disp,
                Style::default().fg(if state.onnx_path.is_some() {
                    TITLE_ACCENT
                } else {
                    LABEL_MUTED
                }),
            ),
        ]),
    ];

    frame.render_widget(Paragraph::new(lines), rows[3]);
}

fn render_health_status_bar(frame: &mut Frame, area: Rect, state: &AppState) {
    let (badge_text, badge_fg, badge_bg, detail_text, detail_color) = match state.health_level {
        HealthLevel::Healthy => (
            " ● Healthy ",
            GAUGE_TEXT_COVERED,
            Color::LightGreen,
            format!(" Training proceeding normally  │  {}", state.status_line),
            Color::Rgb(176, 214, 188),
        ),
        HealthLevel::Warning => {
            let tags = if state.warning_tags.is_empty() {
                "non-fatal".to_string()
            } else {
                state.warning_tags.join(", ")
            };
            let supp = if state.suppressed_warning_count > 0 {
                format!(", {} suppressed in log", state.suppressed_warning_count)
            } else {
                String::new()
            };
            (
                " ▲ Warning ",
                GAUGE_TEXT_COVERED,
                Color::Yellow,
                format!(
                    " {} non-fatal warning(s) [{}{}]  │  {}",
                    state.warning_count, tags, supp, state.status_line
                ),
                Color::Yellow,
            )
        }
        HealthLevel::Error => {
            let err_detail = state
                .last_error_msg
                .as_deref()
                .unwrap_or(&state.status_line);
            (
                " ✖ Error ",
                Color::White,
                Color::Red,
                format!(" {err_detail}"),
                Color::LightRed,
            )
        }
    };

    let line = Line::from(vec![
        Span::styled(
            badge_text,
            Style::default()
                .fg(badge_fg)
                .bg(badge_bg)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(detail_text, Style::default().fg(detail_color)),
    ]);
    frame.render_widget(
        Paragraph::new(line).style(Style::default().bg(BAR_TRACK_BG)),
        area,
    );
}

fn render_bottom_log_and_footer(frame: &mut Frame, area: Rect, state: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(3),
            Constraint::Length(1),
        ])
        .split(area);

    render_health_status_bar(frame, chunks[0], state);

    let log_block = Block::default()
        .title(Span::styled(
            format!(
                " Live Training Log ({}/{} lines) ",
                state.logs.len(),
                MAX_LOG_LINES
            ),
            Style::default()
                .fg(TITLE_ACCENT)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(BORDER_SOFT));

    let inner = log_block.inner(chunks[1]);
    frame.render_widget(log_block, chunks[1]);

    let padded = Rect {
        x: inner.x + 1,
        y: inner.y,
        width: inner.width.saturating_sub(2),
        height: inner.height,
    };

    let visible_height = padded.height as usize;
    let total_lines = state.logs.len();
    let max_offset = total_lines.saturating_sub(visible_height);
    let offset_from_bottom = state.log_scroll_offset.min(max_offset);
    let start_idx = total_lines.saturating_sub(visible_height + offset_from_bottom);
    let end_idx = (start_idx + visible_height).min(total_lines);

    let visible_lines: Vec<Line<'static>> = state
        .logs
        .iter()
        .skip(start_idx)
        .take(end_idx.saturating_sub(start_idx))
        .map(|l| parse_ansi_or_styled_log_line(l))
        .collect();

    frame.render_widget(Paragraph::new(visible_lines), padded);

    let pause_label = if state.is_paused { "Resume" } else { "Pause" };
    let footer = Line::from(vec![
        Span::styled(
            " [Space] ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!(" {pause_label} "),
            Style::default().fg(Color::White),
        ),
        Span::styled(
            " [c] ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::LightGreen)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Checkpoint ", Style::default().fg(Color::White)),
        Span::styled(
            " [↑/↓] ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Scroll ", Style::default().fg(Color::White)),
        Span::styled(
            " [F1/?] ",
            Style::default()
                .fg(GAUGE_TEXT_COVERED)
                .bg(BAR_GPU_TEAL)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Help ", Style::default().fg(Color::White)),
        Span::styled(
            " [F3] ",
            Style::default()
                .fg(GAUGE_TEXT_COVERED)
                .bg(BAR_STEP_BLUE)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Config ", Style::default().fg(Color::White)),
        Span::styled(
            " [F4/h] ",
            Style::default()
                .fg(GAUGE_TEXT_COVERED)
                .bg(BAR_RAM_SLATE)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" History ", Style::default().fg(Color::White)),
        Span::styled(
            " [F12] ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Copy Log ", Style::default().fg(Color::White)),
        Span::styled(
            " [q] ",
            Style::default()
                .fg(Color::White)
                .bg(Color::Red)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Stop & Quit ", Style::default().fg(Color::White)),
    ]);

    frame.render_widget(
        Paragraph::new(footer).alignment(Alignment::Center),
        chunks[2],
    );
}

fn render_launcher_screen(frame: &mut Frame, area: Rect, state: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),
            Constraint::Min(12),
            Constraint::Length(1),
        ])
        .split(area);

    render_top_gpu_strip(frame, chunks[0], state);

    let block = Block::default()
        .title(Span::styled(
            " LS-Trainer TUI — Launch Menu ",
            Style::default()
                .fg(TITLE_ACCENT)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(BORDER_SOFT));

    let inner = block.inner(chunks[1]);
    frame.render_widget(block, chunks[1]);

    if inner.height > 0 && inner.width > 2 {
        let padded = Rect {
            x: inner.x + 2,
            y: inner.y + 1,
            width: inner.width.saturating_sub(4),
            height: inner.height.saturating_sub(1),
        };

        let s = &state.setup;
        let ds_display = if s.dataset.is_empty() && !s.editing_field {
            "<enter dataset path>".to_string()
        } else {
            s.dataset.clone()
        };
        let workers_display = if s.workers_input.trim().is_empty() && !s.editing_field {
            "auto".to_string()
        } else {
            s.workers_input.clone()
        };

        let fields: [(&str, String, &str, bool); 6] = [
            (
                "Architecture",
                format!(
                    "◀ {} ({}) · {} ▶",
                    s.arch.as_family_str(),
                    s.arch.as_cli_str(),
                    s.model
                ),
                "Left/Right or Enter to switch yolo | rf-detr | rt-detr",
                false,
            ),
            (
                "Dataset Path",
                ds_display,
                "Enter to edit YOLO dataset folder/archive or COCO export path",
                true,
            ),
            (
                "Epochs",
                s.epochs.clone(),
                "Total training epochs (Enter to edit)",
                true,
            ),
            (
                "Batch Size",
                s.batch.clone(),
                "Batch size per training step (Enter to edit)",
                true,
            ),
            (
                "Device",
                s.device.clone(),
                "Left/Right to cycle (auto, xpu, xpu:0,1, cuda, cpu) or Enter to edit",
                true,
            ),
            (
                "Workers",
                workers_display,
                "Left/Right to cycle (auto, 0, 1, 2, 4, 8) or Enter to edit",
                true,
            ),
        ];

        let mut lines: Vec<Line<'static>> = Vec::new();
        lines.push(Line::from(vec![
            Span::styled(
                "Configure your training run below. Press ",
                Style::default().fg(LABEL_MUTED),
            ),
            Span::styled(
                "Enter",
                Style::default()
                    .fg(TITLE_ACCENT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" to edit a field, or ", Style::default().fg(LABEL_MUTED)),
            Span::styled(
                "F2",
                Style::default()
                    .fg(Color::LightGreen)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                " / Enter on Start to begin.",
                Style::default().fg(LABEL_MUTED),
            ),
        ]));
        lines.push(Line::from(""));

        for (idx, (label, val, hint, editable)) in fields.iter().enumerate() {
            let focused = s.focused_field == idx;
            let is_editing = focused && s.editing_field && *editable;
            let prefix = if focused { " ▶ " } else { "   " };
            let label_style = if focused {
                Style::default()
                    .fg(TITLE_ACCENT)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(LABEL_MUTED)
            };
            let val_style = if is_editing {
                Style::default()
                    .fg(GAUGE_TEXT_COVERED)
                    .bg(Color::LightYellow)
                    .add_modifier(Modifier::BOLD)
            } else if focused {
                Style::default()
                    .fg(GAUGE_TEXT_COVERED)
                    .bg(BAR_GPU_TEAL)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White).bg(BAR_TRACK_BG)
            };

            let cursor = if is_editing { "▏" } else { "" };
            let edit_tag = if is_editing { " [EDITING]" } else { "" };

            lines.push(Line::from(vec![
                Span::styled(prefix, label_style),
                Span::styled(format!("{label:<16} "), label_style),
                Span::styled(format!(" {val}{cursor} "), val_style),
                Span::styled(
                    edit_tag.to_string(),
                    Style::default()
                        .fg(Color::LightYellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(format!("   {hint}"), Style::default().fg(TABLE_BORDER_SOFT)),
            ]));
            lines.push(Line::from(""));
        }

        let start_focused = s.focused_field == 6;
        let start_prefix = if start_focused { " ▶ " } else { "   " };
        let start_btn_style = if start_focused {
            Style::default()
                .fg(GAUGE_TEXT_COVERED)
                .bg(Color::LightGreen)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default()
                .fg(Color::LightGreen)
                .bg(BAR_TRACK_BG)
                .add_modifier(Modifier::BOLD)
        };
        lines.push(Line::from(vec![
            Span::styled(
                start_prefix,
                Style::default()
                    .fg(Color::LightGreen)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("   [ Start Training (Enter / F2) ]   ", start_btn_style),
            Span::styled(
                "   Launch training and switch to live TUI",
                Style::default().fg(LABEL_MUTED),
            ),
        ]));

        if let Some(ref err) = s.error_banner {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                format!("   ⚠ {err}"),
                Style::default()
                    .fg(Color::LightRed)
                    .add_modifier(Modifier::BOLD),
            )));
        }

        frame.render_widget(Paragraph::new(lines), padded);
    }

    let footer = Line::from(vec![
        Span::styled(
            " [↑/↓/Tab] ",
            Style::default()
                .fg(GAUGE_TEXT_COVERED)
                .bg(BAR_GPU_TEAL)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Select Field  ", Style::default().fg(Color::White)),
        Span::styled(
            " [Enter] ",
            Style::default()
                .fg(GAUGE_TEXT_COVERED)
                .bg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Edit / Confirm  ", Style::default().fg(Color::White)),
        Span::styled(
            " [←/→] ",
            Style::default()
                .fg(GAUGE_TEXT_COVERED)
                .bg(BAR_STEP_BLUE)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Cycle Option  ", Style::default().fg(Color::White)),
        Span::styled(
            " [F2] ",
            Style::default()
                .fg(GAUGE_TEXT_COVERED)
                .bg(Color::LightGreen)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Start Training  ", Style::default().fg(Color::White)),
        Span::styled(
            " [Esc] ",
            Style::default()
                .fg(Color::White)
                .bg(Color::Red)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Quit ", Style::default().fg(Color::White)),
    ]);

    frame.render_widget(
        Paragraph::new(footer).alignment(Alignment::Center),
        chunks[2],
    );
}

fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let popup_vert = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_vert[1])[1]
}

fn render_help_overlay(frame: &mut Frame, area: Rect) {
    let modal = centered_rect(72, 74, area);
    frame.render_widget(Clear, modal);

    let block = Block::default()
        .title(Span::styled(
            " Keyboard Shortcuts Help (Press Esc, F1, or ? to close) ",
            Style::default()
                .fg(TITLE_ACCENT)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(TITLE_ACCENT))
        .style(Style::default().bg(OVERLAY_BG));

    let inner = block.inner(modal);
    frame.render_widget(block, modal);
    if inner.height == 0 || inner.width <= 4 {
        return;
    }

    let padded = Rect {
        x: inner.x + 2,
        y: inner.y + 1,
        width: inner.width.saturating_sub(4),
        height: inner.height.saturating_sub(1),
    };

    let rows: [(&str, &str); 11] = [
        ("Key", "Action"),
        ("Space", "Pause / Resume active training process"),
        ("c", "Request immediate checkpoint save (CHECKPOINT)"),
        (
            "Up / Down",
            "Scroll live training log (scroll to bottom to auto-follow)",
        ),
        ("F1 / ?", "Toggle this Keyboard Shortcuts Help overlay"),
        (
            "F3",
            "Toggle Run Configuration Snapshot overlay (read-only)",
        ),
        (
            "F4 / h",
            "Toggle Recent Runs History overlay (view metrics & paths)",
        ),
        ("F12", "Copy live training log lines to system clipboard"),
        ("q / Ctrl+C", "Graceful stop, print exit summary, and quit"),
        (
            "Tab / Arrows",
            "Navigate fields and cycle presets in Launch Menu",
        ),
        ("F2 / Enter", "Start training from Launch Menu"),
    ];

    let mut lines: Vec<Line<'static>> = Vec::new();
    lines.push(Line::from(Span::styled(
        "╭──────────────────┬──────────────────────────────────────────────────────────╮",
        Style::default().fg(TABLE_BORDER_SOFT),
    )));
    for (idx, (k, desc)) in rows.iter().enumerate() {
        let key_style = if idx == 0 {
            Style::default()
                .fg(TITLE_ACCENT)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default()
                .fg(Color::LightGreen)
                .add_modifier(Modifier::BOLD)
        };
        let desc_style = if idx == 0 {
            Style::default()
                .fg(TITLE_ACCENT)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        };
        lines.push(Line::from(vec![
            Span::styled("│ ", Style::default().fg(TABLE_BORDER_SOFT)),
            Span::styled(format!("{k:<16} "), key_style),
            Span::styled("│ ", Style::default().fg(TABLE_BORDER_SOFT)),
            Span::styled(format!("{desc:<56} "), desc_style),
            Span::styled("│", Style::default().fg(TABLE_BORDER_SOFT)),
        ]));
        if idx == 0 {
            lines.push(Line::from(Span::styled(
                "├──────────────────┼──────────────────────────────────────────────────────────┤",
                Style::default().fg(TABLE_BORDER_SOFT),
            )));
        }
    }
    lines.push(Line::from(Span::styled(
        "╰──────────────────┴──────────────────────────────────────────────────────────╯",
        Style::default().fg(TABLE_BORDER_SOFT),
    )));

    frame.render_widget(Paragraph::new(lines), padded);
}

fn render_config_overlay(frame: &mut Frame, area: Rect, state: &AppState) {
    let modal = centered_rect(76, 76, area);
    frame.render_widget(Clear, modal);

    let block = Block::default()
        .title(Span::styled(
            " Run Configuration Snapshot — Read-Only (Press Esc or F3 to close) ",
            Style::default()
                .fg(TITLE_ACCENT)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(TITLE_ACCENT))
        .style(Style::default().bg(OVERLAY_BG));

    let inner = block.inner(modal);
    frame.render_widget(block, modal);
    if inner.height == 0 || inner.width <= 4 {
        return;
    }

    let padded = Rect {
        x: inner.x + 2,
        y: inner.y + 1,
        width: inner.width.saturating_sub(4),
        height: inner.height.saturating_sub(1),
    };

    let (arch, task, model, dataset, epochs, batch, imgsz, device, workers, run_name, aug_summary) =
        if let Some(ref req) = state.active_request {
            let w_str = req
                .workers
                .map(|w| w.to_string())
                .unwrap_or_else(|| "auto".to_string());
            let aug = format!(
                "amp={} · freeze_backbone={} · degrees={:.0} · flipud={:.2} · scale={:.2} · mosaic={:.2} · close_mosaic={}",
                req.amp,
                req.freeze_backbone,
                req.degrees,
                req.flipud,
                req.scale,
                req.mosaic,
                req.close_mosaic
            );
            (
                req.model_family.clone(),
                req.task.clone(),
                req.model_choice.clone(),
                req.dataset_path.clone(),
                req.epochs.to_string(),
                req.batch.to_string(),
                req.imgsz.to_string(),
                req.device.clone(),
                w_str,
                req.run_name.clone(),
                aug,
            )
        } else {
            let s = &state.setup;
            let task = if s.model.to_ascii_lowercase().contains("-seg") {
                "segment".to_string()
            } else {
                "detect".to_string()
            };
            (
                s.arch.as_family_str().to_string(),
                task,
                s.model.clone(),
                s.dataset.clone(),
                s.epochs.clone(),
                s.batch.clone(),
                s.imgsz.to_string(),
                s.device.clone(),
                s.workers_input.clone(),
                s.run_name.clone(),
                "amp=true · degrees=180 · flipud=0.50 · scale=0.05 · mosaic=0.00 · close_mosaic=10"
                    .to_string(),
            )
        };

    let run_dir = state
        .results_dir
        .clone()
        .unwrap_or_else(|| format!("runs\\{task}\\{run_name}"));
    let best_wts = state
        .best_weights_path
        .clone()
        .unwrap_or_else(|| "pending".to_string());
    let onnx_file = state
        .onnx_path
        .clone()
        .unwrap_or_else(|| "pending (auto-export on completion)".to_string());
    let train_imgs = state
        .dataset_train_images
        .map(|n| format!("{n} images"))
        .unwrap_or_else(|| "auto-detected".to_string());

    let kv_rows: [(&str, String); 13] = [
        ("Architecture", arch),
        ("Task", task),
        ("Model / Weights", model),
        ("Dataset Path", dataset),
        ("Train Split Size", train_imgs),
        ("Epochs", epochs),
        ("Batch Size", batch),
        ("Image Size (imgsz)", imgsz),
        ("Device", device),
        ("DataLoader Workers", workers),
        ("Run Directory", run_dir),
        ("Best Checkpoint", best_wts),
        ("ONNX Export", onnx_file),
    ];

    let mut lines: Vec<Line<'static>> = Vec::new();
    for (k, v) in kv_rows {
        lines.push(Line::from(vec![
            Span::styled(
                format!("{k:<20} │ "),
                Style::default()
                    .fg(LABEL_MUTED)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(v, Style::default().fg(Color::White)),
        ]));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled(
            format!("{:<20} │ ", "Hyperparameters"),
            Style::default()
                .fg(LABEL_MUTED)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(aug_summary, Style::default().fg(TITLE_ACCENT)),
    ]));

    frame.render_widget(Paragraph::new(lines), padded);
}

fn render_history_overlay(frame: &mut Frame, area: Rect, state: &AppState) {
    let modal = centered_rect(82, 78, area);
    frame.render_widget(Clear, modal);

    let block = Block::default()
        .title(Span::styled(
            " Recent Training Runs (↑/↓ Select · Enter/y Copy Path · Esc/F4/h Close) ",
            Style::default()
                .fg(TITLE_ACCENT)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(TITLE_ACCENT))
        .style(Style::default().bg(OVERLAY_BG));

    let inner = block.inner(modal);
    frame.render_widget(block, modal);
    if inner.height == 0 || inner.width <= 4 {
        return;
    }

    let padded = Rect {
        x: inner.x + 2,
        y: inner.y + 1,
        width: inner.width.saturating_sub(4),
        height: inner.height.saturating_sub(1),
    };

    if state.history_entries.is_empty() {
        let msg = Paragraph::new("No previous runs found in runs/detect or runs/segment.")
            .style(Style::default().fg(LABEL_MUTED));
        frame.render_widget(msg, padded);
        return;
    }

    let split = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(6), Constraint::Length(6)])
        .split(padded);

    let mut list_lines: Vec<Line<'static>> = Vec::new();
    list_lines.push(Line::from(vec![Span::styled(
        format!(
            "   {:<20} {:<24} {:<12} {:<12} {:<10}",
            "Run Name", "Architecture / Model", "mAP50", "mAP50-95", "Updated"
        ),
        Style::default()
            .fg(TITLE_ACCENT)
            .add_modifier(Modifier::BOLD),
    )]));

    let max_rows = split[0].height.saturating_sub(1) as usize;
    let sel = state
        .history_selected
        .min(state.history_entries.len().saturating_sub(1));
    let start_idx = if sel >= max_rows && max_rows > 0 {
        sel + 1 - max_rows
    } else {
        0
    };

    for (idx, entry) in state
        .history_entries
        .iter()
        .enumerate()
        .skip(start_idx)
        .take(max_rows.max(1))
    {
        let is_sel = idx == sel;
        let marker = if is_sel { " ▶ " } else { "   " };
        let m50_s = entry
            .map50
            .map(|v| format!("{:.2}%", v * 100.0))
            .unwrap_or_else(|| "--".to_string());
        let m95_s = entry
            .map50_95
            .map(|v| format!("{:.2}%", v * 100.0))
            .unwrap_or_else(|| "--".to_string());
        let row_text = format!(
            "{marker}{:<20} {:<24} {:<12} {:<12} {:<10}",
            compact_display_path(&entry.run_name, 19),
            compact_display_path(&entry.arch_model, 23),
            m50_s,
            m95_s,
            entry.timestamp_label
        );
        let style = if is_sel {
            Style::default()
                .fg(GAUGE_TEXT_COVERED)
                .bg(BAR_GPU_TEAL)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        };
        list_lines.push(Line::from(Span::styled(row_text, style)));
    }
    frame.render_widget(Paragraph::new(list_lines), split[0]);

    if let Some(entry) = state.history_entries.get(sel) {
        let detail_lines = vec![
            Line::from(Span::styled(
                "── Selected Run Summary ──────────────────────────────────────────────────",
                Style::default().fg(TABLE_BORDER_SOFT),
            )),
            Line::from(vec![
                Span::styled("Run Directory : ", Style::default().fg(LABEL_MUTED)),
                Span::styled(entry.run_dir.clone(), Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("Best Weights  : ", Style::default().fg(LABEL_MUTED)),
                Span::styled(
                    entry
                        .best_weights
                        .clone()
                        .unwrap_or_else(|| "Not saved".to_string()),
                    Style::default().fg(Color::LightGreen),
                ),
            ]),
            Line::from(vec![
                Span::styled("ONNX Model    : ", Style::default().fg(LABEL_MUTED)),
                Span::styled(
                    entry
                        .onnx_path
                        .clone()
                        .unwrap_or_else(|| "Not exported".to_string()),
                    Style::default().fg(TITLE_ACCENT),
                ),
            ]),
            Line::from(vec![
                Span::styled("Task / Age    : ", Style::default().fg(LABEL_MUTED)),
                Span::styled(
                    format!("{} · {}", entry.task, entry.timestamp_label),
                    Style::default().fg(Color::White),
                ),
            ]),
        ];
        frame.render_widget(Paragraph::new(detail_lines), split[1]);
    }
}

fn draw_ui(frame: &mut Frame, state: &AppState) {
    if state.mode == UiMode::Setup {
        render_launcher_screen(frame, frame.area(), state);
        match state.overlay {
            OverlayMode::Help => render_help_overlay(frame, frame.area()),
            OverlayMode::Config => render_config_overlay(frame, frame.area(), state),
            OverlayMode::History => render_history_overlay(frame, frame.area(), state),
            OverlayMode::None => {}
        }
        return;
    }

    let total_h = frame.area().height;
    let top_h = 5u16;
    let rem_h = total_h.saturating_sub(top_h);
    let min_center_h = 19u16.min(rem_h.saturating_sub(6));
    let prop_center_h = ((u32::from(rem_h) * 62) / 100) as u16;
    let center_h = prop_center_h.max(min_center_h);

    let root_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(top_h),
            Constraint::Length(center_h),
            Constraint::Min(6),
        ])
        .split(frame.area());

    render_top_gpu_strip(frame, root_chunks[0], state);

    let center_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(root_chunks[1]);

    render_center_chart(frame, center_chunks[0], state);
    render_run_summary(frame, center_chunks[1], state);

    render_bottom_log_and_footer(frame, root_chunks[2], state);

    match state.overlay {
        OverlayMode::Help => render_help_overlay(frame, frame.area()),
        OverlayMode::Config => render_config_overlay(frame, frame.area(), state),
        OverlayMode::History => render_history_overlay(frame, frame.area(), state),
        OverlayMode::None => {}
    }
}

fn parse_cli_args() -> Result<Option<CliConfig>, String> {
    parse_cli_args_from(env::args().skip(1))
}

fn parse_cli_args_from<I, S>(args_iter: I) -> Result<Option<CliConfig>, String>
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

fn cycle_preset(current: &str, presets: &[&str], forward: bool) -> String {
    let cur_trimmed = current.trim();
    let idx = presets
        .iter()
        .position(|p| p.eq_ignore_ascii_case(cur_trimmed))
        .unwrap_or(0);
    let next_idx = if forward {
        (idx + 1) % presets.len()
    } else if idx == 0 {
        presets.len() - 1
    } else {
        idx - 1
    };
    presets[next_idx].to_string()
}

fn handle_setup_key(state: &Arc<Mutex<AppState>>, code: KeyCode) -> bool {
    let mut launch_req: Option<TrainingRequest> = None;
    if let Ok(mut guard) = state.lock() {
        let focused = guard.setup.focused_field;
        match code {
            KeyCode::Esc => {
                if guard.setup.editing_field {
                    guard.setup.editing_field = false;
                } else {
                    return true;
                }
            }
            KeyCode::F(2) => match guard.setup.to_training_request() {
                Ok(req) => {
                    guard.setup.editing_field = false;
                    guard.setup.error_banner = None;
                    launch_req = Some(req);
                }
                Err(err) => {
                    guard.setup.error_banner = Some(err);
                }
            },
            KeyCode::Tab | KeyCode::Down => {
                guard.setup.editing_field = false;
                guard.setup.focused_field = (focused + 1) % LAUNCHER_FIELD_COUNT;
            }
            KeyCode::BackTab | KeyCode::Up => {
                guard.setup.editing_field = false;
                guard.setup.focused_field = if focused == 0 {
                    LAUNCHER_FIELD_COUNT - 1
                } else {
                    focused - 1
                };
            }
            KeyCode::Left => match focused {
                0 => {
                    let prev_default = guard.setup.arch.default_model();
                    guard.setup.arch = guard.setup.arch.prev();
                    if guard.setup.model == prev_default {
                        guard.setup.model = guard.setup.arch.default_model().to_string();
                    }
                }
                4 if !guard.setup.editing_field => {
                    guard.setup.device = cycle_preset(&guard.setup.device, &DEVICE_PRESETS, false);
                }
                5 if !guard.setup.editing_field => {
                    guard.setup.workers_input =
                        cycle_preset(&guard.setup.workers_input, &WORKER_PRESETS, false);
                }
                _ => {}
            },
            KeyCode::Right => match focused {
                0 => {
                    let prev_default = guard.setup.arch.default_model();
                    guard.setup.arch = guard.setup.arch.next();
                    if guard.setup.model == prev_default {
                        guard.setup.model = guard.setup.arch.default_model().to_string();
                    }
                }
                4 if !guard.setup.editing_field => {
                    guard.setup.device = cycle_preset(&guard.setup.device, &DEVICE_PRESETS, true);
                }
                5 if !guard.setup.editing_field => {
                    guard.setup.workers_input =
                        cycle_preset(&guard.setup.workers_input, &WORKER_PRESETS, true);
                }
                _ => {}
            },
            KeyCode::Backspace => match focused {
                1 => {
                    guard.setup.editing_field = true;
                    guard.setup.dataset.pop();
                }
                2 => {
                    guard.setup.editing_field = true;
                    guard.setup.epochs.pop();
                }
                3 => {
                    guard.setup.editing_field = true;
                    guard.setup.batch.pop();
                }
                4 => {
                    guard.setup.editing_field = true;
                    guard.setup.device.pop();
                }
                5 => {
                    guard.setup.editing_field = true;
                    if guard.setup.workers_input.eq_ignore_ascii_case("auto") {
                        guard.setup.workers_input.clear();
                    } else {
                        guard.setup.workers_input.pop();
                    }
                }
                _ => {}
            },
            KeyCode::Char(c) => match focused {
                0 if c == ' ' => {
                    let prev_default = guard.setup.arch.default_model();
                    guard.setup.arch = guard.setup.arch.next();
                    if guard.setup.model == prev_default {
                        guard.setup.model = guard.setup.arch.default_model().to_string();
                    }
                }
                1 => {
                    guard.setup.editing_field = true;
                    guard.setup.dataset.push(c);
                }
                2 if c.is_ascii_digit() => {
                    guard.setup.editing_field = true;
                    guard.setup.epochs.push(c);
                }
                3 if c.is_ascii_digit() => {
                    guard.setup.editing_field = true;
                    guard.setup.batch.push(c);
                }
                4 => {
                    guard.setup.editing_field = true;
                    guard.setup.device.push(c);
                }
                5 => {
                    guard.setup.editing_field = true;
                    if c.is_ascii_digit() {
                        if guard.setup.workers_input.eq_ignore_ascii_case("auto") {
                            guard.setup.workers_input.clear();
                        }
                        guard.setup.workers_input.push(c);
                    } else if c.eq_ignore_ascii_case(&'a') {
                        guard.setup.workers_input = "auto".to_string();
                    }
                }
                _ => {}
            },
            KeyCode::Enter => {
                if focused == 6 {
                    match guard.setup.to_training_request() {
                        Ok(req) => {
                            guard.setup.editing_field = false;
                            guard.setup.error_banner = None;
                            launch_req = Some(req);
                        }
                        Err(err) => {
                            guard.setup.error_banner = Some(err);
                        }
                    }
                } else if focused == 0 {
                    let prev_default = guard.setup.arch.default_model();
                    guard.setup.arch = guard.setup.arch.next();
                    if guard.setup.model == prev_default {
                        guard.setup.model = guard.setup.arch.default_model().to_string();
                    }
                } else {
                    if guard.setup.editing_field
                        && focused == 5
                        && guard.setup.workers_input.trim().is_empty()
                    {
                        guard.setup.workers_input = "auto".to_string();
                    }
                    guard.setup.editing_field = !guard.setup.editing_field;
                }
            }
            _ => {}
        }
    }

    if let Some(req) = launch_req {
        launch_training_thread(Arc::clone(state), req);
    }
    false
}

fn graceful_stop_and_wait(state: &Arc<Mutex<AppState>>) {
    if let Ok(mut guard) = state.lock() {
        guard.stop_initiated = true;
        guard.status_line = "Stopping training gracefully...".to_string();
    }

    if let Some(handle) = active_training_control() {
        let _ = handle.send_command(ProcessControlCommand::Stop);
    } else {
        let _ = send_training_control("STOP\n");
    }
    let _ = tauri::async_runtime::block_on(stop_training());

    let deadline = Instant::now() + Duration::from_millis(2500);
    while Instant::now() < deadline {
        let still_running = state.lock().map(|g| g.is_running).unwrap_or(false);
        if !still_running {
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }

    // Ensure no orphaned child process remains when the TUI exits after Ctrl+C / [q].
    force_kill_active_child();
}

/// Format the BTOP++-style ASCII splash for the normal terminal buffer (`stdout`).
/// Written before entering the alternate screen so it remains in terminal scrollback.
fn format_splash_stdout() -> String {
    let mut out = String::new();
    out.push('\n');
    for (idx, art_line) in SPLASH_ASCII_LINES.iter().enumerate() {
        let ansi_rgb = match idx {
            0..=2 => "\x1b[1;38;2;116;202;218m",
            3..=5 => "\x1b[1;38;2;68;168;186m",
            _ => "\x1b[1;38;2;62;154;174m",
        };
        out.push_str(ansi_rgb);
        out.push_str(art_line);
        out.push_str("\x1b[0m\n");
    }
    out.push('\n');
    out.push_str(
        " \x1b[1;38;2;116;202;218m── LS-Trainer TUI \x1b[0m\x1b[38;2;138;150;168m· YOLO · RF-DETR · RT-DETR · ONNX ──\x1b[0m\n\n",
    );
    out
}

fn write_splash_to<W: Write>(writer: &mut W) -> std::io::Result<()> {
    let splash = format_splash_stdout();
    writer.write_all(splash.as_bytes())?;
    writer.flush()
}

/// Print the BTOP++-style splash to the normal `stdout` buffer (so it stays in scrollback)
/// and wait for `duration` (~1.5–2s) or until any key is pressed before entering the TUI.
/// Returns `Ok(true)` if the user pressed `Ctrl+C` to abort immediately.
fn run_stdout_splash(duration: Duration) -> std::io::Result<bool> {
    let mut stdout = std::io::stdout();
    write_splash_to(&mut stdout)?;

    let raw_enabled = enable_raw_mode().is_ok();
    let deadline = Instant::now() + duration;
    let poll_step = Duration::from_millis(50);
    let mut aborted = false;

    loop {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        let wait = (deadline - now).min(poll_step);
        if event::poll(wait)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    if key.modifiers.contains(KeyModifiers::CONTROL)
                        && key.code == KeyCode::Char('c')
                    {
                        aborted = true;
                    }
                    break;
                }
            }
        }
    }

    if raw_enabled {
        let _ = disable_raw_mode();
    }
    Ok(aborted)
}

fn format_exit_summary(state: &AppState) -> String {
    let best_metrics_str = match (state.best_map50, state.best_map50_95) {
        (Some(m50), Some(m95)) => format!(
            "mAP50: {:.4} ({:.1}%)  ·  mAP50-95: {:.4} ({:.1}%)",
            m50,
            m50 * 100.0,
            m95,
            m95 * 100.0
        ),
        (Some(m50), None) => format!("mAP50: {:.4} ({:.1}%)", m50, m50 * 100.0),
        (None, Some(m95)) => format!("mAP50-95: {:.4} ({:.1}%)", m95, m95 * 100.0),
        (None, None) => {
            if state.step_counter > 0 {
                format!(
                    "train_loss: {:.4} (epoch {}/{})",
                    state.latest_metrics.train_loss,
                    state.latest_metrics.epoch,
                    state.latest_metrics.total_epochs
                )
            } else {
                "N/A (no epochs completed)".to_string()
            }
        }
    };

    let best_weights_str = state.best_weights_path.as_deref().unwrap_or("Not saved");
    let onnx_str = state.onnx_path.as_deref().unwrap_or("Not exported");
    let run_dir_str = state
        .results_dir
        .as_deref()
        .unwrap_or("N/A (no run started)");
    let elapsed_secs = state.elapsed_seconds();
    let elapsed_str = if state.training_started_at.is_some() {
        let hms = format_eta_seconds(elapsed_secs.max(1.0));
        if hms == "--:--" {
            format!("{elapsed_secs:.1}s")
        } else {
            format!("{hms} ({elapsed_secs:.1}s)")
        }
    } else {
        "00:00".to_string()
    };

    let rows: [(&str, &str); 5] = [
        ("Best Metrics", &best_metrics_str),
        ("Best Weights", best_weights_str),
        ("ONNX Model", onnx_str),
        ("Run Directory", run_dir_str),
        ("Elapsed Time", &elapsed_str),
    ];

    let title = "─ LS-Trainer — Exit Summary ";
    let mut inner_width = 72usize;
    for (_label, val) in &rows {
        let row_len = 2 + 14 + 3 + val.chars().count() + 2;
        if row_len > inner_width {
            inner_width = row_len;
        }
    }

    let top_dashes = inner_width.saturating_sub(title.chars().count());
    let mut out = String::new();
    out.push_str(&format!("╭{title}{}╮\n", "─".repeat(top_dashes)));
    out.push_str(&format!("│{}│\n", " ".repeat(inner_width)));

    for (label, val) in &rows {
        let content = format!("  {label:<14} : {val}");
        let pad = inner_width.saturating_sub(content.chars().count());
        out.push_str(&format!("│{content}{}│\n", " ".repeat(pad)));
    }

    out.push_str(&format!("│{}│\n", " ".repeat(inner_width)));
    out.push_str(&format!("╰{}╯\n", "─".repeat(inner_width)));
    out
}

fn write_exit_summary_to<W: Write>(writer: &mut W, state: &AppState) -> std::io::Result<()> {
    let summary = format_exit_summary(state);
    writeln!(writer, "\x1b[38;2;116;202;218m{summary}\x1b[0m")?;
    writer.flush()
}

fn print_exit_summary(state: &Arc<Mutex<AppState>>) {
    if let Ok(mut guard) = state.lock() {
        guard.refresh_run_artifacts();
        let mut stdout = std::io::stdout();
        let _ = write_exit_summary_to(&mut stdout, &guard);
    }
}

fn handle_training_key(
    state: &Arc<Mutex<AppState>>,
    code: KeyCode,
    clipboard: &mut Option<Clipboard>,
) -> bool {
    let active_overlay = state.lock().map(|g| g.overlay).unwrap_or(OverlayMode::None);
    if active_overlay != OverlayMode::None {
        match active_overlay {
            OverlayMode::Help => {
                if matches!(code, KeyCode::Esc | KeyCode::F(1) | KeyCode::Char('?')) {
                    if let Ok(mut guard) = state.lock() {
                        guard.overlay = OverlayMode::None;
                    }
                }
            }
            OverlayMode::Config => {
                if matches!(code, KeyCode::Esc | KeyCode::F(3)) {
                    if let Ok(mut guard) = state.lock() {
                        guard.overlay = OverlayMode::None;
                    }
                }
            }
            OverlayMode::History => match code {
                KeyCode::Esc | KeyCode::F(4) | KeyCode::Char('h') | KeyCode::Char('H') => {
                    if let Ok(mut guard) = state.lock() {
                        guard.overlay = OverlayMode::None;
                    }
                }
                KeyCode::Up => {
                    if let Ok(mut guard) = state.lock() {
                        guard.history_selected = guard.history_selected.saturating_sub(1);
                    }
                }
                KeyCode::Down => {
                    if let Ok(mut guard) = state.lock() {
                        if !guard.history_entries.is_empty() {
                            let max_idx = guard.history_entries.len() - 1;
                            guard.history_selected = (guard.history_selected + 1).min(max_idx);
                        }
                    }
                }
                KeyCode::Enter | KeyCode::Char('y') | KeyCode::Char('Y') => {
                    let selected_path = state.lock().ok().and_then(|g| {
                        g.history_entries
                            .get(g.history_selected)
                            .map(|e| e.run_dir.clone())
                    });
                    if let Some(path) = selected_path {
                        let copied = copy_text_to_clipboard(clipboard, &path).is_ok();
                        if let Ok(mut guard) = state.lock() {
                            guard.status_line = if copied {
                                format!("Copied run path to clipboard: {path}")
                            } else {
                                format!("Selected run path: {path}")
                            };
                        }
                    }
                }
                _ => {}
            },
            OverlayMode::None => {}
        }
        return false;
    }

    match code {
        KeyCode::F(1) | KeyCode::Char('?') => {
            if let Ok(mut guard) = state.lock() {
                guard.overlay = OverlayMode::Help;
            }
        }
        KeyCode::F(3) => {
            if let Ok(mut guard) = state.lock() {
                guard.overlay = OverlayMode::Config;
            }
        }
        KeyCode::F(4) | KeyCode::Char('h') | KeyCode::Char('H') => {
            let entries = scan_run_history();
            if let Ok(mut guard) = state.lock() {
                guard.history_entries = entries;
                guard.history_selected = 0;
                guard.overlay = OverlayMode::History;
            }
        }
        KeyCode::Char('q') | KeyCode::Char('Q') => {
            graceful_stop_and_wait(state);
            if let Ok(mut guard) = state.lock() {
                guard.refresh_run_artifacts();
            }
            return true;
        }
        KeyCode::Char(' ') => {
            let currently_paused = state.lock().map(|g| g.is_paused).unwrap_or(false);
            let cmd = if currently_paused {
                ProcessControlCommand::Resume
            } else {
                ProcessControlCommand::Pause
            };
            let sent = if let Some(handle) = active_training_control() {
                handle.send_command(cmd).is_ok()
            } else {
                send_training_control(cmd.as_wire_line()).unwrap_or(false)
            };
            if let Ok(mut guard) = state.lock() {
                if sent || guard.is_running {
                    guard.is_paused = !currently_paused;
                    guard.status_line = if guard.is_paused {
                        "Paused".to_string()
                    } else {
                        "Training in progress...".to_string()
                    };
                }
            }
        }
        KeyCode::Char('c') | KeyCode::Char('C') => {
            let sent = if let Some(handle) = active_training_control() {
                handle
                    .send_command(ProcessControlCommand::Checkpoint)
                    .is_ok()
            } else {
                send_training_control("CHECKPOINT\n").unwrap_or(false)
            };
            if let Ok(mut guard) = state.lock() {
                if sent {
                    guard.push_log(
                        "[LS-Trainer] Requested immediate checkpoint save (CHECKPOINT)."
                            .to_string(),
                    );
                }
            }
        }
        KeyCode::Up => {
            if let Ok(mut guard) = state.lock() {
                guard.follow_logs = false;
                guard.log_scroll_offset = guard.log_scroll_offset.saturating_add(1);
            }
        }
        KeyCode::Down => {
            if let Ok(mut guard) = state.lock() {
                guard.log_scroll_offset = guard.log_scroll_offset.saturating_sub(1);
                if guard.log_scroll_offset == 0 {
                    guard.follow_logs = true;
                }
            }
        }
        KeyCode::F(12) => {
            copy_logs_with_feedback(state, clipboard);
        }
        _ => {}
    }
    false
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(cli_cfg) = (match parse_cli_args() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(2);
        }
    }) else {
        return Ok(());
    };

    let initial_req = if !cli_cfg.interactive {
        match cli_cfg.to_training_request() {
            Ok(req) => Some(req),
            Err(e) => {
                eprintln!("Error: {e}");
                std::process::exit(2);
            }
        }
    } else {
        None
    };

    // Always restore terminal raw mode and alternate screen on panic
    let orig_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        ratatui::restore();
        orig_hook(panic_info);
    }));

    let state = Arc::new(Mutex::new(AppState::new(cli_cfg)));
    spawn_hardware_polling_tasks(Arc::clone(&state));

    // Write splash to normal terminal stdout BEFORE entering alternate screen so it stays in scrollback
    let aborted_in_splash = run_stdout_splash(Duration::from_millis(1650))?;
    if aborted_in_splash {
        return Ok(());
    }

    let mut terminal = ratatui::init();

    if let Some(req) = initial_req {
        launch_training_thread(Arc::clone(&state), req);
    }

    let mut clipboard: Option<Clipboard> = None;
    let tick_rate = Duration::from_millis(80);
    let mut show_exit_summary = false;

    loop {
        if let Ok(mut guard) = state.lock() {
            if guard.is_running && !guard.is_paused {
                guard.spinner_tick = guard.spinner_tick.wrapping_add(1);
                if guard.spinner_tick % 10 == 0 {
                    guard.refresh_run_artifacts();
                }
            }
            let _ = terminal.draw(|frame| draw_ui(frame, &guard));
        }

        if event::poll(tick_rate)? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }

                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
                    graceful_stop_and_wait(&state);
                    show_exit_summary = true;
                    break;
                }

                let mode = state.lock().map(|g| g.mode).unwrap_or(UiMode::Training);

                if mode == UiMode::Setup {
                    let is_quit_q = matches!(key.code, KeyCode::Char('q') | KeyCode::Char('Q'))
                        && state
                            .lock()
                            .map(|g| !g.setup.editing_field)
                            .unwrap_or(false);
                    if is_quit_q {
                        show_exit_summary = true;
                        break;
                    }
                    if handle_setup_key(&state, key.code) {
                        break;
                    }
                    continue;
                }

                if handle_training_key(&state, key.code, &mut clipboard) {
                    show_exit_summary = true;
                    break;
                }
            }
        }
    }

    ratatui::restore();
    if show_exit_summary {
        print_exit_summary(&state);
    }
    Ok(())
}

fn copy_logs_with_feedback(state: &Arc<Mutex<AppState>>, clipboard: &mut Option<Clipboard>) {
    let (logs_text, line_count) = {
        let Ok(guard) = state.lock() else {
            return;
        };
        let count = guard.logs.len();
        let text = guard.logs.iter().cloned().collect::<Vec<_>>().join("\n");
        (text, count)
    };

    if line_count == 0 {
        if let Ok(mut guard) = state.lock() {
            guard.status_line = "No logs to copy.".to_string();
        }
        return;
    }

    match copy_text_to_clipboard(clipboard, &logs_text) {
        Ok(()) => {
            if let Ok(mut guard) = state.lock() {
                guard.status_line = format!("Copied {line_count} log lines to clipboard (F12)");
                guard.push_log(format!(
                    "[LS-Trainer] Copied {line_count} log lines to clipboard."
                ));
            }
        }
        Err(err) => {
            if let Ok(mut guard) = state.lock() {
                guard.status_line = format!("Clipboard copy failed: {err}");
                guard.push_log(format!("[LS-Trainer Warning] Clipboard copy failed: {err}"));
            }
        }
    }
}

fn copy_text_to_clipboard(clipboard: &mut Option<Clipboard>, text: &str) -> Result<(), String> {
    if clipboard.is_none() {
        *clipboard = Some(Clipboard::new().map_err(|e| e.to_string())?);
    }
    let cb = clipboard
        .as_mut()
        .ok_or_else(|| "clipboard unavailable".to_string())?;
    cb.set_text(text.to_owned()).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{backend::TestBackend, Terminal};

    #[test]
    fn epoch_bar_uses_muted_teal_and_vram_text_has_dark_contrast_when_covered() {
        let backend = TestBackend::new(120, 32);
        let mut terminal = Terminal::new(backend).expect("create test terminal");

        let mut cfg = CliConfig::default();
        cfg.interactive = false;
        cfg.dataset = "datasets/sample".to_string();
        let mut state = AppState::new(cfg);
        state.mode = UiMode::Training;
        state.is_running = true;
        state.apply_metrics(TrainMetrics {
            epoch: 3,
            total_epochs: 4,
            step: 8,
            total_steps: 10,
            train_loss: 1.2345,
            val_loss: Some(1.1100),
            map50: Some(0.812),
            map50_95: Some(0.584),
            lr: 1e-4,
            samples_per_sec: 24.5,
            eta_secs: 42.0,
            gpus: vec![GpuTelemetry {
                index: 0,
                name: "Intel Arc B580".to_string(),
                vram_used_gb: 12.0,
                vram_total_gb: 16.0,
                util_percent: 75.0,
            }],
        });

        terminal
            .draw(|frame| draw_ui(frame, &state))
            .expect("draw_ui succeeds");

        let buf = terminal.backend().buffer();

        // 1. Check VRAM gauge row (row y=1 inside top GPU strip):
        // With 75% fill across width ~116, the centered label "VRAM 12.00 / 16.00 GiB (75%)"
        // including "(75%)" is covered by the GPU bar and must use GAUGE_TEXT_COVERED (dark/black)
        // on BAR_GPU_TEAL background.
        let mut found_vram_pct_covered = false;
        for x in 1..118 {
            let cell = &buf[(x, 1)];
            if cell.symbol() == "%" && cell.bg == BAR_GPU_TEAL {
                assert_eq!(
                    cell.fg, GAUGE_TEXT_COVERED,
                    "VRAM percentage covered by GPU bar must be drawn in dark/black text"
                );
                found_vram_pct_covered = true;
            }
        }
        assert!(
            found_vram_pct_covered,
            "Expected VRAM percentage character '%' covered by GPU bar in row 1"
        );

        // 2. Check partial coverage (e.g. 52% where part of the label is covered and part is on track):
        state.gpus[0].vram_used_gb = 8.32;
        state.gpus[0].vram_total_gb = 16.0;
        terminal
            .draw(|frame| draw_ui(frame, &state))
            .expect("draw_ui partial vram succeeds");
        let buf = terminal.backend().buffer();
        let mut saw_covered_label_char = false;
        let mut saw_uncovered_label_char = false;
        for x in 2..118 {
            let cell = &buf[(x, 1)];
            if cell.symbol() != " " {
                if cell.bg == BAR_GPU_TEAL {
                    assert_eq!(cell.fg, GAUGE_TEXT_COVERED);
                    saw_covered_label_char = true;
                } else if cell.bg == BAR_TRACK_BG {
                    assert_eq!(cell.fg, GAUGE_TEXT_UNCOVERED);
                    saw_uncovered_label_char = true;
                }
            }
        }
        assert!(saw_covered_label_char && saw_uncovered_label_char);

        // 3. Confirm no cell in the entire buffer uses loud pink (LightMagenta) for the Epoch bar,
        // and that BAR_EPOCH_TEAL is used instead.
        let mut saw_epoch_teal = false;
        for y in 0..32 {
            for x in 0..120 {
                let cell = &buf[(x, y)];
                assert_ne!(
                    cell.bg,
                    Color::LightMagenta,
                    "Epoch bar must not use loud LightMagenta background"
                );
                if cell.bg == BAR_EPOCH_TEAL {
                    saw_epoch_teal = true;
                }
            }
        }
        assert!(
            saw_epoch_teal,
            "Expected Epoch progress bar to use BAR_EPOCH_TEAL"
        );
    }

    #[test]
    fn log_metric_tables_soften_heavy_borders() {
        let heavy_line = "┏━━━━━━━━━━━━━━┳━━━━━━━┓";
        let styled = parse_ansi_or_styled_log_line(heavy_line);
        let rendered: String = styled.spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(rendered, "╭──────────────┬───────╮");
        for span in &styled.spans {
            assert_eq!(span.style.fg, Some(TABLE_BORDER_SOFT));
        }
    }

    #[test]
    fn no_flags_opens_fullscreen_launcher_with_all_fields() {
        let empty_args: Vec<&str> = Vec::new();
        let cfg = parse_cli_args_from(empty_args)
            .expect("parse empty args")
            .expect("some config");
        assert!(
            cfg.interactive,
            "No flags must open the interactive launcher"
        );

        let state = Arc::new(Mutex::new(AppState::new(cfg)));
        {
            let guard = state.lock().unwrap();
            assert_eq!(guard.mode, UiMode::Setup);

            let backend = TestBackend::new(120, 32);
            let mut terminal = Terminal::new(backend).expect("create test terminal");
            terminal
                .draw(|frame| draw_ui(frame, &guard))
                .expect("draw launcher");

            let buf = terminal.backend().buffer();
            let mut screen_text = String::new();
            for y in 0..32 {
                for x in 0..120 {
                    screen_text.push_str(buf[(x, y)].symbol());
                }
                screen_text.push('\n');
            }
            assert!(screen_text.contains("LS-Trainer TUI — Launch Menu"));
            assert!(screen_text.contains("Architecture"));
            assert!(screen_text.contains("Dataset Path"));
            assert!(screen_text.contains("Epochs"));
            assert!(screen_text.contains("Batch Size"));
            assert!(screen_text.contains("Device"));
            assert!(screen_text.contains("Workers"));
            assert!(screen_text.contains("[ Start Training (Enter / F2) ]"));
        }

        // Navigate and edit fields in the launcher:
        // Field 0: Architecture -> Right arrow switches YOLO -> RF-DETR
        handle_setup_key(&state, KeyCode::Right);
        assert_eq!(state.lock().unwrap().setup.arch, ArchChoice::RfDetr);
        handle_setup_key(&state, KeyCode::Left);
        assert_eq!(state.lock().unwrap().setup.arch, ArchChoice::Yolo);

        // Down to Field 1 (Dataset Path), Enter to edit, set dataset path, Enter to confirm
        handle_setup_key(&state, KeyCode::Down);
        assert_eq!(state.lock().unwrap().setup.focused_field, 1);
        handle_setup_key(&state, KeyCode::Enter);
        assert!(state.lock().unwrap().setup.editing_field);
        state.lock().unwrap().setup.dataset =
            "C:\\Users\\Terry.minett\\Downloads\\PAK-BB_yolo.tar".to_string();
        handle_setup_key(&state, KeyCode::Enter);
        assert!(!state.lock().unwrap().setup.editing_field);

        // Down to Field 2 (Epochs), edit to "1"
        handle_setup_key(&state, KeyCode::Down);
        assert_eq!(state.lock().unwrap().setup.focused_field, 2);
        handle_setup_key(&state, KeyCode::Backspace);
        handle_setup_key(&state, KeyCode::Backspace);
        handle_setup_key(&state, KeyCode::Char('1'));
        handle_setup_key(&state, KeyCode::Enter);
        assert_eq!(state.lock().unwrap().setup.epochs, "1");

        // Down to Field 3 (Batch Size), Field 4 (Device), Field 5 (Workers)
        handle_setup_key(&state, KeyCode::Down); // 3: Batch
        handle_setup_key(&state, KeyCode::Down); // 4: Device
        handle_setup_key(&state, KeyCode::Right); // auto -> xpu
        assert_eq!(state.lock().unwrap().setup.device, "xpu");

        handle_setup_key(&state, KeyCode::Down); // 5: Workers
        handle_setup_key(&state, KeyCode::Right); // auto -> 0
        assert_eq!(state.lock().unwrap().setup.workers_input, "0");

        let req = state
            .lock()
            .unwrap()
            .setup
            .to_training_request()
            .expect("valid training request from launcher");
        assert_eq!(req.model_family, "YOLO");
        assert_eq!(req.epochs, 1);
        assert_eq!(req.batch, 4);
        assert_eq!(req.device, "xpu");
        assert_eq!(req.workers, Some(0));
    }

    #[test]
    fn with_flags_runs_immediately_without_launcher() {
        let args = vec![
            "--arch",
            "yolo",
            "--dataset",
            "C:\\Users\\Terry.minett\\Downloads\\PAK-BB_yolo.tar",
            "--epochs",
            "1",
            "--batch",
            "4",
            "--device",
            "xpu",
            "--workers",
            "0",
        ];
        let cfg = parse_cli_args_from(args)
            .expect("parse flags")
            .expect("some config");
        assert!(
            !cfg.interactive,
            "Providing training flags must start immediately without launcher"
        );
        let state = AppState::new(cfg.clone());
        assert_eq!(state.mode, UiMode::Training);
        let req = cfg.to_training_request().expect("valid request");
        assert_eq!(req.epochs, 1);
        assert_eq!(req.workers, Some(0));
    }

    fn write_tiny_bmp_64x64(path: &std::path::Path) {
        let width: u32 = 64;
        let height: u32 = 64;
        let row_bytes = width * 3;
        let pixel_bytes = row_bytes * height;
        let file_size = 54 + pixel_bytes;

        let mut data = Vec::with_capacity(file_size as usize);
        // BMP Header (14 bytes)
        data.extend_from_slice(b"BM");
        data.extend_from_slice(&file_size.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&54u32.to_le_bytes());
        // DIB Header (BITMAPINFOHEADER, 40 bytes)
        data.extend_from_slice(&40u32.to_le_bytes());
        data.extend_from_slice(&(width as i32).to_le_bytes());
        data.extend_from_slice(&(height as i32).to_le_bytes());
        data.extend_from_slice(&1u16.to_le_bytes()); // planes
        data.extend_from_slice(&24u16.to_le_bytes()); // bpp
        data.extend_from_slice(&0u32.to_le_bytes()); // compression
        data.extend_from_slice(&pixel_bytes.to_le_bytes());
        data.extend_from_slice(&2835u32.to_le_bytes());
        data.extend_from_slice(&2835u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());

        for y in 0..height {
            for x in 0..width {
                let b = ((x * 4) & 0xFF) as u8;
                let g = ((y * 4) & 0xFF) as u8;
                let r = 128u8;
                data.extend_from_slice(&[b, g, r]);
            }
        }
        std::fs::write(path, data).expect("write tiny bmp");
    }

    #[test]
    fn splash_writes_to_normal_scrollback_and_persists_after_quit() {
        let mut normal_scrollback: Vec<u8> = Vec::new();
        write_splash_to(&mut normal_scrollback).expect("write splash to normal stdout buffer");

        let splash_out = String::from_utf8(normal_scrollback.clone()).expect("utf8 splash");
        // Verify each of the 8 Big Money-ne ASCII art lines is present left-aligned in normal stdout
        for (idx, expected_line) in SPLASH_ASCII_LINES.iter().enumerate() {
            assert!(
                splash_out.contains(expected_line),
                "Splash ASCII line {idx} must be written to normal stdout scrollback"
            );
        }
        // Must not emit alternate screen or clear-screen escape codes on the normal buffer
        assert!(!splash_out.contains("\x1b[2J"));
        assert!(!splash_out.contains("\x1b[3J"));
        assert!(!splash_out.contains("\x1b[?1049h"));

        // Transition to TUI launcher on alternate screen
        let backend = TestBackend::new(100, 24);
        let mut terminal = Terminal::new(backend).expect("create test terminal");
        let state = AppState::new(CliConfig::default());
        terminal
            .draw(|frame| draw_ui(frame, &state))
            .expect("draw launcher after splash");
        let buf_after = terminal.backend().buffer();
        let mut text_after = String::new();
        for y in 0..24 {
            for x in 0..100 {
                text_after.push_str(buf_after[(x, y)].symbol());
            }
            text_after.push('\n');
        }
        assert!(text_after.contains("LS-Trainer TUI — Launch Menu"));

        // On quit (q or Ctrl+C), exit summary is written to normal stdout without clearing scrollback
        write_exit_summary_to(&mut normal_scrollback, &state)
            .expect("write exit summary to normal stdout buffer");
        let full_scrollback =
            String::from_utf8(normal_scrollback).expect("utf8 full normal scrollback");
        assert!(!full_scrollback.contains("\x1b[2J"));
        assert!(!full_scrollback.contains("\x1b[3J"));
        for expected_line in SPLASH_ASCII_LINES {
            assert!(
                full_scrollback.contains(expected_line),
                "Splash art must still be present in normal terminal scrollback after quit"
            );
        }
        assert!(
            full_scrollback.contains("╭─ LS-Trainer — Exit Summary"),
            "Exit summary must also be present in normal terminal scrollback after quit"
        );
    }

    static REAL_RUN_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn short_real_run_from_launcher_verifies_tui_rendering() {
        let _run_guard = REAL_RUN_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let tmp_ds = std::env::temp_dir().join("ls_trainer_tui_tiny_ds");
        let _ = std::fs::remove_dir_all(&tmp_ds);
        if let Ok(entries) = std::fs::read_dir("runs/detect") {
            for entry in entries.flatten() {
                if entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("tui-short-verify")
                {
                    let _ = std::fs::remove_dir_all(entry.path());
                }
            }
        }
        let img_dir = tmp_ds.join("images").join("train");
        let lbl_dir = tmp_ds.join("labels").join("train");
        std::fs::create_dir_all(&img_dir).expect("create train images dir");
        std::fs::create_dir_all(&lbl_dir).expect("create train labels dir");

        for i in 0..2 {
            write_tiny_bmp_64x64(&img_dir.join(format!("sample_{i}.bmp")));
            std::fs::write(
                lbl_dir.join(format!("sample_{i}.txt")),
                "0 0.5 0.5 0.4 0.4\n",
            )
            .expect("write label");
        }
        let yaml_content = format!(
            "path: {}\ntrain: images/train\nval: images/train\nnames:\n  0: part\n",
            tmp_ds.to_string_lossy().replace('\\', "/")
        );
        std::fs::write(tmp_ds.join("data.yaml"), yaml_content).expect("write data.yaml");

        // 0. Print splash to normal terminal stdout buffer before starting TUI
        let mut normal_stdout_buf: Vec<u8> = Vec::new();
        write_splash_to(&mut normal_stdout_buf).expect("write splash to stdout");

        let backend = TestBackend::new(120, 32);
        let mut terminal = Terminal::new(backend).expect("create test terminal");

        // 1. Launch with no flags -> transitions from splash to Setup (Launcher) mode
        let empty_args: Vec<&str> = Vec::new();
        let cfg = parse_cli_args_from(empty_args)
            .expect("parse empty args")
            .expect("some config");
        assert!(cfg.interactive);

        let state = Arc::new(Mutex::new(AppState::new(cfg)));
        assert_eq!(state.lock().unwrap().mode, UiMode::Setup);

        // Configure 1-epoch fast run in launcher and press F2 to start
        {
            let mut guard = state.lock().unwrap();
            guard.setup.arch = ArchChoice::Yolo;
            guard.setup.model = "yolo11n.pt".to_string();
            guard.setup.dataset = tmp_ds.to_string_lossy().to_string();
            guard.setup.epochs = "1".to_string();
            guard.setup.batch = "2".to_string();
            guard.setup.imgsz = 64;
            guard.setup.device = "cpu".to_string();
            guard.setup.workers_input = "0".to_string();
            guard.setup.run_name = "tui-short-verify".to_string();
        }

        // Press F2 in launcher -> transitions to UiMode::Training and spawns run_training + ONNX export
        let should_quit = handle_setup_key(&state, KeyCode::F(2));
        assert!(!should_quit);
        assert_eq!(state.lock().unwrap().mode, UiMode::Training);
        assert!(state.lock().unwrap().is_running);

        let deadline = Instant::now() + Duration::from_secs(60);
        let mut saw_live_metrics = false;
        while Instant::now() < deadline {
            let done = {
                let mut guard = state.lock().unwrap();
                if guard.is_running {
                    guard.spinner_tick = guard.spinner_tick.wrapping_add(1);
                }
                terminal
                    .draw(|frame| draw_ui(frame, &guard))
                    .expect("draw_ui during live run");
                if guard.step_counter > 0 {
                    saw_live_metrics = true;
                }
                !guard.is_running && guard.exit_code.is_some()
            };
            if done {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }

        {
            let guard = state.lock().unwrap();
            if guard.exit_code != Some(0) {
                for l in &guard.logs {
                    println!("LOG: {l}");
                }
                force_kill_active_child();
            }
            assert_eq!(
                guard.exit_code,
                Some(0),
                "Expected short training run + ONNX export to exit with code 0; status={}",
                guard.status_line
            );
            assert!(
                saw_live_metrics,
                "Expected live @@LS_METRICS@@ updates during run"
            );
            assert_eq!(guard.latest_metrics.epoch, 1);

            // Verify both weights (best.pt) and ONNX (best.onnx in run dir) were produced
            let run_dir = guard.results_dir.as_ref().expect("results_dir recorded");
            let best_pt = guard
                .best_weights_path
                .as_ref()
                .expect("best_weights_path recorded");
            let best_onnx = guard.onnx_path.as_ref().expect("onnx_path recorded");
            assert!(
                Path::new(best_pt).is_file(),
                "Expected best.pt weights file to exist at {best_pt}"
            );
            assert!(
                Path::new(best_onnx).is_file(),
                "Expected exported ONNX file to exist at {best_onnx}"
            );
            assert!(
                Path::new(run_dir).join("best.onnx").is_file(),
                "Expected best.onnx to be saved in run directory {run_dir}"
            );
            assert!(
                guard
                    .logs
                    .iter()
                    .any(|l| l.contains("Training complete!") && l.contains(best_onnx)),
                "Expected training complete log message to include ONNX path"
            );

            terminal
                .draw(|frame| draw_ui(frame, &guard))
                .expect("final draw_ui");
            let buf = terminal.backend().buffer();

            // Confirm epoch bar uses BAR_EPOCH_TEAL and never LightMagenta, and VRAM text is readable
            let mut saw_epoch_teal = false;
            let mut saw_vram_bar = false;
            for y in 0..32 {
                for x in 0..120 {
                    let cell = &buf[(x, y)];
                    assert_ne!(cell.bg, Color::LightMagenta);
                    if cell.bg == BAR_EPOCH_TEAL {
                        saw_epoch_teal = true;
                    }
                    if y == 1 && cell.bg == BAR_GPU_TEAL && cell.symbol() != " " {
                        assert_eq!(cell.fg, GAUGE_TEXT_COVERED);
                        saw_vram_bar = true;
                    }
                }
            }
            assert!(saw_epoch_teal);
            assert!(
                !guard.gpus.is_empty() || saw_vram_bar,
                "Expected GPU telemetry during training run"
            );
        }

        // Press 'q' to quit -> verify exit summary contains border, metrics, weights path, ONNX path, run dir, and elapsed time
        let mut cb = None;
        let quit_triggered = handle_training_key(&state, KeyCode::Char('q'), &mut cb);
        assert!(quit_triggered, "Pressing q must quit training TUI");

        let guard = state.lock().unwrap();
        write_exit_summary_to(&mut normal_stdout_buf, &guard)
            .expect("write exit summary to normal stdout");
        let scrollback = String::from_utf8(normal_stdout_buf).expect("valid utf8 scrollback");
        assert!(scrollback.contains(SPLASH_ASCII_LINES[0]));
        assert!(scrollback.contains("╭─ LS-Trainer — Exit Summary"));

        let summary = format_exit_summary(&guard);
        println!("EXIT SUMMARY:\n{summary}");
        assert!(summary.contains("╭─ LS-Trainer — Exit Summary"));
        assert!(summary.contains("╰"));
        assert!(summary.contains("mAP50:"));
        assert!(summary.contains("mAP50-95:"));
        assert!(summary.contains(guard.best_weights_path.as_deref().unwrap()));
        assert!(summary.contains(guard.onnx_path.as_deref().unwrap()));
        assert!(summary.contains(guard.results_dir.as_deref().unwrap()));
        assert!(summary.contains("Elapsed Time"));

        let _ = std::fs::remove_dir_all(&tmp_ds);
        if let Ok(entries) = std::fs::read_dir("runs/detect") {
            for entry in entries.flatten() {
                if entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("tui-short-verify")
                {
                    let _ = std::fs::remove_dir_all(entry.path());
                }
            }
        }
    }

    fn buffer_to_string(buf: &ratatui::buffer::Buffer) -> String {
        let area = buf.area;
        let mut out = String::new();
        for y in area.top()..area.bottom() {
            for x in area.left()..area.right() {
                out.push_str(buf[(x, y)].symbol());
            }
            out.push('\n');
        }
        out
    }

    #[test]
    fn dual_xpu_ui_paths_overall_progress_new_best_health_and_overlays() {
        let backend = TestBackend::new(140, 34);
        let mut terminal = Terminal::new(backend).expect("create test terminal");

        let args = vec![
            "--arch",
            "rf-detr",
            "--dataset",
            "datasets/coco_parts",
            "--epochs",
            "4",
            "--batch",
            "4",
            "--device",
            "xpu:0,1",
            "--workers",
            "0",
            "--run-name",
            "rfdetr-dual-xpu-test",
        ];
        let cfg = parse_cli_args_from(args)
            .expect("parse flags")
            .expect("some config");
        let req = cfg.to_training_request().expect("valid request");

        let state = Arc::new(Mutex::new(AppState::new(cfg)));
        {
            let mut guard = state.lock().unwrap();
            guard.mode = UiMode::Training;
            guard.active_request = Some(req);
            guard.is_running = true;
            guard.dataset_train_images = Some(80);
            guard.results_dir =
                Some(r"c:\Scripts\LS-Trainer\runs\detect\rfdetr-dual-xpu-test".to_string());
        }

        // 1. Verify Healthy status initially & dual-XPU telemetry + Overall progress + New-best indicator
        {
            let mut guard = state.lock().unwrap();
            guard.apply_metrics(TrainMetrics {
                epoch: 2,
                total_epochs: 4,
                step: 5,
                total_steps: 10,
                train_loss: 1.85,
                val_loss: Some(1.62),
                map50: Some(0.645),
                map50_95: Some(0.412),
                lr: 1e-4,
                samples_per_sec: 18.4,
                eta_secs: 35.0,
                gpus: vec![
                    GpuTelemetry {
                        index: 0,
                        name: "Intel Arc B580".to_string(),
                        vram_used_gb: 9.2,
                        vram_total_gb: 12.0,
                        util_percent: 82.0,
                    },
                    GpuTelemetry {
                        index: 1,
                        name: "Intel Arc B580".to_string(),
                        vram_used_gb: 8.9,
                        vram_total_gb: 12.0,
                        util_percent: 79.0,
                    },
                ],
            });
            guard.best_weights_path = Some(
                r"c:\Scripts\LS-Trainer\runs\detect\rfdetr-dual-xpu-test\checkpoint_best_total.pth"
                    .to_string(),
            );
            guard.onnx_path = Some(
                r"c:\Scripts\LS-Trainer\runs\detect\rfdetr-dual-xpu-test\best.onnx".to_string(),
            );

            assert!(
                guard.is_new_best_active(),
                "New-best indicator must fire when mAP/loss improves"
            );
            assert_eq!(guard.health_level, HealthLevel::Healthy);

            terminal
                .draw(|frame| draw_ui(frame, &guard))
                .expect("draw dual-xpu training ui");
            let screen = buffer_to_string(terminal.backend().buffer());

            // Dual-XPU cards rendered side-by-side
            assert!(screen.contains("GPU 0: Intel Arc B580"));
            assert!(screen.contains("GPU 1: Intel Arc B580"));
            // Overall progress bar with combined epoch + step + images
            assert!(
                screen.contains("Overall 38% · Ep 2/4 · Step 5/10 · 120/320 imgs"),
                "Expected Overall progress indicator in screen:\n{screen}"
            );
            // New-best indicator badge next to Best mAP
            assert!(
                screen.contains("★ New best"),
                "Expected New best marker in Run Summary:\n{screen}"
            );
            // Model output paths in Run Summary during the run
            assert!(screen.contains("Run Dir"));
            assert!(screen.contains("Best Wts"));
            assert!(screen.contains("ONNX"));
            assert!(screen.contains("checkpoint_best_total.pth"));
            assert!(screen.contains("best.onnx"));
            // Healthy status line
            assert!(screen.contains("● Healthy"));
        }

        // 2. Verify noisy warnings (triton, sync_dist) update Status Line to Warning and are suppressed in live log
        {
            let mut guard = state.lock().unwrap();
            let log_len_before = guard.logs.len();
            guard.push_log(
                "W1115 16:00:00.000000 1234 torch/utils/flop_counter.py:45] triton not found; flop counting will not work"
                    .to_string(),
            );
            guard.push_log(
                "PossibleUserWarning: It is recommended to use `self.log('val/loss', ..., sync_dist=True)`"
                    .to_string(),
            );
            assert_eq!(
                guard.logs.len(),
                log_len_before,
                "Noisy triton and sync_dist warnings must be suppressed from flooding live logs"
            );
            assert_eq!(guard.health_level, HealthLevel::Warning);
            assert_eq!(guard.suppressed_warning_count, 2);

            terminal
                .draw(|frame| draw_ui(frame, &guard))
                .expect("draw warning state");
            let screen = buffer_to_string(terminal.backend().buffer());
            assert!(screen.contains("▲ Warning"));
            assert!(screen.contains("triton, sync_dist"));
            assert!(screen.contains("2 suppressed in log"));
        }

        // 3. Verify F1 / ? Keyboard Shortcuts Help overlay and F3 Config Snapshot overlay and F4 History overlay
        let mut cb = None;
        handle_training_key(&state, KeyCode::Char('?'), &mut cb);
        assert_eq!(state.lock().unwrap().overlay, OverlayMode::Help);
        {
            let guard = state.lock().unwrap();
            terminal
                .draw(|frame| draw_ui(frame, &guard))
                .expect("draw help overlay");
            let screen = buffer_to_string(terminal.backend().buffer());
            assert!(screen.contains("Keyboard Shortcuts Help"));
            assert!(screen.contains("Toggle Run Configuration Snapshot overlay"));
        }
        handle_training_key(&state, KeyCode::Esc, &mut cb);
        assert_eq!(state.lock().unwrap().overlay, OverlayMode::None);

        handle_training_key(&state, KeyCode::F(3), &mut cb);
        assert_eq!(state.lock().unwrap().overlay, OverlayMode::Config);
        {
            let guard = state.lock().unwrap();
            terminal
                .draw(|frame| draw_ui(frame, &guard))
                .expect("draw config overlay");
            let screen = buffer_to_string(terminal.backend().buffer());
            assert!(screen.contains("Run Configuration Snapshot"));
            assert!(screen.contains("RF-DETR"));
            assert!(screen.contains("xpu:0,1"));
            assert!(screen.contains("datasets/coco_parts"));
        }
        handle_training_key(&state, KeyCode::F(3), &mut cb);
        assert_eq!(state.lock().unwrap().overlay, OverlayMode::None);

        handle_training_key(&state, KeyCode::F(4), &mut cb);
        assert_eq!(state.lock().unwrap().overlay, OverlayMode::History);
        {
            let guard = state.lock().unwrap();
            terminal
                .draw(|frame| draw_ui(frame, &guard))
                .expect("draw history overlay");
            let screen = buffer_to_string(terminal.backend().buffer());
            assert!(screen.contains("Recent Training Runs"));
        }
        handle_training_key(&state, KeyCode::Esc, &mut cb);
        assert_eq!(state.lock().unwrap().overlay, OverlayMode::None);
    }

    #[test]
    fn short_dual_xpu_rfdetr_real_run_verifies_live_ui_and_onnx() {
        let _run_guard = REAL_RUN_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let tmp_coco = std::env::temp_dir().join("ls_trainer_tui_rfdetr_coco_ds");
        let _ = std::fs::remove_dir_all(&tmp_coco);
        let run_name = "tui-rfdetr-dual-xpu-verify";
        let _ = std::fs::remove_dir_all(PathBuf::from("runs/detect").join(run_name));

        for split in ["train", "valid", "test"] {
            let split_dir = tmp_coco.join(split);
            std::fs::create_dir_all(&split_dir).expect("create coco split dir");
            for i in 0..2 {
                write_tiny_bmp_64x64(&split_dir.join(format!("img_{i}.bmp")));
            }
            let coco_json = serde_json::json!({
                "info": { "description": "tiny rfdetr verify", "version": "1.0" },
                "licenses": [],
                "categories": [
                    { "id": 0, "name": "part", "supercategory": "defect" }
                ],
                "images": [
                    { "id": 1, "file_name": "img_0.bmp", "width": 64, "height": 64 },
                    { "id": 2, "file_name": "img_1.bmp", "width": 64, "height": 64 }
                ],
                "annotations": [
                    { "id": 1, "image_id": 1, "category_id": 0, "bbox": [12.0, 12.0, 32.0, 32.0], "area": 1024.0, "iscrowd": 0 },
                    { "id": 2, "image_id": 2, "category_id": 0, "bbox": [10.0, 10.0, 36.0, 36.0], "area": 1296.0, "iscrowd": 0 }
                ]
            });
            std::fs::write(
                split_dir.join("_annotations.coco.json"),
                serde_json::to_string_pretty(&coco_json).unwrap(),
            )
            .expect("write _annotations.coco.json");
        }

        let args = vec![
            "--arch".to_string(),
            "rf-detr".to_string(),
            "--dataset".to_string(),
            tmp_coco.to_string_lossy().to_string(),
            "--epochs".to_string(),
            "1".to_string(),
            "--batch".to_string(),
            "2".to_string(),
            "--imgsz".to_string(),
            "224".to_string(),
            "--device".to_string(),
            "xpu:0,1".to_string(),
            "--workers".to_string(),
            "0".to_string(),
            "--run-name".to_string(),
            run_name.to_string(),
        ];
        let cfg = parse_cli_args_from(args)
            .expect("parse rf-detr flags")
            .expect("some config");
        let req = cfg.to_training_request().expect("valid rf-detr request");

        let state = Arc::new(Mutex::new(AppState::new(cfg)));
        let backend = TestBackend::new(140, 34);
        let mut terminal = Terminal::new(backend).expect("create test terminal");

        launch_training_thread(Arc::clone(&state), req);

        // Verify Run Dir path is immediately visible in UI while run is starting
        {
            let guard = state.lock().unwrap();
            assert!(guard.results_dir.is_some());
            assert_eq!(guard.dataset_train_images, Some(2));
            terminal
                .draw(|frame| draw_ui(frame, &guard))
                .expect("initial draw");
            let screen = buffer_to_string(terminal.backend().buffer());
            assert!(screen.contains("Run Dir"));
            assert!(screen.contains(run_name));
            assert!(screen.contains("Overall"));
        }

        // Toggle F1 (?) and F3 (Config Snapshot) during live run
        let mut cb = None;
        handle_training_key(&state, KeyCode::F(1), &mut cb);
        assert_eq!(state.lock().unwrap().overlay, OverlayMode::Help);
        handle_training_key(&state, KeyCode::F(1), &mut cb);
        assert_eq!(state.lock().unwrap().overlay, OverlayMode::None);

        handle_training_key(&state, KeyCode::F(3), &mut cb);
        assert_eq!(state.lock().unwrap().overlay, OverlayMode::Config);
        handle_training_key(&state, KeyCode::Esc, &mut cb);
        assert_eq!(state.lock().unwrap().overlay, OverlayMode::None);

        let deadline = Instant::now() + Duration::from_secs(180);
        let mut saw_live_weights_during_run = false;
        let mut saw_new_best_during_run = false;
        while Instant::now() < deadline {
            let done = {
                let mut guard = state.lock().unwrap();
                if guard.is_running {
                    guard.spinner_tick = guard.spinner_tick.wrapping_add(1);
                    guard.refresh_run_artifacts();
                    if guard.best_weights_path.is_some() {
                        saw_live_weights_during_run = true;
                    }
                    if guard.is_new_best_active() {
                        saw_new_best_during_run = true;
                    }
                }
                terminal
                    .draw(|frame| draw_ui(frame, &guard))
                    .expect("draw_ui during rf-detr run");
                !guard.is_running && guard.exit_code.is_some()
            };
            if done {
                break;
            }
            std::thread::sleep(Duration::from_millis(120));
        }

        {
            let guard = state.lock().unwrap();
            if guard.exit_code != Some(0) {
                for l in &guard.logs {
                    println!("RFDETR LOG: {l}");
                }
                force_kill_active_child();
            }
            assert_eq!(
                guard.exit_code,
                Some(0),
                "Expected short RF-DETR run + ONNX export to succeed; status={}",
                guard.status_line
            );
            assert!(
                saw_live_weights_during_run || guard.best_weights_path.is_some(),
                "Expected best weights path to be populated"
            );
            assert!(
                saw_new_best_during_run || guard.new_best_until.is_some(),
                "Expected new-best indicator to fire during RF-DETR run"
            );
            let best_ckpt = guard
                .best_weights_path
                .as_ref()
                .expect("RF-DETR best checkpoint recorded");
            let best_onnx = guard
                .onnx_path
                .as_ref()
                .expect("RF-DETR ONNX path recorded");
            assert!(
                Path::new(best_ckpt).is_file(),
                "RF-DETR checkpoint file must exist at {best_ckpt}"
            );
            assert!(
                Path::new(best_onnx).is_file(),
                "RF-DETR exported ONNX file must exist at {best_onnx}"
            );

            terminal
                .draw(|frame| draw_ui(frame, &guard))
                .expect("final draw_ui after rf-detr run");
            let final_screen = buffer_to_string(terminal.backend().buffer());
            assert!(final_screen.contains("Overall 100%"));
            assert!(final_screen.contains("best.onnx"));
            assert!(
                final_screen.contains("● Healthy") || final_screen.contains("▲ Warning"),
                "Status line must reflect Healthy or Warning state"
            );
        }

        let _ = std::fs::remove_dir_all(&tmp_coco);
        let _ = std::fs::remove_dir_all(PathBuf::from("runs/detect").join(run_name));
    }
}
