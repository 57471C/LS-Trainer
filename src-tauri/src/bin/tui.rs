//! `ls-trainer-tui` — Cross-platform `btop`-style terminal UI for LS-Trainer.

use std::collections::VecDeque;
use std::env;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use app_lib as ls_trainer_lib;
use arboard::Clipboard;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ls_trainer_lib::{
    GpuTelemetry, ProcessControlCommand, TrainMetrics, TrainingEventSink, TrainingRequest,
    active_training_control, force_kill_active_child, probe_local_gpu_info, run_training,
    send_training_control, stop_training,
};
use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{Axis, Block, BorderType, Borders, Chart, Dataset, GraphType, Paragraph, Sparkline},
};
use sysinfo::System;

const MAX_LOG_LINES: usize = 500;
const MAX_HISTORY_POINTS: usize = 300;
const MAX_SPARKLINE_SAMPLES: usize = 60;

const BAR_TRACK_BG: Color = Color::Rgb(26, 32, 44);
const BAR_GPU_TEAL: Color = Color::Rgb(68, 168, 186);
const BAR_EPOCH_TEAL: Color = Color::Rgb(62, 154, 174);
const BAR_STEP_BLUE: Color = Color::Rgb(72, 128, 184);
const BAR_RAM_SLATE: Color = Color::Rgb(92, 148, 188);
const GAUGE_TEXT_COVERED: Color = Color::Rgb(16, 20, 26);
const GAUGE_TEXT_UNCOVERED: Color = Color::Rgb(228, 234, 242);
const BORDER_SOFT: Color = Color::Rgb(64, 76, 94);
const TABLE_BORDER_SOFT: Color = Color::Rgb(74, 86, 104);
const TITLE_ACCENT: Color = Color::Rgb(116, 202, 218);
const LABEL_MUTED: Color = Color::Rgb(138, 150, 168);

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

pub struct AppState {
    mode: UiMode,
    setup: CliConfig,
    active_request: Option<TrainingRequest>,
    is_running: bool,
    is_paused: bool,
    stop_initiated: bool,
    exit_code: Option<i32>,
    status_line: String,
    latest_metrics: TrainMetrics,
    step_counter: u64,
    train_loss_history: VecDeque<(f64, f64)>,
    val_loss_history: VecDeque<(f64, f64)>,
    map50_95_history: VecDeque<(f64, f64)>,
    best_map50: Option<f64>,
    best_map50_95: Option<f64>,
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
            setup,
            active_request: None,
            is_running: false,
            is_paused: false,
            stop_initiated: false,
            exit_code: None,
            status_line: "Ready".to_string(),
            latest_metrics: TrainMetrics::default(),
            step_counter: 0,
            train_loss_history: VecDeque::with_capacity(MAX_HISTORY_POINTS),
            val_loss_history: VecDeque::with_capacity(MAX_HISTORY_POINTS),
            map50_95_history: VecDeque::with_capacity(MAX_HISTORY_POINTS),
            best_map50: None,
            best_map50_95: None,
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
        }
    }

    fn push_log(&mut self, line: String) {
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

        if metrics.train_loss.is_finite() && metrics.train_loss >= 0.0 {
            if self.train_loss_history.len() >= MAX_HISTORY_POINTS {
                self.train_loss_history.pop_front();
            }
            self.train_loss_history.push_back((x, metrics.train_loss));
        }

        if let Some(vl) = metrics.val_loss {
            if vl.is_finite() && vl >= 0.0 {
                if self.val_loss_history.len() >= MAX_HISTORY_POINTS {
                    self.val_loss_history.pop_front();
                }
                self.val_loss_history.push_back((x, vl));
            }
        }

        if let Some(m95) = metrics.map50_95 {
            if m95.is_finite() && m95 >= 0.0 {
                if self.map50_95_history.len() >= MAX_HISTORY_POINTS {
                    self.map50_95_history.pop_front();
                }
                self.map50_95_history.push_back((x, m95));
                self.best_map50_95 = Some(self.best_map50_95.map_or(m95, |prev| prev.max(m95)));
            }
        }

        if let Some(m50) = metrics.map50 {
            if m50.is_finite() && m50 >= 0.0 {
                self.best_map50 = Some(self.best_map50.map_or(m50, |prev| prev.max(m50)));
            }
        }

        self.merge_metric_gpus(&metrics.gpus);
        self.latest_metrics = metrics;
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
            guard.exit_code = code;
            guard.is_running = false;
            guard.is_paused = false;
            guard.training_ended_at = Some(Instant::now());
            guard.status_line = match code {
                Some(0) => "Completed".to_string(),
                Some(c) if guard.stop_initiated => format!("Stopped (exit {c})"),
                Some(c) => format!("Exited (code {c})"),
                None if guard.stop_initiated => "Stopped".to_string(),
                None => "Finished".to_string(),
            };
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

fn launch_training_thread(state: Arc<Mutex<AppState>>, request: TrainingRequest) {
    if let Ok(mut guard) = state.lock() {
        guard.mode = UiMode::Training;
        guard.active_request = Some(request.clone());
        guard.is_running = true;
        guard.is_paused = false;
        guard.stop_initiated = false;
        guard.exit_code = None;
        guard.training_started_at = Some(Instant::now());
        guard.training_ended_at = None;
        guard.status_line = "Initializing training...".to_string();
        guard.latest_metrics = TrainMetrics {
            epoch: 0,
            total_epochs: request.epochs,
            ..Default::default()
        };
    }

    let sink: Arc<dyn TrainingEventSink> = Arc::new(TuiEventSink::new(Arc::clone(&state)));
    tauri::async_runtime::spawn(async move {
        let result = run_training(sink.clone(), request).await;
        if let Err(err) = result {
            sink.on_log(format!("[LS-Trainer Error] {err}"));
            sink.on_exit(Some(1));
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

fn render_run_summary(frame: &mut Frame, area: Rect, state: &AppState) {
    let split = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(38), Constraint::Percentage(62)])
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

    let (l1, l5, l15) = state.host_load_avg;
    let load_line = Line::from(vec![
        Span::styled("Load     ", Style::default().fg(LABEL_MUTED)),
        Span::styled(
            format!("{l1:.2}  {l5:.2}  {l15:.2}"),
            Style::default().fg(Color::White),
        ),
    ]);
    frame.render_widget(Paragraph::new(load_line), rows[2]);

    let cpu_samples: Vec<u64> = state.host_cpu_history.iter().copied().collect();
    let sparkline = Sparkline::default()
        .data(&cpu_samples)
        .max(100)
        .style(Style::default().fg(BAR_GPU_TEAL));
    frame.render_widget(sparkline, rows[3]);
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

    let (arch_str, model_str, dataset_str, dev_str) = if let Some(ref req) = state.active_request {
        (
            req.model_family.as_str(),
            req.model_choice.as_str(),
            req.dataset_path.as_str(),
            req.device.as_str(),
        )
    } else {
        (
            state.setup.arch.as_family_str(),
            state.setup.model.as_str(),
            state.setup.dataset.as_str(),
            state.setup.device.as_str(),
        )
    };

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

    let divider_width = padded.width.saturating_sub(14) as usize;
    let divider_tail = "─".repeat(divider_width.max(4));
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
            Span::styled("Dataset  ", Style::default().fg(LABEL_MUTED)),
            Span::styled(
                dataset_str.to_string(),
                Style::default().fg(Color::Rgb(196, 206, 218)),
            ),
        ]),
        Line::from(Span::styled(
            format!("── Key Metrics {divider_tail}"),
            Style::default().fg(TABLE_BORDER_SOFT),
        )),
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
        Line::from(vec![
            Span::styled("Best mAP ", Style::default().fg(LABEL_MUTED)),
            Span::styled("50 ", Style::default().fg(LABEL_MUTED)),
            Span::styled(
                format!("{map50_str:<7}"),
                Style::default()
                    .fg(Color::LightGreen)
                    .add_modifier(Modifier::BOLD),
            ),
            sep_span.clone(),
            Span::styled("50:95 ", Style::default().fg(LABEL_MUTED)),
            Span::styled(
                map95_str,
                Style::default()
                    .fg(Color::LightGreen)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("Speed    ", Style::default().fg(LABEL_MUTED)),
            Span::styled(
                format!("{:>5.1} samples/s", m.samples_per_sec),
                Style::default()
                    .fg(Color::LightGreen)
                    .add_modifier(Modifier::BOLD),
            ),
            sep_span,
            Span::styled("Elapsed ", Style::default().fg(LABEL_MUTED)),
            Span::styled(elapsed_str, Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("ETA      ", Style::default().fg(LABEL_MUTED)),
            Span::styled(
                format_eta_seconds(m.eta_secs),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
    ];

    frame.render_widget(Paragraph::new(lines), rows[2]);
}

fn render_bottom_log_and_footer(frame: &mut Frame, area: Rect, state: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(3), Constraint::Length(1)])
        .split(area);

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

    let inner = log_block.inner(chunks[0]);
    frame.render_widget(log_block, chunks[0]);

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

    let pause_label = if state.is_paused {
        "Resume"
    } else {
        "Pause / Resume"
    };
    let footer = Line::from(vec![
        Span::styled(
            " [Space] ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!(" {pause_label}  "),
            Style::default().fg(Color::White),
        ),
        Span::styled(
            " [c] ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::LightGreen)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Force Checkpoint  ", Style::default().fg(Color::White)),
        Span::styled(
            " [Up/Down] ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Scroll Logs  ", Style::default().fg(Color::White)),
        Span::styled(
            " [F12] ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Copy Log  ", Style::default().fg(Color::White)),
        Span::styled(
            " [q] ",
            Style::default()
                .fg(Color::White)
                .bg(Color::Red)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Graceful Stop & Exit ", Style::default().fg(Color::White)),
    ]);

    frame.render_widget(
        Paragraph::new(footer).alignment(Alignment::Center),
        chunks[1],
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

fn draw_ui(frame: &mut Frame, state: &AppState) {
    if state.mode == UiMode::Setup {
        render_launcher_screen(frame, frame.area(), state);
        return;
    }

    let root_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),
            Constraint::Percentage(52),
            Constraint::Min(6),
        ])
        .split(frame.area());

    render_top_gpu_strip(frame, root_chunks[0], state);

    let center_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(68), Constraint::Percentage(32)])
        .split(root_chunks[1]);

    render_center_chart(frame, center_chunks[0], state);
    render_run_summary(frame, center_chunks[1], state);

    render_bottom_log_and_footer(frame, root_chunks[2], state);
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

    if let Some(req) = initial_req {
        launch_training_thread(Arc::clone(&state), req);
    }

    let mut terminal = ratatui::init();
    let mut clipboard: Option<Clipboard> = None;
    let tick_rate = Duration::from_millis(80);

    loop {
        if let Ok(mut guard) = state.lock() {
            if guard.is_running && !guard.is_paused {
                guard.spinner_tick = guard.spinner_tick.wrapping_add(1);
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
                    break;
                }

                let mode = state.lock().map(|g| g.mode).unwrap_or(UiMode::Training);

                if mode == UiMode::Setup {
                    if handle_setup_key(&state, key.code) {
                        break;
                    }
                    continue;
                }

                match key.code {
                    KeyCode::Char('q') | KeyCode::Char('Q') => {
                        graceful_stop_and_wait(&state);
                        break;
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
                        copy_logs_with_feedback(&state, &mut clipboard);
                    }
                    _ => {}
                }
            }
        }
    }

    ratatui::restore();
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
    use ratatui::{Terminal, backend::TestBackend};

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
    fn short_real_run_from_launcher_verifies_tui_rendering() {
        let tmp_ds = std::env::temp_dir().join("ls_trainer_tui_tiny_ds");
        let _ = std::fs::remove_dir_all(&tmp_ds);
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

        // 1. Launch with no flags -> starts in Setup (Launcher) mode
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

        // Press F2 in launcher -> transitions to UiMode::Training and spawns run_training
        let should_quit = handle_setup_key(&state, KeyCode::F(2));
        assert!(!should_quit);
        assert_eq!(state.lock().unwrap().mode, UiMode::Training);
        assert!(state.lock().unwrap().is_running);

        let backend = TestBackend::new(120, 32);
        let mut terminal = Terminal::new(backend).expect("create test terminal");

        let deadline = Instant::now() + Duration::from_secs(35);
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
            "Expected short training run to exit with code 0; status={}",
            guard.status_line
        );
        assert!(
            saw_live_metrics,
            "Expected live @@LS_METRICS@@ updates during run"
        );
        assert_eq!(guard.latest_metrics.epoch, 1);

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
            "Expected GPU telemetry during XPU training run"
        );
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
}
