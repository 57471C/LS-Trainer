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
    widgets::{Axis, Block, BorderType, Borders, Chart, Dataset, Gauge, GraphType, Paragraph, Sparkline},
};
use sysinfo::System;

const MAX_LOG_LINES: usize = 500;
const MAX_HISTORY_POINTS: usize = 300;
const MAX_SPARKLINE_SAMPLES: usize = 60;

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
    run_name: String,
    focused_field: usize,
    error_banner: Option<String>,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            interactive: true,
            arch: ArchChoice::Yolo,
            dataset: String::new(),
            model: ArchChoice::Yolo.default_model().to_string(),
            epochs: "60".to_string(),
            batch: "4".to_string(),
            device: "auto".to_string(),
            imgsz: 960,
            workers: None,
            run_name: "BB-Parts-v3".to_string(),
            focused_field: 0,
            error_banner: None,
        }
    }
}

impl CliConfig {
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
            workers: self.workers,
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

    fn apply_metrics(&mut self, metrics: TrainMetrics) {
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
            let used_ram_gb = sys.used_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
            let total_ram_gb = sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);

            if let Ok(mut guard) = state.lock() {
                guard.host_cpu_percent = cpu_usage.clamp(0.0, 100.0);
                guard.host_ram_used_gb = used_ram_gb;
                guard.host_ram_total_gb = total_ram_gb;
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
                    spans.push(Span::styled(std::mem::take(&mut segment), current_style));
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
            spans.push(Span::styled(segment, current_style));
        }
        return Line::from(spans);
    }

    let base_color = if raw.contains("Error]") || raw.contains("Traceback") {
        Color::LightRed
    } else if raw.contains("Warning]") {
        Color::Yellow
    } else if raw.contains("[LS-Trainer Metrics]") || raw.contains("Final Metrics") {
        Color::LightGreen
    } else if raw.starts_with("[LS-Trainer") {
        Color::Cyan
    } else if raw.starts_with("Epoch ") {
        Color::LightBlue
    } else {
        Color::Gray
    };

    Line::from(Span::styled(raw.to_string(), Style::default().fg(base_color)))
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
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::DarkGray));

        let inner = block.inner(card_area);
        frame.render_widget(block, card_area);

        if inner.height == 0 || inner.width == 0 {
            continue;
        }

        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(1)])
            .split(inner);

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

        let gauge = Gauge::default()
            .gauge_style(
                Style::default()
                    .fg(Color::LightCyan)
                    .bg(Color::Rgb(28, 36, 48)),
            )
            .ratio(vram_ratio)
            .label(Span::styled(
                vram_label,
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ));
        frame.render_widget(gauge, rows[0]);

        let spark_cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Length(14), Constraint::Min(4)])
            .split(rows[1]);

        let util_text = Paragraph::new(Line::from(vec![
            Span::styled("Util ", Style::default().fg(Color::Gray)),
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
            .style(Style::default().fg(Color::LightGreen));
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
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::DarkGray));

    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height == 0 {
        return;
    }

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(1)])
        .split(inner);

    let ram_ratio = if state.host_ram_total_gb > 0.0 {
        (state.host_ram_used_gb / state.host_ram_total_gb).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let gauge = Gauge::default()
        .gauge_style(
            Style::default()
                .fg(Color::LightBlue)
                .bg(Color::Rgb(28, 36, 48)),
        )
        .ratio(ram_ratio)
        .label(format!(
            "System RAM {:.1} / {:.1} GiB",
            state.host_ram_used_gb, state.host_ram_total_gb
        ));
    frame.render_widget(gauge, rows[0]);

    let samples: Vec<u64> = state.host_cpu_history.iter().copied().collect();
    let spark = Sparkline::default()
        .data(&samples)
        .max(100)
        .style(Style::default().fg(Color::LightGreen));
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

    let all_pts = train_pts
        .iter()
        .chain(val_pts.iter())
        .chain(map_pts.iter());

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
                        .fg(Color::LightCyan)
                        .add_modifier(Modifier::BOLD),
                ))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::DarkGray)),
        )
        .x_axis(
            Axis::default()
                .title(Span::styled("Step", Style::default().fg(Color::Gray)))
                .style(Style::default().fg(Color::DarkGray))
                .bounds([min_x, max_x])
                .labels(vec![
                    Span::raw(format!("{min_x:.0}")),
                    Span::raw(format!("{mid_x:.0}")),
                    Span::raw(format!("{max_x:.0}")),
                ]),
        )
        .y_axis(
            Axis::default()
                .title(Span::styled("Value", Style::default().fg(Color::Gray)))
                .style(Style::default().fg(Color::DarkGray))
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
    let block = Block::default()
        .title(Span::styled(
            " Run Summary ",
            Style::default()
                .fg(Color::LightCyan)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::DarkGray));

    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height == 0 || inner.width == 0 {
        return;
    }

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(4),
        ])
        .split(inner);

    let m = &state.latest_metrics;
    let total_ep = m.total_epochs.max(1);
    let ep_ratio = if m.total_epochs > 0 {
        (m.epoch as f64 / total_ep as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let ep_gauge = Gauge::default()
        .gauge_style(
            Style::default()
                .fg(Color::LightMagenta)
                .bg(Color::Rgb(28, 36, 48)),
        )
        .ratio(ep_ratio)
        .label(format!("Epoch {} / {}", m.epoch, m.total_epochs));
    frame.render_widget(ep_gauge, rows[0]);

    let step_ratio = if m.total_steps > 0 {
        (m.step as f64 / m.total_steps as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let step_gauge = Gauge::default()
        .gauge_style(
            Style::default()
                .fg(Color::LightBlue)
                .bg(Color::Rgb(28, 36, 48)),
        )
        .ratio(step_ratio)
        .label(format!("Step {} / {}", m.step, m.total_steps));
    frame.render_widget(step_gauge, rows[1]);

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
        Color::Cyan
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
        (Some(start), Some(end)) => format_eta_seconds(end.saturating_duration_since(start).as_secs_f64()),
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

    let lines = vec![
        Line::from(vec![
            Span::styled("State:   ", Style::default().fg(Color::Gray)),
            Span::styled(
                status_badge,
                Style::default()
                    .fg(status_color)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("Arch:    ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{arch_str} ({model_str})"),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled("Device:  ", Style::default().fg(Color::Gray)),
            Span::styled(dev_str.to_string(), Style::default().fg(Color::LightCyan)),
        ]),
        Line::from(vec![
            Span::styled("Dataset: ", Style::default().fg(Color::Gray)),
            Span::styled(dataset_str.to_string(), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("Loss:    ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.4}", m.train_loss),
                Style::default().fg(Color::Cyan),
            ),
            Span::styled("  LR: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.2e}", m.lr),
                Style::default().fg(Color::LightYellow),
            ),
        ]),
        Line::from(vec![
            Span::styled("Speed:   ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.1} samples/s", m.samples_per_sec),
                Style::default().fg(Color::LightGreen),
            ),
        ]),
        Line::from(vec![
            Span::styled("ETA:     ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}  (elapsed {elapsed_str})", format_eta_seconds(m.eta_secs)),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("Best mAP:", Style::default().fg(Color::Gray)),
            Span::styled(
                format!(" 50={map50_str} │ 50:95={map95_str}"),
                Style::default().fg(Color::LightGreen),
            ),
        ]),
        Line::from(vec![
            Span::styled("Host:    ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!(
                    "CPU {:.0}% │ RAM {:.1}/{:.1}G",
                    state.host_cpu_percent, state.host_ram_used_gb, state.host_ram_total_gb
                ),
                Style::default().fg(Color::DarkGray),
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
                .fg(Color::LightCyan)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::DarkGray));

    let inner = log_block.inner(chunks[0]);
    frame.render_widget(log_block, chunks[0]);

    let visible_height = inner.height as usize;
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

    frame.render_widget(Paragraph::new(visible_lines), inner);

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

fn render_setup_modal(frame: &mut Frame, area: Rect, state: &AppState) {
    let block = Block::default()
        .title(Span::styled(
            " LS-Trainer TUI — Interactive Run Setup ",
            Style::default()
                .fg(Color::LightCyan)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::Cyan));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let s = &state.setup;
    let fields: [(&str, String, &str); 7] = [
        (
            "Architecture",
            format!("◀ {} ({}) ▶", s.arch.as_family_str(), s.arch.as_cli_str()),
            "Use Left/Right arrows to switch yolo | rf-detr | rt-detr",
        ),
        (
            "Dataset Path",
            s.dataset.clone(),
            "Path to YOLO dataset folder/archive or COCO export",
        ),
        (
            "Model / Weights",
            s.model.clone(),
            "e.g. yolo11s.pt, rf-detr-medium, rtdetr-v2-r18, or checkpoint path",
        ),
        ("Epochs", s.epochs.clone(), "Total training epochs"),
        ("Batch Size", s.batch.clone(), "Batch size per step"),
        (
            "Device",
            s.device.clone(),
            "auto, xpu, xpu:0, cuda, 0, or cpu",
        ),
        (
            "[ Start Training ]",
            "Press Enter to launch training".to_string(),
            "Or press Esc to exit",
        ),
    ];

    let mut lines: Vec<Line<'static>> = Vec::new();
    lines.push(Line::from(Span::styled(
        "Use Up/Down/Tab to select a field, type to edit, and press Enter on [ Start Training ].",
        Style::default().fg(Color::Gray),
    )));
    lines.push(Line::from(""));

    for (idx, (label, val, hint)) in fields.iter().enumerate() {
        let focused = s.focused_field == idx;
        let prefix = if focused { " ▶ " } else { "   " };
        let label_style = if focused {
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Gray)
        };
        let val_style = if focused {
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        };

        lines.push(Line::from(vec![
            Span::styled(prefix, label_style),
            Span::styled(format!("{label:<18}: "), label_style),
            Span::styled(format!(" {val} "), val_style),
            Span::styled(format!("   ({hint})"), Style::default().fg(Color::DarkGray)),
        ]));
        lines.push(Line::from(""));
    }

    if let Some(ref err) = s.error_banner {
        lines.push(Line::from(Span::styled(
            format!(" ⚠ {err}"),
            Style::default()
                .fg(Color::LightRed)
                .add_modifier(Modifier::BOLD),
        )));
    }

    frame.render_widget(Paragraph::new(lines), inner);
}

fn draw_ui(frame: &mut Frame, state: &AppState) {
    let root_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),
            Constraint::Percentage(52),
            Constraint::Min(6),
        ])
        .split(frame.area());

    render_top_gpu_strip(frame, root_chunks[0], state);

    if state.mode == UiMode::Setup {
        render_setup_modal(frame, root_chunks[1], state);
    } else {
        let center_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(68), Constraint::Percentage(32)])
            .split(root_chunks[1]);

        render_center_chart(frame, center_chunks[0], state);
        render_run_summary(frame, center_chunks[1], state);
    }

    render_bottom_log_and_footer(frame, root_chunks[2], state);
}

fn parse_cli_args() -> Result<Option<CliConfig>, String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        return Ok(Some(CliConfig::default()));
    }

    let mut cfg = CliConfig {
        interactive: false,
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
            }
            "--run-name" => {
                cfg.run_name = next_val()?;
            }
            other => {
                return Err(format!("Unknown argument '{other}'. Run with --help for usage."));
            }
        }
        i += 1;
    }

    if cfg.dataset.trim().is_empty() {
        cfg.interactive = true;
    }

    Ok(Some(cfg))
}

fn handle_setup_key(state: &Arc<Mutex<AppState>>, code: KeyCode) -> bool {
    let mut launch_req: Option<TrainingRequest> = None;
    if let Ok(mut guard) = state.lock() {
        let focused = guard.setup.focused_field;
        match code {
            KeyCode::Esc => return true,
            KeyCode::Tab | KeyCode::Down => {
                guard.setup.focused_field = (focused + 1) % 7;
            }
            KeyCode::BackTab | KeyCode::Up => {
                guard.setup.focused_field = if focused == 0 { 6 } else { focused - 1 };
            }
            KeyCode::Left if focused == 0 => {
                let prev_default = guard.setup.arch.default_model();
                guard.setup.arch = guard.setup.arch.prev();
                if guard.setup.model == prev_default {
                    guard.setup.model = guard.setup.arch.default_model().to_string();
                }
            }
            KeyCode::Right if focused == 0 => {
                let prev_default = guard.setup.arch.default_model();
                guard.setup.arch = guard.setup.arch.next();
                if guard.setup.model == prev_default {
                    guard.setup.model = guard.setup.arch.default_model().to_string();
                }
            }
            KeyCode::Backspace => match focused {
                1 => {
                    guard.setup.dataset.pop();
                }
                2 => {
                    guard.setup.model.pop();
                }
                3 => {
                    guard.setup.epochs.pop();
                }
                4 => {
                    guard.setup.batch.pop();
                }
                5 => {
                    guard.setup.device.pop();
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
                1 => guard.setup.dataset.push(c),
                2 => guard.setup.model.push(c),
                3 if c.is_ascii_digit() => guard.setup.epochs.push(c),
                4 if c.is_ascii_digit() => guard.setup.batch.push(c),
                5 => guard.setup.device.push(c),
                _ => {}
            },
            KeyCode::Enter => {
                if focused == 6 {
                    match guard.setup.to_training_request() {
                        Ok(req) => {
                            guard.setup.error_banner = None;
                            launch_req = Some(req);
                        }
                        Err(err) => {
                            guard.setup.error_banner = Some(err);
                        }
                    }
                } else {
                    guard.setup.focused_field = (focused + 1) % 7;
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

                let mode = state
                    .lock()
                    .map(|g| g.mode)
                    .unwrap_or(UiMode::Training);

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
