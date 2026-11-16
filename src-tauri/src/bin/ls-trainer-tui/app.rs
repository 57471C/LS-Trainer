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

pub const APP_VERSION_LABEL: &str = "Version v0.2.0";
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

pub const LAUNCHER_FIELD_COUNT: usize = 21;
pub const DEVICE_PRESETS: [&str; 7] = ["auto", "xpu", "xpu:0", "xpu:0,1", "0", "cuda", "cpu"];
pub const WORKER_PRESETS: [&str; 8] = ["auto", "0", "1", "2", "4", "6", "7", "8"];
pub const IMGSZ_PRESETS: [&str; 3] = ["640", "960", "1280"];
pub const EPOCH_PRESETS: [&str; 7] = ["1", "10", "30", "60", "100", "150", "200"];
pub const BATCH_PRESETS: [&str; 6] = ["1", "2", "4", "8", "16", "32"];
pub const AUG_PRESET_NAMES: [&str; 4] = ["bench", "off", "light", "medium"];
pub const DEGREES_PRESETS: [&str; 6] = ["0", "10", "15", "45", "90", "180"];
pub const FLIPUD_PRESETS: [&str; 6] = ["0.0", "0.1", "0.25", "0.5", "0.75", "1.0"];
pub const SCALE_PRESETS: [&str; 6] = ["0.0", "0.05", "0.1", "0.25", "0.5", "0.9"];
pub const TRANSLATE_PRESETS: [&str; 5] = ["0.0", "0.05", "0.1", "0.2", "0.5"];
pub const MOSAIC_PRESETS: [&str; 4] = ["0.0", "0.25", "0.5", "1.0"];
pub const CLOSE_MOSAIC_PRESETS: [&str; 5] = ["0", "5", "10", "15", "20"];
pub const MIXUP_PRESETS: [&str; 5] = ["0.0", "0.05", "0.1", "0.2", "0.5"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelScaleOption {
    pub scale: &'static str,
    pub id: &'static str,
    pub label: &'static str,
    pub desc: &'static str,
}

/// Webview model options for RF-DETR (`src/models.ts`). Default is `medium` (`rf-detr-medium`).
pub const RFDETR_MODEL_OPTIONS: [ModelScaleOption; 10] = [
    ModelScaleOption {
        scale: "nano",
        id: "rf-detr-nano",
        label: "RF-DETR Nano",
        desc: "30.5M params · Fast real-time transformer detector",
    },
    ModelScaleOption {
        scale: "small",
        id: "rf-detr-small",
        label: "RF-DETR Small",
        desc: "32.1M params · Balanced real-time transformer",
    },
    ModelScaleOption {
        scale: "medium",
        id: "rf-detr-medium",
        label: "RF-DETR Medium",
        desc: "33.7M params · Default high-accuracy RF-DETR",
    },
    ModelScaleOption {
        scale: "large",
        id: "rf-detr-large",
        label: "RF-DETR Large",
        desc: "128M params · Maximum detection accuracy",
    },
    ModelScaleOption {
        scale: "base",
        id: "rf-detr-base",
        label: "RF-DETR Base",
        desc: "29.0M params · Legacy base checkpoint",
    },
    ModelScaleOption {
        scale: "seg-nano",
        id: "rf-detr-seg-nano",
        label: "RF-DETR Seg Nano",
        desc: "33.6M params · Fast instance segmentation",
    },
    ModelScaleOption {
        scale: "seg-small",
        id: "rf-detr-seg-small",
        label: "RF-DETR Seg Small",
        desc: "35.3M params · Balanced instance segmentation",
    },
    ModelScaleOption {
        scale: "seg-medium",
        id: "rf-detr-seg-medium",
        label: "RF-DETR Seg Medium",
        desc: "37.6M params · High-accuracy instance segmentation",
    },
    ModelScaleOption {
        scale: "seg-large",
        id: "rf-detr-seg-large",
        label: "RF-DETR Seg Large",
        desc: "39.5M params · Maximum segmentation accuracy",
    },
    ModelScaleOption {
        scale: "seg-preview",
        id: "rf-detr-seg-preview",
        label: "RF-DETR Seg Preview",
        desc: "33.6M params · Legacy preview segmentation head",
    },
];

/// Webview model options for Ultralytics YOLO (`src/models.ts`). Default is `medium` (`yolo11m.pt`).
pub const YOLO_MODEL_OPTIONS: [ModelScaleOption; 13] = [
    ModelScaleOption {
        scale: "nano",
        id: "yolo11n.pt",
        label: "YOLO11 Nano",
        desc: "2.6M params · Fastest inference, lowest VRAM",
    },
    ModelScaleOption {
        scale: "small",
        id: "yolo11s.pt",
        label: "YOLO11 Small",
        desc: "9.4M params · Balanced speed & accuracy",
    },
    ModelScaleOption {
        scale: "medium",
        id: "yolo11m.pt",
        label: "YOLO11 Medium",
        desc: "20.1M params · Default general-purpose detector",
    },
    ModelScaleOption {
        scale: "large",
        id: "yolo11l.pt",
        label: "YOLO11 Large",
        desc: "25.3M params · High accuracy for fine defects",
    },
    ModelScaleOption {
        scale: "xlarge",
        id: "yolo11x.pt",
        label: "YOLO11 XLarge",
        desc: "56.9M params · Maximum accuracy, high compute",
    },
    ModelScaleOption {
        scale: "v26-nano",
        id: "yolo26n.pt",
        label: "YOLO26 Nano",
        desc: "2.5M params · Next-gen ultra-fast edge detector",
    },
    ModelScaleOption {
        scale: "v26-small",
        id: "yolo26s.pt",
        label: "YOLO26 Small",
        desc: "9.0M params · Next-gen balanced detector",
    },
    ModelScaleOption {
        scale: "v26-medium",
        id: "yolo26m.pt",
        label: "YOLO26 Medium",
        desc: "19.5M params · Next-gen high-accuracy detector",
    },
    ModelScaleOption {
        scale: "v26-large",
        id: "yolo26l.pt",
        label: "YOLO26 Large",
        desc: "24.8M params · Next-gen precision detector",
    },
    ModelScaleOption {
        scale: "v26-xlarge",
        id: "yolo26x.pt",
        label: "YOLO26 XLarge",
        desc: "55.0M params · Next-gen maximum capacity",
    },
    ModelScaleOption {
        scale: "seg-small",
        id: "yolo11s-seg.pt",
        label: "YOLO11 Small-Seg",
        desc: "10.1M params · Fast instance segmentation",
    },
    ModelScaleOption {
        scale: "seg-medium",
        id: "yolo11m-seg.pt",
        label: "YOLO11 Medium-Seg",
        desc: "22.4M params · High-accuracy polygon masks",
    },
    ModelScaleOption {
        scale: "v26-seg-small",
        id: "yolo26s-seg.pt",
        label: "YOLO26 Small-Seg",
        desc: "9.8M params · Next-gen instance segmentation",
    },
];

/// Webview model options for RT-DETR (`src/models.ts`). Default is `rtdetr-v2-r18`.
pub const RTDETR_MODEL_OPTIONS: [ModelScaleOption; 3] = [
    ModelScaleOption {
        scale: "r18",
        id: "rtdetr-v2-r18",
        label: "RT-DETRv2 R18",
        desc: "20.0M params · Native PyTorch/HF ResNet-18 DETR",
    },
    ModelScaleOption {
        scale: "large",
        id: "rtdetr-l.pt",
        label: "RT-DETR Large",
        desc: "32.0M params · Ultralytics HGNetv2-L DETR",
    },
    ModelScaleOption {
        scale: "xlarge",
        id: "rtdetr-x.pt",
        label: "RT-DETR XLarge",
        desc: "67.0M params · Ultralytics HGNetv2-X DETR",
    },
];

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
            Self::Yolo => "yolo11m.pt",
            Self::RfDetr => "rf-detr-medium",
            Self::RtDetr => "rtdetr-v2-r18",
        }
    }

    pub fn model_options(self) -> &'static [ModelScaleOption] {
        match self {
            Self::Yolo => &YOLO_MODEL_OPTIONS,
            Self::RfDetr => &RFDETR_MODEL_OPTIONS,
            Self::RtDetr => &RTDETR_MODEL_OPTIONS,
        }
    }

    /// Resolve a user-entered or selected scale shorthand (`nano`, `small`, `medium`, `large`, `base`, etc.)
    /// or full model ID into the canonical model choice string for this architecture.
    pub fn resolve_model_or_scale(self, raw: &str) -> String {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return self.default_model().to_string();
        }
        let lower = trimmed.to_ascii_lowercase();
        for opt in self.model_options() {
            if opt.id.eq_ignore_ascii_case(&lower)
                || opt.scale.eq_ignore_ascii_case(&lower)
                || opt.label.eq_ignore_ascii_case(&lower)
            {
                return opt.id.to_string();
            }
        }
        // Also support common short aliases
        match (self, lower.as_str()) {
            (Self::Yolo, "n") => "yolo11n.pt".to_string(),
            (Self::Yolo, "s") => "yolo11s.pt".to_string(),
            (Self::Yolo, "m") => "yolo11m.pt".to_string(),
            (Self::Yolo, "l") => "yolo11l.pt".to_string(),
            (Self::Yolo, "x" | "xl") => "yolo11x.pt".to_string(),
            (Self::RfDetr, "n") => "rf-detr-nano".to_string(),
            (Self::RfDetr, "s") => "rf-detr-small".to_string(),
            (Self::RfDetr, "m") => "rf-detr-medium".to_string(),
            (Self::RfDetr, "l") => "rf-detr-large".to_string(),
            (Self::RfDetr, "b") => "rf-detr-base".to_string(),
            (Self::RtDetr, "l") => "rtdetr-l.pt".to_string(),
            (Self::RtDetr, "x" | "xl") => "rtdetr-x.pt".to_string(),
            _ => trimmed.to_string(),
        }
    }

    pub fn find_option(self, raw: &str) -> Option<&'static ModelScaleOption> {
        let resolved = self.resolve_model_or_scale(raw);
        self.model_options()
            .iter()
            .find(|o| o.id.eq_ignore_ascii_case(&resolved))
    }

    pub fn is_preset_model(self, raw: &str) -> bool {
        let trimmed = raw.trim();
        self.model_options().iter().any(|o| {
            o.id.eq_ignore_ascii_case(trimmed) || o.scale.eq_ignore_ascii_case(trimmed)
        })
    }

    pub fn cycle_model(self, current: &str, forward: bool) -> String {
        let opts = self.model_options();
        let resolved = self.resolve_model_or_scale(current);
        let idx = opts
            .iter()
            .position(|o| o.id.eq_ignore_ascii_case(&resolved))
            .unwrap_or_else(|| {
                opts.iter()
                    .position(|o| o.id == self.default_model())
                    .unwrap_or(0)
            });
        let next_idx = if forward {
            (idx + 1) % opts.len()
        } else if idx == 0 {
            opts.len() - 1
        } else {
            idx - 1
        };
        opts[next_idx].id.to_string()
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
    pub imgsz_input: String,
    pub workers: Option<u32>,
    pub workers_input: String,
    pub freeze_backbone: bool,
    pub freeze_layers: u32,
    pub amp: bool,
    pub run_name: String,
    pub degrees: String,
    pub flipud: String,
    pub scale: String,
    pub translate: String,
    pub mosaic: String,
    pub close_mosaic: String,
    pub mixup: String,
    pub copy_paste: bool,
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
            imgsz_input: "960".to_string(),
            workers: None,
            workers_input: "auto".to_string(),
            freeze_backbone: false,
            freeze_layers: 10,
            amp: true,
            run_name: "BB-Parts-v3".to_string(),
            degrees: "180".to_string(),
            flipud: "0.5".to_string(),
            scale: "0.05".to_string(),
            translate: "0.1".to_string(),
            mosaic: "0.0".to_string(),
            close_mosaic: "10".to_string(),
            mixup: "0.0".to_string(),
            copy_paste: false,
            focused_field: 0,
            editing_field: false,
            error_banner: None,
        }
    }
}

impl CliConfig {
    pub fn switch_arch(&mut self, next_arch: ArchChoice) {
        let old_arch = self.arch;
        let current_scale = old_arch.find_option(&self.model).map(|o| o.scale);
        let was_preset = old_arch.is_preset_model(&self.model) || self.model.trim().is_empty();
        self.arch = next_arch;
        if was_preset {
            if let Some(scale_name) = current_scale {
                if old_arch != ArchChoice::RtDetr && next_arch != ArchChoice::RtDetr {
                    if let Some(mapped) = next_arch
                        .model_options()
                        .iter()
                        .find(|o| o.scale == scale_name)
                    {
                        self.model = mapped.id.to_string();
                        return;
                    }
                }
            }
            self.model = next_arch.default_model().to_string();
        }
    }

    pub fn active_aug_preset(&self) -> &'static str {
        let deg = self.degrees.trim().parse::<f64>().unwrap_or(180.0);
        let flip = self.flipud.trim().parse::<f64>().unwrap_or(0.5);
        let sc = self.scale.trim().parse::<f64>().unwrap_or(0.05);
        let tr = self.translate.trim().parse::<f64>().unwrap_or(0.1);
        let mos = self.mosaic.trim().parse::<f64>().unwrap_or(0.0);
        let cm = self.close_mosaic.trim().parse::<u32>().unwrap_or(10);
        let mix = self.mixup.trim().parse::<f64>().unwrap_or(0.0);
        let cp = self.copy_paste;

        let approx = |a: f64, b: f64| (a - b).abs() < 1e-4;

        if approx(mos, 0.0)
            && approx(mix, 0.0)
            && !cp
            && approx(deg, 180.0)
            && approx(flip, 0.5)
            && approx(tr, 0.1)
            && approx(sc, 0.05)
            && cm == 10
        {
            "bench"
        } else if approx(mos, 0.0)
            && approx(mix, 0.0)
            && !cp
            && approx(deg, 0.0)
            && approx(flip, 0.0)
            && approx(tr, 0.0)
            && approx(sc, 0.0)
            && cm == 0
        {
            "off"
        } else if approx(mos, 0.5)
            && approx(mix, 0.0)
            && !cp
            && approx(deg, 5.0)
            && approx(flip, 0.0)
            && approx(tr, 0.1)
            && approx(sc, 0.25)
            && cm == 10
        {
            "light"
        } else if approx(mos, 1.0)
            && approx(mix, 0.1)
            && !cp
            && approx(deg, 10.0)
            && approx(flip, 0.1)
            && approx(tr, 0.1)
            && approx(sc, 0.5)
            && cm == 10
        {
            "medium"
        } else {
            "custom"
        }
    }

    pub fn apply_aug_preset(&mut self, preset: &str) {
        match preset.trim().to_ascii_lowercase().as_str() {
            "bench" => {
                self.mosaic = "0.0".to_string();
                self.mixup = "0.0".to_string();
                self.copy_paste = false;
                self.degrees = "180".to_string();
                self.flipud = "0.5".to_string();
                self.translate = "0.1".to_string();
                self.scale = "0.05".to_string();
                self.close_mosaic = "10".to_string();
            }
            "off" => {
                self.mosaic = "0.0".to_string();
                self.mixup = "0.0".to_string();
                self.copy_paste = false;
                self.degrees = "0".to_string();
                self.flipud = "0.0".to_string();
                self.translate = "0.0".to_string();
                self.scale = "0.0".to_string();
                self.close_mosaic = "0".to_string();
            }
            "light" => {
                self.mosaic = "0.5".to_string();
                self.mixup = "0.0".to_string();
                self.copy_paste = false;
                self.degrees = "5".to_string();
                self.flipud = "0.0".to_string();
                self.translate = "0.1".to_string();
                self.scale = "0.25".to_string();
                self.close_mosaic = "10".to_string();
            }
            "medium" => {
                self.mosaic = "1.0".to_string();
                self.mixup = "0.1".to_string();
                self.copy_paste = false;
                self.degrees = "10".to_string();
                self.flipud = "0.1".to_string();
                self.translate = "0.1".to_string();
                self.scale = "0.5".to_string();
                self.close_mosaic = "10".to_string();
            }
            _ => {}
        }
    }

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

    pub fn resolve_imgsz(&self) -> Result<u32, String> {
        if self.imgsz != 960 && self.imgsz_input.trim() == "960" {
            return Ok(self.imgsz.max(32));
        }
        let raw = self.imgsz_input.trim();
        if raw.is_empty() {
            return Ok(self.imgsz.max(32));
        }
        let parsed = raw
            .parse::<u32>()
            .map_err(|_| format!("Invalid resolution/imgsz value: '{raw}'"))?;
        Ok(parsed.max(32))
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
        let imgsz_val = self.resolve_imgsz()?;
        let workers_val = self.parse_workers()?;
        let model_resolved = self.arch.resolve_model_or_scale(&self.model);
        let task = if model_resolved.to_ascii_lowercase().contains("-seg") {
            "segment".to_string()
        } else {
            "detect".to_string()
        };

        let degrees_val = self
            .degrees
            .trim()
            .parse::<f32>()
            .map_err(|_| format!("Invalid degrees value: '{}'", self.degrees))?
            .clamp(0.0, 180.0);
        let flipud_val = self
            .flipud
            .trim()
            .parse::<f32>()
            .map_err(|_| format!("Invalid flipud value: '{}'", self.flipud))?
            .clamp(0.0, 1.0);
        let scale_val = self
            .scale
            .trim()
            .parse::<f32>()
            .map_err(|_| format!("Invalid scale value: '{}'", self.scale))?
            .clamp(0.0, 1.0);
        let translate_val = self
            .translate
            .trim()
            .parse::<f32>()
            .map_err(|_| format!("Invalid translate value: '{}'", self.translate))?
            .clamp(0.0, 1.0);
        let mosaic_val = self
            .mosaic
            .trim()
            .parse::<f32>()
            .map_err(|_| format!("Invalid mosaic value: '{}'", self.mosaic))?
            .clamp(0.0, 1.0);
        let close_mosaic_val = self
            .close_mosaic
            .trim()
            .parse::<u32>()
            .map_err(|_| format!("Invalid close_mosaic value: '{}'", self.close_mosaic))?;
        let mixup_val = self
            .mixup
            .trim()
            .parse::<f32>()
            .map_err(|_| format!("Invalid mixup value: '{}'", self.mixup))?
            .clamp(0.0, 1.0);
        let copy_paste_val: f32 = if self.copy_paste { 0.3 } else { 0.0 };

        Ok(TrainingRequest {
            dataset_path: dataset_trimmed.to_string(),
            model_choice: model_resolved,
            run_name: if self.run_name.trim().is_empty() {
                "BB-Parts-v3".to_string()
            } else {
                self.run_name.trim().to_string()
            },
            epochs: epochs_val,
            imgsz: imgsz_val,
            device: if self.device.trim().is_empty() {
                "auto".to_string()
            } else {
                self.device.trim().to_string()
            },
            batch: batch_val,
            amp: self.amp,
            freeze_backbone: self.freeze_backbone,
            freeze_layers: self.freeze_layers,
            mosaic: mosaic_val,
            mixup: mixup_val,
            copy_paste: copy_paste_val,
            degrees: degrees_val,
            translate: translate_val,
            scale: scale_val,
            flipud: flipud_val,
            close_mosaic: close_mosaic_val,
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
    let p = app_lib::expand_dataset_path(dataset_path);
    if !p.exists() {
        return None;
    }
    let root = if p.is_file() { p.parent()? } else { p.as_path() };

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
                    "ls-trainer-tui — Terminal trainer for YOLO, RF-DETR, and RT-DETR\n\
                     {APP_VERSION_LABEL}\n\n\
                     USAGE:\n  \
                       ls-trainer-tui [OPTIONS]\n\n\
                     OPTIONS:\n  \
                       --arch <yolo|rf-detr|rt-detr>   Model architecture family\n  \
                       --scale <nano|small|medium|...> Model scale shorthand (default: medium)\n  \
                       --model <path>                  Model ID or weights path\n  \
                       --dataset <path>                Dataset folder or archive path\n  \
                       --epochs <n>                    Number of training epochs (default: 60)\n  \
                       --batch <n>                     Batch size (default: 4)\n  \
                       --device <str>                  Device (auto, xpu, xpu:0, xpu:0,1 for YOLO, cuda, 0, cpu)\n  \
                       --imgsz <n>                     Image resolution (640, 960, 1280; default: 960)\n  \
                       --workers <n>                   DataLoader workers (default: auto; 0 for Windows Arc)\n  \
                       --freeze-backbone               Freeze backbone layers (freeze=10)\n  \
                       --no-amp                        Disable mixed precision (use FP32)\n  \
                       --aug-preset <bench|off|light|medium> Augmentation preset (default: bench)\n  \
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
                } else {
                    cfg.model = cfg.arch.resolve_model_or_scale(&cfg.model);
                }
            }
            "--dataset" => {
                cfg.dataset = next_val()?;
            }
            "--model" | "--scale" => {
                let raw = next_val()?;
                cfg.model = cfg.arch.resolve_model_or_scale(&raw);
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
            "--imgsz" | "--resolution" => {
                let raw = next_val()?;
                let parsed = raw
                    .parse::<u32>()
                    .map_err(|_| format!("Invalid --imgsz value '{raw}'"))?;
                cfg.imgsz = parsed;
                cfg.imgsz_input = parsed.to_string();
            }
            "--workers" => {
                let raw = next_val()?;
                if raw.eq_ignore_ascii_case("auto") {
                    cfg.workers = None;
                    cfg.workers_input = "auto".to_string();
                } else {
                    let parsed = raw
                        .parse::<u32>()
                        .map_err(|_| format!("Invalid --workers value '{raw}'"))?;
                    cfg.workers = Some(parsed);
                    cfg.workers_input = parsed.to_string();
                }
            }
            "--freeze-backbone" => {
                if let Some(ref v) = inline_val {
                    cfg.freeze_backbone = matches!(
                        v.trim().to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    );
                } else {
                    cfg.freeze_backbone = true;
                }
            }
            "--amp" => {
                if let Some(ref v) = inline_val {
                    cfg.amp = !matches!(
                        v.trim().to_ascii_lowercase().as_str(),
                        "0" | "false" | "no" | "off"
                    );
                } else {
                    cfg.amp = true;
                }
            }
            "--no-amp" => {
                cfg.amp = false;
            }
            "--aug-preset" => {
                let raw = next_val()?;
                cfg.apply_aug_preset(&raw);
            }
            "--degrees" => {
                cfg.degrees = next_val()?;
            }
            "--flipud" => {
                cfg.flipud = next_val()?;
            }
            "--scale-gain" => {
                cfg.scale = next_val()?;
            }
            "--translate" => {
                cfg.translate = next_val()?;
            }
            "--mosaic" => {
                cfg.mosaic = next_val()?;
            }
            "--close-mosaic" => {
                cfg.close_mosaic = next_val()?;
            }
            "--mixup" => {
                cfg.mixup = next_val()?;
            }
            "--copy-paste" => {
                if let Some(ref v) = inline_val {
                    cfg.copy_paste = matches!(
                        v.trim().to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on" | "0.3"
                    );
                } else {
                    cfg.copy_paste = true;
                }
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
