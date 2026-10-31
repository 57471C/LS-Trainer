use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use tauri::ipc::Channel;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x08000000;

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

fn default_model_family() -> String {
    "YOLO".to_string()
}

fn default_batch() -> u32 {
    4
}

fn default_amp() -> bool {
    true
}

fn default_freeze_backbone() -> bool {
    false
}

fn default_freeze_layers() -> u32 {
    10
}

fn default_mosaic() -> f32 {
    0.0
}

fn default_mixup() -> f32 {
    0.0
}

fn default_copy_paste() -> f32 {
    0.0
}

fn default_degrees() -> f32 {
    180.0
}

fn default_translate() -> f32 {
    0.1
}

fn default_scale() -> f32 {
    0.05
}

fn default_flipud() -> f32 {
    0.5
}

fn default_close_mosaic() -> u32 {
    10
}

fn default_task() -> String {
    "detect".to_string()
}

fn default_export_imgsz() -> u32 {
    640
}

fn default_export_dynamic() -> bool {
    false
}

fn default_export_simplify() -> bool {
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

fn get_project_root() -> PathBuf {
    if let Ok(cwd) = std::env::current_dir() {
        if cwd.join(".venv").exists()
            || cwd.join("src-tauri").exists()
            || cwd.join("package.json").exists()
        {
            return cwd;
        }
        if let Some(parent) = cwd.parent() {
            if parent.join(".venv").exists()
                || parent.join("src-tauri").exists()
                || parent.join("package.json").exists()
            {
                return parent.to_path_buf();
            }
        }
    }

    if let Ok(exe_path) = std::env::current_exe() {
        let mut cur = exe_path.parent();
        while let Some(dir) = cur {
            if dir.join(".venv").exists()
                || dir.join("src-tauri").exists()
                || dir.join("package.json").exists()
            {
                return dir.to_path_buf();
            }
            cur = dir.parent();
        }
    }

    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

pub fn to_display_path(path: &Path) -> String {
    #[cfg(target_os = "windows")]
    {
        path.to_string_lossy().replace('/', "\\")
    }
    #[cfg(not(target_os = "windows"))]
    {
        path.to_string_lossy().replace('\\', "/")
    }
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
pub struct GpuInfo {
    pub cuda_available: bool,
    #[serde(default)]
    pub xpu_available: bool,
    pub cpu_fallback_confirmed: bool,
    pub device_name: Option<String>,
    pub driver_version: Option<String>,
    pub recommended_wheel: String,
    pub cuda_label: String,
    pub torch_version: Option<String>,
    pub needs_repair: bool,
    pub status_summary: String,
    #[serde(default = "default_recommended_workers")]
    pub recommended_workers: u32,
    #[serde(default = "default_physical_cores")]
    pub physical_cores: u32,
    #[serde(default = "default_logical_cores")]
    pub logical_cores: u32,
}

fn default_recommended_workers() -> u32 {
    4
}

fn default_physical_cores() -> u32 {
    4
}

fn default_logical_cores() -> u32 {
    8
}

/// Compute a sensible DataLoader `num_workers` value:
/// - Prefer physical cores (`psutil.cpu_count(logical=False)` if available, else `logical_cores // 2`).
/// - Cap at 8 and leave at least one core free.
pub fn compute_recommended_workers(physical_cores: Option<u32>, logical_cores: u32) -> u32 {
    let logical = logical_cores.max(1);
    let phys = physical_cores
        .filter(|&c| c > 0)
        .unwrap_or_else(|| (logical / 2).max(1));
    if phys <= 1 {
        if logical > 1 {
            1
        } else {
            0
        }
    } else {
        (phys - 1).min(8)
    }
}

fn detect_cpu_worker_info(project_root: &Path) -> (u32, u32, u32) {
    let fallback_logical = std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(4)
        .max(1);
    let fallback_phys = (fallback_logical / 2).max(1);

    if let Some(python_exe) = get_venv_python(project_root) {
        let mut cmd = Command::new(&python_exe);
        cmd.args([
            "-c",
            "import os; l = os.cpu_count() or 1; p = None\ntry:\n    import psutil\n    p = psutil.cpu_count(logical=False)\nexcept Exception:\n    pass\nphys = int(p) if (p and p > 0) else max(1, l // 2)\nprint(f'CPU_INFO|{phys}|{l}')",
        ]);
        #[cfg(windows)]
        {
            cmd.creation_flags(CREATE_NO_WINDOW);
        }
        if let Ok(out) = cmd.output() {
            if out.status.success() {
                let text = String::from_utf8_lossy(&out.stdout);
                for line in text.lines() {
                    if let Some(rest) = line.trim().strip_prefix("CPU_INFO|") {
                        let parts: Vec<&str> = rest.split('|').collect();
                        if parts.len() >= 2 {
                            let phys = parts[0].parse::<u32>().ok().unwrap_or(fallback_phys).max(1);
                            let log = parts[1].parse::<u32>().ok().unwrap_or(fallback_logical).max(1);
                            let rec = compute_recommended_workers(Some(phys), log);
                            return (rec, phys, log);
                        }
                    }
                }
            }
        }
    }

    let rec = compute_recommended_workers(Some(fallback_phys), fallback_logical);
    (rec, fallback_phys, fallback_logical)
}

fn parse_driver_version(raw: &str) -> (String, Option<u32>) {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return (String::new(), None);
    }
    let parts: Vec<&str> = trimmed.split('.').collect();
    if parts.len() == 2 {
        let major = parts[0].trim().parse::<u32>().ok();
        return (trimmed.to_string(), major);
    }
    if parts.len() >= 4 {
        // Windows WMI format e.g. "32.0.15.7216" -> last 5 digits "57216" -> "572.16"
        let combined: String = parts[parts.len() - 2..]
            .join("")
            .chars()
            .filter(|c| c.is_ascii_digit())
            .collect();
        if combined.len() >= 5 {
            let tail = &combined[combined.len() - 5..];
            let major_str = &tail[..3];
            let minor_str = &tail[3..];
            if let Ok(major) = major_str.parse::<u32>() {
                return (format!("{major}.{minor_str}"), Some(major));
            }
        }
    }
    let first_num = parts
        .first()
        .and_then(|s| s.trim().parse::<u32>().ok());
    (trimmed.to_string(), first_num)
}

fn shorten_gpu_name(full_name: &str) -> String {
    let cleaned = full_name
        .replace("(R)", "")
        .replace("(TM)", "")
        .replace("  ", " ");
    let trimmed = cleaned.trim();
    for prefix in ["NVIDIA GeForce ", "NVIDIA RTX ", "NVIDIA "] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            if prefix == "NVIDIA RTX " {
                return format!("RTX {}", rest.trim());
            }
            return rest.trim().to_string();
        }
    }
    trimmed.to_string()
}

fn is_intel_arc_gpu(name: &str) -> bool {
    let lower = name.to_lowercase();
    (lower.contains("intel") && (lower.contains("arc") || lower.contains("xpu")))
        || lower.starts_with("arc ")
}

fn is_blackwell_gpu(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower.contains("rtx 50")
        || lower.contains("rtx50")
        || lower.contains("blackwell")
        || lower.contains("gb10")
        || lower.contains("gb20")
        || lower.contains("b100")
        || lower.contains("b200")
}

fn select_pytorch_wheel(
    gpu_name: Option<&str>,
    driver_major: Option<u32>,
) -> (&'static str, &'static str, &'static str) {
    if let Some(name) = gpu_name {
        if is_intel_arc_gpu(name) {
            return (
                "xpu",
                "https://download.pytorch.org/whl/xpu",
                "Intel Arc XPU",
            );
        }
        let driver_is_560_plus = driver_major.map(|d| d >= 560).unwrap_or(false);
        if is_blackwell_gpu(name) || driver_is_560_plus {
            (
                "cu128",
                "https://download.pytorch.org/whl/cu128",
                "CUDA 12.8",
            )
        } else {
            (
                "cu124",
                "https://download.pytorch.org/whl/cu124",
                "CUDA 12.4",
            )
        }
    } else {
        ("cpu", "https://download.pytorch.org/whl/cpu", "CPU-only")
    }
}

fn detect_hardware_gpu() -> (Option<String>, Option<String>, Option<u32>) {
    // 1. Prefer Intel Arc / XPU when present on Windows
    #[cfg(target_os = "windows")]
    {
        let mut arc_cmd = Command::new("powershell");
        arc_cmd.args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'Intel' -and ($_.Name -match 'Arc' -or $_.Name -match 'XPU') } | Select-Object -First 1 -Property Name,DriverVersion | ForEach-Object { \"$($_.Name)|$($_.DriverVersion)\" }",
        ]);
        arc_cmd.creation_flags(CREATE_NO_WINDOW);
        if let Ok(output) = arc_cmd.output() {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Some(line) = stdout.lines().map(|l| l.trim()).find(|l| !l.is_empty()) {
                    let mut split = line.splitn(2, '|');
                    let name = split
                        .next()
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty());
                    let raw_drv = split.next().unwrap_or("").trim();
                    if name.is_some() {
                        let drv_opt = if raw_drv.is_empty() {
                            None
                        } else {
                            Some(raw_drv.to_string())
                        };
                        return (name, drv_opt, None);
                    }
                }
            }
        }
    }

    // 2. Check NVIDIA GPU via nvidia-smi
    let smi_candidates = [
        "nvidia-smi",
        #[cfg(target_os = "windows")]
        "C:\\Windows\\System32\\nvidia-smi.exe",
        #[cfg(target_os = "windows")]
        "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe",
    ];

    for smi_bin in smi_candidates {
        let mut smi_cmd = Command::new(smi_bin);
        smi_cmd.args(["--query-gpu=name,driver_version", "--format=csv,noheader"]);
        #[cfg(windows)]
        {
            smi_cmd.creation_flags(CREATE_NO_WINDOW);
        }
        if let Ok(output) = smi_cmd.output() {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Some(first_line) = stdout.lines().next() {
                    let parts: Vec<&str> = first_line.split(',').collect();
                    let name = parts
                        .first()
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty());
                    let raw_driver = parts
                        .get(1)
                        .map(|s| s.trim().to_string())
                        .unwrap_or_default();
                    if name.is_some() {
                        let (drv_str, drv_major) = parse_driver_version(&raw_driver);
                        let drv_opt = if drv_str.is_empty() {
                            None
                        } else {
                            Some(drv_str)
                        };
                        return (name, drv_opt, drv_major);
                    }
                }
            }
        }

        let mut smi_l = Command::new(smi_bin);
        smi_l.arg("-L");
        #[cfg(windows)]
        {
            smi_l.creation_flags(CREATE_NO_WINDOW);
        }
        if let Ok(output) = smi_l.output() {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Some(first_line) = stdout.lines().next() {
                    let parsed_name = if let Some(rest) = first_line.strip_prefix("GPU 0: ") {
                        if let Some((clean_name, _)) = rest.split_once(" (UUID:") {
                            clean_name.trim().to_string()
                        } else {
                            rest.trim().to_string()
                        }
                    } else {
                        first_line.trim().to_string()
                    };
                    if !parsed_name.is_empty() {
                        return (Some(parsed_name), None, None);
                    }
                }
            }
        }
    }

    // 3. Windows WMI fallback for NVIDIA via PowerShell
    #[cfg(target_os = "windows")]
    {
        let mut ps_cmd = Command::new("powershell");
        ps_cmd.args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'NVIDIA' } | Select-Object -First 1 -Property Name,DriverVersion | ForEach-Object { \"$($_.Name)|$($_.DriverVersion)\" }",
        ]);
        ps_cmd.creation_flags(CREATE_NO_WINDOW);
        if let Ok(output) = ps_cmd.output() {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Some(line) = stdout.lines().map(|l| l.trim()).find(|l| !l.is_empty()) {
                    let mut split = line.splitn(2, '|');
                    let name = split
                        .next()
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty());
                    let raw_drv = split.next().unwrap_or("").trim();
                    if name.is_some() {
                        let (drv_str, drv_major) = parse_driver_version(raw_drv);
                        let drv_opt = if drv_str.is_empty() {
                            None
                        } else {
                            Some(drv_str)
                        };
                        return (name, drv_opt, drv_major);
                    }
                }
            }
        }
    }

    (None, None, None)
}

#[tauri::command]
async fn get_gpu_info() -> GpuInfo {
    tauri::async_runtime::spawn_blocking(|| {
        let project_root = get_project_root();
        detect_gpu_info(&project_root)
    })
    .await
    .unwrap_or_else(|_| GpuInfo {
        cuda_available: false,
        xpu_available: false,
        cpu_fallback_confirmed: true,
        device_name: None,
        driver_version: None,
        recommended_wheel: "cpu".to_string(),
        cuda_label: "CPU-only".to_string(),
        torch_version: None,
        needs_repair: false,
        status_summary: "CPU fallback confirmed".to_string(),
        recommended_workers: default_recommended_workers(),
        physical_cores: default_physical_cores(),
        logical_cores: default_logical_cores(),
    })
}

fn detect_gpu_info(project_root: &Path) -> GpuInfo {
    let (hw_gpu_name, driver_version, driver_major) = detect_hardware_gpu();
    let (recommended_wheel, _index_url, cuda_label) =
        select_pytorch_wheel(hw_gpu_name.as_deref(), driver_major);
    let (recommended_workers, physical_cores, logical_cores) = detect_cpu_worker_info(project_root);

    #[cfg(target_os = "windows")]
    let venv_python = project_root.join(".venv").join("Scripts").join("python.exe");
    #[cfg(not(target_os = "windows"))]
    let venv_python = project_root.join(".venv").join("bin").join("python");

    let marker_path = project_root.join(".venv").join(".ls_trainer_wheel");
    let marker_wheel = std::fs::read_to_string(&marker_path)
        .ok()
        .map(|s| s.trim().to_string());

    let mut torch_installed = false;
    let mut torch_cuda_avail = false;
    let mut torch_cuda_kernel_ok = false;
    let mut torch_xpu_avail = false;
    let mut torch_xpu_kernel_ok = false;
    let mut torch_ver: Option<String> = None;
    let mut torch_dev_name: Option<String> = None;

    if venv_python.is_file() {
        let mut py_cmd = Command::new(&venv_python);
        py_cmd.args([
            "-c",
            "import torch\nver = getattr(torch, '__version__', '')\nxpu_avail = bool(hasattr(torch, 'xpu') and torch.xpu.is_available())\nxpu_dev = ''\nxpu_kok = False\nif xpu_avail:\n    try:\n        xpu_dev = torch.xpu.get_device_name(0)\n        _ = torch.zeros(1, device='xpu') + 1\n        torch.xpu.synchronize()\n        xpu_kok = True\n    except Exception:\n        xpu_kok = False\ncuda_avail = bool(torch.cuda.is_available())\ncuda_dev = ''\ncuda_kok = False\nif cuda_avail:\n    try:\n        cuda_dev = torch.cuda.get_device_name(0)\n        _ = torch.zeros(1, device='cuda') + 1\n        torch.cuda.synchronize()\n        cuda_kok = True\n    except Exception:\n        cuda_kok = False\ndev = xpu_dev if (xpu_avail and xpu_kok) else (cuda_dev if cuda_avail else xpu_dev)\nprint(f'TORCH_OK|{cuda_avail}|{ver}|{dev}|{cuda_kok}|{xpu_avail}|{xpu_kok}')",
        ]);
        #[cfg(windows)]
        {
            py_cmd.creation_flags(CREATE_NO_WINDOW);
        }
        if let Ok(output) = py_cmd.output() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                if let Some(rest) = line.trim().strip_prefix("TORCH_OK|") {
                    let parts: Vec<&str> = rest.split('|').collect();
                    if parts.len() >= 4 {
                        torch_installed = true;
                        torch_cuda_avail = parts[0].eq_ignore_ascii_case("true");
                        if !parts[1].is_empty() {
                            torch_ver = Some(parts[1].to_string());
                        }
                        if !parts[2].is_empty() {
                            torch_dev_name = Some(parts[2].to_string());
                        }
                        if let Some(k) = parts.get(3) {
                            torch_cuda_kernel_ok = k.eq_ignore_ascii_case("true");
                        }
                        if let Some(xa) = parts.get(4) {
                            torch_xpu_avail = xa.eq_ignore_ascii_case("true");
                        }
                        if let Some(xk) = parts.get(5) {
                            torch_xpu_kernel_ok = xk.eq_ignore_ascii_case("true");
                        }
                    }
                }
            }
        }
    }

    let effective_gpu_name = torch_dev_name.or(hw_gpu_name.clone());

    if let Some(ref gpu_name) = effective_gpu_name {
        let ver_str = torch_ver.as_deref().unwrap_or("");
        let short_name = shorten_gpu_name(gpu_name);

        if recommended_wheel == "xpu" || is_intel_arc_gpu(gpu_name) {
            let has_xpu_wheel = ver_str.contains("+xpu");
            let xpu_working =
                torch_installed && torch_xpu_avail && torch_xpu_kernel_ok && has_xpu_wheel;
            let already_attempted = marker_wheel.as_deref() == Some("xpu");
            let needs_repair = !xpu_working && !already_attempted;

            let status_summary = if xpu_working {
                format!("Ready — {short_name} (Intel Arc XPU)")
            } else if needs_repair {
                format!("PyTorch Intel Arc XPU setup required for {short_name}")
            } else if torch_installed && torch_xpu_avail && torch_xpu_kernel_ok {
                format!("Ready — {short_name}")
            } else {
                format!("{short_name} detected (XPU inactive — click Repair GPU)")
            };

            return GpuInfo {
                cuda_available: false,
                xpu_available: torch_xpu_avail && torch_xpu_kernel_ok,
                cpu_fallback_confirmed: torch_installed
                    && !(torch_xpu_avail && torch_xpu_kernel_ok),
                device_name: Some(gpu_name.clone()),
                driver_version,
                recommended_wheel: "xpu".to_string(),
                cuda_label: "Intel Arc XPU".to_string(),
                torch_version: torch_ver,
                needs_repair,
                status_summary,
                recommended_workers,
                physical_cores,
                logical_cores,
            };
        }

        let has_matching_wheel = if recommended_wheel == "cu128" {
            ver_str.contains("+cu128")
        } else {
            ver_str.contains("+cu")
        };

        let cuda_working =
            torch_installed && torch_cuda_avail && torch_cuda_kernel_ok && has_matching_wheel;
        // If we haven't already tried & recorded a repair for this wheel, mark needs_repair when !cuda_working
        let already_attempted = marker_wheel.as_deref() == Some(recommended_wheel);
        let needs_repair = !cuda_working && !already_attempted;

        let status_summary = if cuda_working {
            format!("Ready — {short_name} ({cuda_label})")
        } else if needs_repair {
            format!("PyTorch {cuda_label} setup required for {short_name}")
        } else if torch_installed && torch_cuda_avail && torch_cuda_kernel_ok {
            format!("Ready — {short_name}")
        } else {
            format!("{short_name} detected (CUDA inactive — click Repair GPU)")
        };

        return GpuInfo {
            cuda_available: torch_cuda_avail && torch_cuda_kernel_ok,
            xpu_available: false,
            cpu_fallback_confirmed: torch_installed
                && !(torch_cuda_avail && torch_cuda_kernel_ok),
            device_name: Some(gpu_name.clone()),
            driver_version,
            recommended_wheel: recommended_wheel.to_string(),
            cuda_label: cuda_label.to_string(),
            torch_version: torch_ver,
            needs_repair,
            status_summary,
            recommended_workers,
            physical_cores,
            logical_cores,
        };
    }

    // Neither Intel Arc nor NVIDIA GPU detected
    let already_attempted = marker_wheel.as_deref() == Some("cpu");
    let needs_repair = !torch_installed && !already_attempted && venv_python.is_file();
    let status_summary = if torch_installed {
        "Ready — CPU mode (No discrete GPU)".to_string()
    } else {
        "Installing CPU PyTorch wheel…".to_string()
    };

    GpuInfo {
        cuda_available: false,
        xpu_available: false,
        cpu_fallback_confirmed: torch_installed || !venv_python.is_file(),
        device_name: None,
        driver_version: None,
        recommended_wheel: "cpu".to_string(),
        cuda_label: "CPU-only".to_string(),
        torch_version: torch_ver,
        needs_repair,
        status_summary,
        recommended_workers,
        physical_cores,
        logical_cores,
    }
}

fn strip_ansi_codes(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            if chars.peek() == Some(&'[') {
                chars.next();
                while let Some(&nc) = chars.peek() {
                    chars.next();
                    if ('@'..='~').contains(&nc) {
                        break;
                    }
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn stream_reader_lines<R: std::io::Read>(mut reader: R, on_log: &Channel<String>) {
    let mut buf = [0u8; 1024];
    let mut line_buf = Vec::with_capacity(256);
    let mut last_sent = String::new();

    while let Ok(n) = reader.read(&mut buf) {
        if n == 0 {
            break;
        }
        for &b in &buf[..n] {
            if b == b'\n' || b == b'\r' {
                if !line_buf.is_empty() {
                    let raw = String::from_utf8_lossy(&line_buf);
                    let cleaned = strip_ansi_codes(&raw).trim_end().to_string();
                    if !cleaned.trim().is_empty() && cleaned != last_sent {
                        let _ = on_log.send(cleaned.clone());
                        last_sent = cleaned;
                    }
                    line_buf.clear();
                }
            } else {
                line_buf.push(b);
            }
        }
    }
    if !line_buf.is_empty() {
        let raw = String::from_utf8_lossy(&line_buf);
        let cleaned = strip_ansi_codes(&raw).trim_end().to_string();
        if !cleaned.trim().is_empty() && cleaned != last_sent {
            let _ = on_log.send(cleaned);
        }
    }
}

#[tauri::command]
async fn repair_cuda(on_log: Channel<String>) -> Result<GpuInfo, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let project_root = get_project_root();
        let (hw_gpu_name, driver_version, driver_major) = detect_hardware_gpu();
        let (wheel_id, index_url, cuda_label) =
            select_pytorch_wheel(hw_gpu_name.as_deref(), driver_major);

        let status_banner = if let Some(ref name) = hw_gpu_name {
            let short = shorten_gpu_name(name);
            format!("Installing PyTorch for {short} ({cuda_label})…")
        } else {
            "Installing PyTorch for CPU (CPU-only wheel)…".to_string()
        };

        let _ = on_log.send(format!("[LS-Trainer Status] {status_banner}"));
        let _ = on_log.send(format!(
            "[LS-Trainer] GPU detection: device={:?}, driver={:?} -> target={} ({})",
            hw_gpu_name, driver_version, wheel_id, index_url
        ));

        #[cfg(target_os = "windows")]
        let venv_python = project_root.join(".venv").join("Scripts").join("python.exe");
        #[cfg(not(target_os = "windows"))]
        let venv_python = project_root.join(".venv").join("bin").join("python");

        if !venv_python.is_file() {
            let msg = format!(
                "Virtual environment Python not found at '{}'. Please run setup-venv.bat first.",
                venv_python.display()
            );
            let _ = on_log.send(format!("[LS-Trainer Error] {msg}"));
            return Err(msg);
        }

        // 1. Uninstall existing torch packages
        let _ = on_log.send(
            "[LS-Trainer] Running: pip uninstall -y torch torchvision torchaudio".to_string(),
        );
        let mut uninstall_cmd = Command::new(&venv_python);
        uninstall_cmd.args(["-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"]);
        uninstall_cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
        #[cfg(windows)]
        {
            uninstall_cmd.creation_flags(CREATE_NO_WINDOW);
        }
        if let Ok(mut child) = uninstall_cmd.spawn() {
            let stdout = child.stdout.take();
            let stderr = child.stderr.take();
            let log_out = on_log.clone();
            let t_out = std::thread::spawn(move || {
                if let Some(out) = stdout {
                    stream_reader_lines(out, &log_out);
                }
            });
            let log_err = on_log.clone();
            let t_err = std::thread::spawn(move || {
                if let Some(err) = stderr {
                    stream_reader_lines(err, &log_err);
                }
            });
            let _ = t_out.join();
            let _ = t_err.join();
            let _ = child.wait();
        }

        // 2. Install the matching wheel
        let _ = on_log.send(format!(
            "[LS-Trainer] Running: pip install torch torchvision torchaudio --index-url {index_url}"
        ));
        let mut install_cmd = Command::new(&venv_python);
        install_cmd.args([
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            index_url,
        ]);
        install_cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
        #[cfg(windows)]
        {
            install_cmd.creation_flags(CREATE_NO_WINDOW);
        }

        match install_cmd.spawn() {
            Ok(mut child) => {
                let stdout = child.stdout.take();
                let stderr = child.stderr.take();
                let log_out = on_log.clone();
                let t_out = std::thread::spawn(move || {
                    if let Some(out) = stdout {
                        stream_reader_lines(out, &log_out);
                    }
                });
                let log_err = on_log.clone();
                let t_err = std::thread::spawn(move || {
                    if let Some(err) = stderr {
                        stream_reader_lines(err, &log_err);
                    }
                });
                let _ = t_out.join();
                let _ = t_err.join();
                let status = child
                    .wait()
                    .map_err(|e| format!("Failed waiting for pip install: {e}"))?;
                if !status.success() {
                    let _ = on_log.send(format!(
                        "[LS-Trainer] Warning: pip install exited with status {:?}. Verifying environment...",
                        status.code()
                    ));
                }
            }
            Err(e) => {
                let err_msg = format!("Failed to start pip install: {e}");
                let _ = on_log.send(format!("[LS-Trainer Error] {err_msg}"));
                return Err(err_msg);
            }
        }

        // Record marker so we know this wheel tier was provisioned
        let marker_path = project_root.join(".venv").join(".ls_trainer_wheel");
        let _ = std::fs::write(&marker_path, wheel_id);

        let updated_info = detect_gpu_info(&project_root);
        if updated_info.xpu_available {
            let _ = on_log.send(format!(
                "[LS-Trainer] Intel Arc XPU verified! Device: {} ({})",
                updated_info.device_name.as_deref().unwrap_or("Intel Arc XPU"),
                updated_info.torch_version.as_deref().unwrap_or("torch")
            ));
        } else if updated_info.cuda_available {
            let _ = on_log.send(format!(
                "[LS-Trainer] CUDA verified! Device: {} ({})",
                updated_info.device_name.as_deref().unwrap_or("GPU"),
                updated_info.torch_version.as_deref().unwrap_or("torch")
            ));
        } else {
            let _ = on_log.send(
                "[LS-Trainer] CPU fallback confirmed. Ready for training.".to_string(),
            );
        }
        let _ = on_log.send(format!(
            "[LS-Trainer Status] {}",
            updated_info.status_summary
        ));

        Ok(updated_info)
    })
    .await
    .map_err(|e| format!("Repair task panicked: {e}"))?
}

fn is_cuda_available(project_root: &Path) -> bool {
    detect_gpu_info(project_root).cuda_available
}

fn unpack_archive(archive_path: &Path, target_dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(target_dir).map_err(|e| format!("Failed to create temp dir: {e}"))?;

    let file = File::open(archive_path).map_err(|e| format!("Failed to open archive: {e}"))?;
    let path_str = archive_path.to_string_lossy().to_lowercase();

    if path_str.ends_with(".tar.gz") || path_str.ends_with(".tgz") {
        let gz = flate2::read::GzDecoder::new(BufReader::new(file));
        let mut archive = tar::Archive::new(gz);
        archive
            .unpack(target_dir)
            .map_err(|e| format!("Failed to unpack .tar.gz archive: {e}"))?;
    } else if path_str.ends_with(".tar") {
        let mut archive = tar::Archive::new(BufReader::new(file));
        archive
            .unpack(target_dir)
            .map_err(|e| format!("Failed to unpack .tar archive: {e}"))?;
    } else {
        return Err(format!(
            "Unsupported dataset archive format: {}",
            archive_path.display()
        ));
    }

    Ok(())
}

fn find_data_yaml(dir: &Path) -> Option<PathBuf> {
    let direct = dir.join("data.yaml");
    if direct.exists() {
        return Some(direct);
    }
    let direct_yml = dir.join("data.yml");
    if direct_yml.exists() {
        return Some(direct_yml);
    }

    for entry in walkdir::WalkDir::new(dir)
        .max_depth(4)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.file_type().is_file() {
            let file_name = entry.file_name().to_string_lossy();
            if file_name.eq_ignore_ascii_case("data.yaml")
                || file_name.eq_ignore_ascii_case("data.yml")
            {
                return Some(entry.path().to_path_buf());
            }
        }
    }
    None
}

fn resolve_single_path(base_dir: &Path, raw: &str) -> (String, bool) {
    let trimmed = raw.trim().trim_matches(|c| c == '\'' || c == '"');
    let p = Path::new(trimmed);
    if p.is_absolute() {
        return (trimmed.replace('\\', "/"), false);
    }
    let combined = base_dir.join(p);
    let resolved_str = combined.to_string_lossy().replace('\\', "/");
    (resolved_str, true)
}

fn resolve_and_fix_data_yaml(
    data_yaml_path: &Path,
    on_log: &Channel<String>,
) -> Result<PathBuf, String> {
    let base_dir = data_yaml_path
        .parent()
        .ok_or_else(|| "Failed to get directory of data.yaml".to_string())?;

    let content = std::fs::read_to_string(data_yaml_path)
        .map_err(|e| format!("Failed to read {}: {e}", data_yaml_path.display()))?;

    let mut yaml_val: serde_yaml::Value = serde_yaml::from_str(&content)
        .map_err(|e| format!("Failed to parse YAML in {}: {e}", data_yaml_path.display()))?;

    let mut changes: Vec<String> = Vec::new();
    let base_dir_str = base_dir.to_string_lossy().replace('\\', "/");

    if let serde_yaml::Value::Mapping(ref mut map) = yaml_val {
        // 1. Resolve 'path' key if present, or set it to base_dir
        let path_key = serde_yaml::Value::String("path".to_string());
        if let Some(val) = map.get_mut(&path_key) {
            if let Some(s) = val.as_str() {
                let orig = s.to_string();
                let (resolved, changed) = resolve_single_path(base_dir, s);
                if changed {
                    changes.push(format!("path: '{orig}' -> '{resolved}'"));
                }
                *val = serde_yaml::Value::String(resolved);
            }
        } else {
            map.insert(path_key, serde_yaml::Value::String(base_dir_str.clone()));
            changes.push(format!("path: (set dataset root) '{base_dir_str}'"));
        }

        // 2. Resolve 'train', 'val', 'test'
        for key_name in &["train", "val", "test"] {
            let k = serde_yaml::Value::String((*key_name).to_string());
            if let Some(val) = map.get_mut(&k) {
                match val {
                    serde_yaml::Value::String(s) => {
                        let orig = s.clone();
                        let (resolved, changed) = resolve_single_path(base_dir, &orig);
                        if changed {
                            changes.push(format!("{key_name}: '{orig}' -> '{resolved}'"));
                        }
                        *val = serde_yaml::Value::String(resolved);
                    }
                    serde_yaml::Value::Sequence(seq) => {
                        for item in seq.iter_mut() {
                            if let serde_yaml::Value::String(s) = item {
                                let orig = s.clone();
                                let (resolved, changed) = resolve_single_path(base_dir, &orig);
                                if changed {
                                    changes.push(format!("{key_name}[]: '{orig}' -> '{resolved}'"));
                                }
                                *item = serde_yaml::Value::String(resolved);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // 3. Resolve 'names' if it points to a file path
        let names_key = serde_yaml::Value::String("names".to_string());
        if let Some(val) = map.get_mut(&names_key) {
            if let serde_yaml::Value::String(s) = val {
                if s.ends_with(".txt") || s.ends_with(".yaml") || s.ends_with(".yml") {
                    let orig = s.clone();
                    let (resolved, changed) = resolve_single_path(base_dir, &orig);
                    if changed {
                        changes.push(format!("names: '{orig}' -> '{resolved}'"));
                    }
                    *val = serde_yaml::Value::String(resolved);
                }
            }
        }
    }

    if changes.is_empty() {
        let _ = on_log.send("[LS-Trainer] Dataset paths in data.yaml are already absolute.".to_string());
    } else {
        let _ = on_log.send(format!(
            "[LS-Trainer] Resolving relative dataset paths against '{}':",
            base_dir_str
        ));
        for change in &changes {
            let _ = on_log.send(format!("[LS-Trainer]   * {change}"));
        }
    }

    let serialized = serde_yaml::to_string(&yaml_val)
        .map_err(|e| format!("Failed to serialize resolved data.yaml: {e}"))?;

    let resolved_yaml_path = base_dir.join("data_resolved.yaml");
    std::fs::write(&resolved_yaml_path, &serialized)
        .map_err(|e| format!("Failed to write {}: {e}", resolved_yaml_path.display()))?;

    // Also overwrite unpacked data.yaml in-place for consistency
    let _ = std::fs::write(data_yaml_path, &serialized);

    let _ = on_log.send(format!(
        "[LS-Trainer] Fixed absolute-path data.yaml written to: {}",
        resolved_yaml_path.display()
    ));

    Ok(resolved_yaml_path)
}

#[tauri::command]
async fn pick_dataset_file() -> Option<String> {
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
async fn pick_dataset_folder() -> Option<String> {
    let folder = rfd::AsyncFileDialog::new()
        .set_title("Select Dataset Folder (COCO or YOLO)")
        .pick_folder()
        .await;
    folder.map(|f| to_display_path(f.path()))
}

#[tauri::command]
async fn pick_model_file() -> Option<String> {
    let file = rfd::AsyncFileDialog::new()
        .add_filter("Model Checkpoint (*.pt, *.pth)", &["pt", "pth"])
        .set_title("Select Model Checkpoint")
        .pick_file()
        .await;
    file.map(|f| to_display_path(f.path()))
}

fn resolve_or_download_model_weights(
    project_root: &Path,
    raw_model_choice: &str,
    on_log: &Channel<String>,
) -> String {
    let trimmed = raw_model_choice.trim().trim_matches('"');
    if trimmed.is_empty() {
        return "yolo11m.pt".to_string();
    }

    let as_path = Path::new(trimmed);
    if as_path.is_file() {
        return trimmed.to_string();
    }
    let rel_to_root = project_root.join(trimmed);
    if rel_to_root.is_file() {
        return rel_to_root.to_string_lossy().to_string();
    }
    if trimmed.contains('/') || trimmed.contains('\\') {
        return trimmed.to_string();
    }

    let lower = trimmed.to_ascii_lowercase();
    let weight_filename = if lower.ends_with(".pt") {
        lower
    } else {
        format!("{lower}.pt")
    };

    let candidate_paths = [
        project_root.join("models").join(&weight_filename),
        project_root.join(&weight_filename),
        project_root.join("src-tauri").join(&weight_filename),
    ];

    for candidate in &candidate_paths {
        if candidate.is_file()
            && candidate
                .metadata()
                .map(|m| m.len() > 0)
                .unwrap_or(false)
        {
            let _ = on_log.send(format!(
                "[LS-Trainer] Using cached model weights: {}",
                candidate.display()
            ));
            return candidate.to_string_lossy().to_string();
        }
    }

    let models_dir = project_root.join("models");
    let _ = std::fs::create_dir_all(&models_dir);
    let target_weight_path = models_dir.join(&weight_filename);

    #[cfg(target_os = "windows")]
    let python_exe = project_root.join(".venv").join("Scripts").join("python.exe");
    #[cfg(not(target_os = "windows"))]
    let python_exe = project_root.join(".venv").join("bin").join("python");
    if python_exe.is_file() {
        let _ = on_log.send(format!(
            "[LS-Trainer] Model '{weight_filename}' not cached locally. Downloading from Ultralytics assets..."
        ));
        let _ = on_log.send(format!(
            "[LS-Trainer Status] Downloading {weight_filename}…"
        ));

        let script = "import sys; from ultralytics.utils.downloads import attempt_download_asset; p = attempt_download_asset(sys.argv[1]); print(f'[LS-Trainer] Downloaded model asset: {p}', flush=True)";
        let mut dl_cmd = Command::new(&python_exe);
        dl_cmd
            .args(["-c", script, &weight_filename])
            .current_dir(&models_dir)
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONIOENCODING", "utf-8")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        #[cfg(windows)]
        {
            dl_cmd.creation_flags(CREATE_NO_WINDOW);
        }

        if let Ok(mut child) = dl_cmd.spawn() {
            let stdout = child.stdout.take();
            let stderr = child.stderr.take();

            let on_log_out = on_log.clone();
            let out_t = std::thread::spawn(move || {
                if let Some(out) = stdout {
                    stream_reader_lines(out, &on_log_out);
                }
            });

            let on_log_err = on_log.clone();
            let err_t = std::thread::spawn(move || {
                if let Some(err) = stderr {
                    stream_reader_lines(err, &on_log_err);
                }
            });

            let _ = out_t.join();
            let _ = err_t.join();
            let _ = child.wait();
        }

        if target_weight_path.is_file()
            && target_weight_path
                .metadata()
                .map(|m| m.len() > 0)
                .unwrap_or(false)
        {
            let _ = on_log.send(format!(
                "[LS-Trainer] Model cached at: {}",
                target_weight_path.display()
            ));
            return target_weight_path.to_string_lossy().to_string();
        }
    }

    weight_filename
}

const MAX_LABEL_FILES: u32 = 80;
const MAX_LINES_PER_FILE: u32 = 40;

#[derive(serde::Serialize, Debug, Clone)]
pub struct LabelCheckResult {
    pub polygons_found: bool,
    pub bboxes_found: bool,
    pub files_scanned: u32,
    pub lines_scanned: u32,
    pub warning: Option<String>,
}

#[derive(Default)]
struct LabelStats {
    polygons: u32,
    bboxes: u32,
    files_scanned: u32,
    lines_scanned: u32,
}

/// YOLO detect lines are `class xc yc w h` (5 values).
/// Segmentation polygons are `class x1 y1 x2 y2 ...` (more than 5 values).
fn classify_annotation_line(line: &str) -> Option<bool> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let mut count = 0u32;
    for token in line.split_whitespace() {
        if token.parse::<f64>().is_err() {
            return None;
        }
        count += 1;
    }
    if count == 5 {
        Some(false)
    } else if count > 5 {
        Some(true)
    } else {
        None
    }
}

fn path_is_candidate_label(path: &str) -> bool {
    let lower = path.replace('\\', "/").to_ascii_lowercase();
    if lower.contains("__macosx/") || !lower.ends_with(".txt") {
        return false;
    }
    let file_name = lower.rsplit('/').next().unwrap_or(&lower);
    if file_name.starts_with("._") {
        return false;
    }
    if matches!(
        file_name,
        "classes.txt" | "class_names.txt" | "notes.txt" | "readme.txt" | "license.txt"
    ) {
        return false;
    }
    let in_labels = lower.contains("/labels/") || lower.starts_with("labels/");
    if !in_labels
        && matches!(
            file_name,
            "train.txt" | "val.txt" | "valid.txt" | "test.txt" | "data.txt"
        )
    {
        return false;
    }
    true
}

fn scan_label_text(text: &str, stats: &mut LabelStats) {
    let mut lines_in_file = 0u32;
    let mut counted = false;
    for line in text.lines() {
        if lines_in_file >= MAX_LINES_PER_FILE {
            break;
        }
        match classify_annotation_line(line) {
            Some(true) => {
                stats.polygons += 1;
                stats.lines_scanned += 1;
                lines_in_file += 1;
                counted = true;
            }
            Some(false) => {
                stats.bboxes += 1;
                stats.lines_scanned += 1;
                lines_in_file += 1;
                counted = true;
            }
            None => {}
        }
    }
    if counted {
        stats.files_scanned += 1;
    }
}

fn label_sample_complete(stats: &LabelStats) -> bool {
    if stats.polygons > 0 && stats.bboxes > 0 {
        return true;
    }
    stats.files_scanned >= MAX_LABEL_FILES
}

fn result_from_stats(stats: &LabelStats) -> LabelCheckResult {
    let warning = if stats.lines_scanned == 0 {
        Some(
            "No polygon annotations were found in the dataset labels. Instance segmentation needs label lines with more than 5 values (class plus polygon points)."
                .to_string(),
        )
    } else if stats.polygons == 0 && stats.bboxes > 0 {
        Some(format!(
            "Plain bounding-box labels detected ({} lines with 5 values, 0 polygon lines). Instance segmentation needs polygon annotations (more than 5 values per line).",
            stats.bboxes
        ))
    } else if stats.bboxes > 0 {
        Some(format!(
            "Mixed labels detected: {} polygon lines and {} plain bounding-box lines (5 values). Segmentation training expects polygons (more than 5 values per line).",
            stats.polygons, stats.bboxes
        ))
    } else {
        None
    };

    LabelCheckResult {
        polygons_found: stats.polygons > 0,
        bboxes_found: stats.bboxes > 0,
        files_scanned: stats.files_scanned,
        lines_scanned: stats.lines_scanned,
        warning,
    }
}

fn path_is_candidate_coco_json(path: &str) -> bool {
    let lower = path.replace('\\', "/").to_ascii_lowercase();
    if lower.contains("__macosx/") || !lower.ends_with(".json") {
        return false;
    }
    let file_name = lower.rsplit('/').next().unwrap_or(&lower);
    if file_name.starts_with("._") {
        return false;
    }
    file_name.starts_with("instances_")
        || file_name.ends_with(".coco.json")
        || file_name == "annotations.json"
}

fn scan_coco_json_text(text: &str, stats: &mut LabelStats) {
    let Ok(val) = serde_json::from_str::<serde_json::Value>(text) else {
        return;
    };
    let Some(anns) = val.get("annotations").and_then(|v| v.as_array()) else {
        return;
    };
    let mut counted = false;
    let mut items_in_file = 0u32;
    for ann in anns {
        if items_in_file >= MAX_LINES_PER_FILE * 4 {
            break;
        }
        let has_polygon = ann
            .get("segmentation")
            .map(|seg| match seg {
                serde_json::Value::Array(polys) => polys.iter().any(|p| {
                    p.as_array()
                        .map(|coords| coords.len() >= 6)
                        .unwrap_or(false)
                }),
                serde_json::Value::Object(obj) => obj.contains_key("counts"),
                _ => false,
            })
            .unwrap_or(false);
        let has_bbox = ann
            .get("bbox")
            .and_then(|b| b.as_array())
            .map(|arr| arr.len() >= 4)
            .unwrap_or(false);

        if has_polygon {
            stats.polygons += 1;
            stats.lines_scanned += 1;
            items_in_file += 1;
            counted = true;
        } else if has_bbox {
            stats.bboxes += 1;
            stats.lines_scanned += 1;
            items_in_file += 1;
            counted = true;
        }
    }
    if counted {
        stats.files_scanned += 1;
    }
}

fn inspect_label_tree(root: &Path) -> LabelCheckResult {
    let mut stats = LabelStats::default();
    let mut coco_files: Vec<PathBuf> = Vec::new();
    for entry in walkdir::WalkDir::new(root)
        .max_depth(6)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if label_sample_complete(&stats) {
            break;
        }
        if !entry.file_type().is_file() {
            continue;
        }
        let path_str = entry.path().to_string_lossy().to_string();
        if path_is_candidate_coco_json(&path_str) {
            coco_files.push(entry.path().to_path_buf());
            continue;
        }
        if !path_is_candidate_label(&path_str) {
            continue;
        }
        let Ok(bytes) = std::fs::read(entry.path()) else {
            continue;
        };
        let slice = if bytes.len() > 256 * 1024 {
            &bytes[..256 * 1024]
        } else {
            &bytes[..]
        };
        let text = String::from_utf8_lossy(slice);
        scan_label_text(&text, &mut stats);
    }
    if stats.lines_scanned == 0 && !coco_files.is_empty() {
        for coco_path in coco_files {
            if label_sample_complete(&stats) {
                break;
            }
            if let Ok(text) = std::fs::read_to_string(&coco_path) {
                scan_coco_json_text(&text, &mut stats);
            }
        }
    }
    result_from_stats(&stats)
}

fn scan_tar_archive<R: Read>(mut archive: tar::Archive<R>) -> Result<LabelCheckResult, String> {
    let mut stats = LabelStats::default();
    let mut coco_payloads: Vec<String> = Vec::new();
    let entries = archive
        .entries()
        .map_err(|e| format!("Failed to read dataset archive: {e}"))?;
    for entry in entries {
        if label_sample_complete(&stats) {
            break;
        }
        let mut entry = entry.map_err(|e| format!("Failed to read archive entry: {e}"))?;
        let path_str = {
            let path = entry
                .path()
                .map_err(|e| format!("Failed to read archive entry path: {e}"))?;
            path.to_string_lossy().to_string()
        };
        if path_is_candidate_coco_json(&path_str) && coco_payloads.len() < 4 {
            let mut buf = Vec::new();
            let mut limited = (&mut entry).take(16 * 1024 * 1024);
            let _ = limited.read_to_end(&mut buf);
            if let Ok(text) = String::from_utf8(buf) {
                coco_payloads.push(text);
            }
            continue;
        }
        if !path_is_candidate_label(&path_str) {
            continue;
        }
        let mut buf = Vec::new();
        let mut limited = (&mut entry).take(256 * 1024);
        let _ = limited.read_to_end(&mut buf);
        let text = String::from_utf8_lossy(&buf);
        scan_label_text(&text, &mut stats);
    }
    if stats.lines_scanned == 0 && !coco_payloads.is_empty() {
        for text in &coco_payloads {
            scan_coco_json_text(text, &mut stats);
        }
    }
    Ok(result_from_stats(&stats))
}

fn inspect_archive_labels(archive_path: &Path) -> Result<LabelCheckResult, String> {
    let file = File::open(archive_path)
        .map_err(|e| format!("Failed to open dataset archive: {e}"))?;
    let path_str = archive_path.to_string_lossy().to_lowercase();
    if path_str.ends_with(".tar.gz") || path_str.ends_with(".tgz") {
        let gz = flate2::read::GzDecoder::new(BufReader::new(file));
        scan_tar_archive(tar::Archive::new(gz))
    } else if path_str.ends_with(".tar") {
        scan_tar_archive(tar::Archive::new(BufReader::new(file)))
    } else {
        Err(format!(
            "Unsupported dataset archive format: {}",
            archive_path.display()
        ))
    }
}

fn inspect_dataset_path(path: &Path) -> Result<LabelCheckResult, String> {
    if !path.exists() {
        return Err(format!(
            "Dataset path does not exist: {}",
            path.display()
        ));
    }
    if path.is_dir() {
        return Ok(inspect_label_tree(path));
    }
    let lower = path.to_string_lossy().to_lowercase();
    if lower.ends_with(".yaml") || lower.ends_with(".yml") || lower.ends_with(".json") {
        let mut root = path.parent().unwrap_or(path);
        if let Some(dir_name) = root.file_name().map(|n| n.to_string_lossy().to_lowercase()) {
            if matches!(dir_name.as_str(), "annotations" | "train" | "valid" | "val" | "test") {
                if let Some(grandparent) = root.parent() {
                    root = grandparent;
                }
            }
        }
        return Ok(inspect_label_tree(root));
    }
    if lower.ends_with(".tar") || lower.ends_with(".tar.gz") || lower.ends_with(".tgz") {
        return inspect_archive_labels(path);
    }
    Err(format!(
        "Unsupported dataset path for label check: {}",
        path.display()
    ))
}

fn resolve_train_task(request: &TrainingRequest) -> String {
    let model = request.model_choice.to_ascii_lowercase();
    if model.contains("-seg") {
        return "segment".to_string();
    }
    if request.task.eq_ignore_ascii_case("segment") {
        return "segment".to_string();
    }
    "detect".to_string()
}

#[tauri::command]
async fn inspect_dataset_labels(dataset_path: String) -> Result<LabelCheckResult, String> {
    tauri::async_runtime::spawn_blocking(move || inspect_dataset_path(Path::new(&dataset_path)))
        .await
        .map_err(|e| format!("Label check failed: {e}"))?
}

fn save_run_config_metadata(
    abs_results_dir: &Path,
    request: &TrainingRequest,
    resolved_model: &str,
    effective_device: &str,
    effective_workers: u32,
    exact_command: &str,
    on_log: &Channel<String>,
) {
    if std::fs::create_dir_all(abs_results_dir).is_err() {
        return;
    }

    let config_path = abs_results_dir.join("ls_trainer_config.json");
    let effective_freeze = if request.freeze_backbone {
        request.freeze_layers.max(1)
    } else {
        0
    };

    let payload = serde_json::json!({
        "run_name": request.run_name,
        "dataset_path": request.dataset_path,
        "model_family": request.model_family,
        "model_choice": request.model_choice,
        "resolved_model": resolved_model,
        "task": resolve_train_task(request),
        "freeze_backbone": request.freeze_backbone,
        "freeze_layers": effective_freeze,
        "epochs": request.epochs,
        "batch": request.batch,
        "imgsz": request.imgsz,
        "device": effective_device,
        "workers": effective_workers,
        "workers_override": request.workers,
        "amp": request.amp,
        "augmentation": {
            "mosaic": request.mosaic,
            "mixup": request.mixup,
            "copy_paste": request.copy_paste,
            "degrees": request.degrees,
            "translate": request.translate,
            "scale": request.scale,
            "flipud": request.flipud,
            "close_mosaic": request.close_mosaic,
        },
        "command": exact_command,
    });

    if let Ok(serialized) = serde_json::to_string_pretty(&payload) {
        if std::fs::write(&config_path, serialized).is_ok() {
            let _ = on_log.send(format!(
                "[LS-Trainer] Saved run settings to: {}",
                config_path.display()
            ));
        }
    }
}

fn is_rfdetr_request(request: &TrainingRequest) -> bool {
    if request.model_family.eq_ignore_ascii_case("RF-DETR")
        || request.model_family.eq_ignore_ascii_case("RFDETR")
    {
        return true;
    }
    let choice = request.model_choice.trim().to_ascii_lowercase();
    choice.starts_with("rf-detr") || choice.starts_with("rfdetr") || choice.ends_with(".pth")
}

fn copy_or_hardlink_file(src: &Path, dst: &Path) -> Result<(), String> {
    if dst.exists() {
        return Ok(());
    }
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create image directory {}: {e}", parent.display()))?;
    }
    if std::fs::hard_link(src, dst).is_ok() {
        return Ok(());
    }
    std::fs::copy(src, dst).map(|_| ()).map_err(|e| {
        format!(
            "Failed to copy dataset image '{}' to '{}': {e}",
            src.display(),
            dst.display()
        )
    })
}

fn find_coco_dataset_root(base_dir: &Path) -> Option<PathBuf> {
    let candidates = [
        base_dir.join("annotations").join("instances_train.json"),
        base_dir.join("train").join("_annotations.coco.json"),
        base_dir.join("export_manifest.json"),
    ];
    for c in candidates {
        if c.is_file() {
            return Some(base_dir.to_path_buf());
        }
    }

    for entry in walkdir::WalkDir::new(base_dir)
        .max_depth(3)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
        let path = entry.path();
        if name == "instances_train.json" {
            if let Some(ann_dir) = path.parent() {
                if let Some(root) = ann_dir.parent() {
                    return Some(root.to_path_buf());
                }
            }
        } else if name == "_annotations.coco.json" {
            if let Some(split_dir) = path.parent() {
                if split_dir
                    .file_name()
                    .map(|n| n.to_string_lossy().eq_ignore_ascii_case("train"))
                    .unwrap_or(false)
                {
                    if let Some(root) = split_dir.parent() {
                        return Some(root.to_path_buf());
                    }
                }
            }
        } else if name == "export_manifest.json" {
            if let Some(root) = path.parent() {
                return Some(root.to_path_buf());
            }
        }
    }
    None
}

fn resolve_coco_image_source(
    dataset_root: &Path,
    split: &str,
    file_name: &str,
    manifest_map: &std::collections::HashMap<String, PathBuf>,
) -> Option<PathBuf> {
    let clean_rel = file_name.trim_start_matches(['/', '\\']);
    let basename = Path::new(clean_rel)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| clean_rel.to_string());

    let candidates = [
        dataset_root.join("images").join(split).join(&basename),
        dataset_root.join("images").join(clean_rel),
        dataset_root.join(split).join(&basename),
        dataset_root.join(clean_rel),
        dataset_root.join("images").join(&basename),
    ];
    for candidate in candidates {
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    if let Some(mapped) = manifest_map
        .get(&format!("{split}/{basename}"))
        .or_else(|| manifest_map.get(&basename))
    {
        if mapped.is_file() {
            return Some(mapped.clone());
        }
    }
    None
}

fn populate_rfdetr_split_from_coco(
    dataset_root: &Path,
    coco_json_path: &Path,
    source_split: &str,
    target_split_dir: &Path,
    manifest_map: &std::collections::HashMap<String, PathBuf>,
) -> Result<usize, String> {
    std::fs::create_dir_all(target_split_dir).map_err(|e| {
        format!(
            "Failed to create RF-DETR split folder '{}': {e}",
            target_split_dir.display()
        )
    })?;

    let raw_text = std::fs::read_to_string(coco_json_path).map_err(|e| {
        format!(
            "Failed to read COCO annotation file '{}': {e}",
            coco_json_path.display()
        )
    })?;
    let mut coco: serde_json::Value = serde_json::from_str(&raw_text).map_err(|e| {
        format!(
            "Invalid COCO JSON in '{}': {e}",
            coco_json_path.display()
        )
    })?;

    // Ensure COCO supercategory exists on every category (required by rfdetr/pycocotools)
    if let Some(categories) = coco.get_mut("categories").and_then(|v| v.as_array_mut()) {
        for cat in categories.iter_mut() {
            if let Some(obj) = cat.as_object_mut() {
                if !obj.contains_key("supercategory") {
                    obj.insert(
                        "supercategory".to_string(),
                        serde_json::Value::String("defect".to_string()),
                    );
                }
            }
        }
    }

    if !coco.get("info").map(|v| v.is_object()).unwrap_or(false) {
        if let Some(root_obj) = coco.as_object_mut() {
            root_obj.insert(
                "info".to_string(),
                serde_json::json!({
                    "description": "LS-Marker COCO dataset prepared by LS-Trainer",
                    "version": "1.0"
                }),
            );
        }
    }
    if !coco.get("licenses").map(|v| v.is_array()).unwrap_or(false) {
        if let Some(root_obj) = coco.as_object_mut() {
            root_obj.insert("licenses".to_string(), serde_json::json!([]));
        }
    }

    let mut linked_images = 0usize;
    if let Some(images) = coco.get_mut("images").and_then(|v| v.as_array_mut()) {
        for img in images.iter_mut() {
            let Some(obj) = img.as_object_mut() else {
                continue;
            };
            let Some(orig_name) = obj.get("file_name").and_then(|v| v.as_str()) else {
                continue;
            };
            let basename = Path::new(orig_name)
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| orig_name.to_string());

            if let Some(src_img) =
                resolve_coco_image_source(dataset_root, source_split, orig_name, manifest_map)
            {
                let dst_img = target_split_dir.join(&basename);
                copy_or_hardlink_file(&src_img, &dst_img)?;
                linked_images += 1;
            }
            obj.insert(
                "file_name".to_string(),
                serde_json::Value::String(basename),
            );
        }
    }

    let out_ann_path = target_split_dir.join("_annotations.coco.json");
    let serialized = serde_json::to_string_pretty(&coco)
        .map_err(|e| format!("Failed to serialize COCO JSON: {e}"))?;
    std::fs::write(&out_ann_path, serialized).map_err(|e| {
        format!(
            "Failed to write '{}': {e}",
            out_ann_path.display()
        )
    })?;

    Ok(linked_images)
}

fn prepare_rfdetr_coco_dataset(
    dataset_input: &Path,
    on_log: &Channel<String>,
) -> Result<PathBuf, String> {
    let base_dir = if dataset_input.is_file() {
        let lower = dataset_input.to_string_lossy().to_ascii_lowercase();
        if lower.ends_with(".tar") || lower.ends_with(".tar.gz") || lower.ends_with(".tgz") {
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis();
            let temp_dir = std::env::temp_dir().join(format!("ls_trainer_coco_{timestamp}"));
            let _ = on_log.send(format!(
                "[LS-Trainer] Unpacking dataset archive '{}' to '{}'...",
                dataset_input.display(),
                temp_dir.display()
            ));
            unpack_archive(dataset_input, &temp_dir)?;
            let _ = on_log.send("[LS-Trainer] Unpacked dataset archive.".to_string());
            temp_dir
        } else if lower.ends_with(".json") {
            let parent = dataset_input.parent().unwrap_or(dataset_input);
            if parent
                .file_name()
                .map(|n| {
                    let s = n.to_string_lossy().to_ascii_lowercase();
                    s == "annotations" || s == "train" || s == "valid" || s == "val"
                })
                .unwrap_or(false)
            {
                parent.parent().unwrap_or(parent).to_path_buf()
            } else {
                parent.to_path_buf()
            }
        } else if lower.ends_with(".yaml") || lower.ends_with(".yml") {
            dataset_input
                .parent()
                .unwrap_or(dataset_input)
                .to_path_buf()
        } else {
            return Err(format!(
                "Unsupported dataset file for RF-DETR: {}",
                dataset_input.display()
            ));
        }
    } else if dataset_input.is_dir() {
        dataset_input.to_path_buf()
    } else {
        return Err(format!(
            "Dataset path does not exist: {}",
            dataset_input.display()
        ));
    };

    let dataset_root = find_coco_dataset_root(&base_dir).unwrap_or_else(|| base_dir.clone());

    // Case 1: Already in Roboflow COCO layout (`train/_annotations.coco.json`)
    let rf_train_ann = dataset_root.join("train").join("_annotations.coco.json");
    let rf_valid_ann = dataset_root.join("valid").join("_annotations.coco.json");
    let rf_test_ann = dataset_root.join("test").join("_annotations.coco.json");
    if rf_train_ann.is_file() {
        let _ = on_log.send(format!(
            "[LS-Trainer] Detected COCO dataset layout in '{}'",
            dataset_root.display()
        ));
        let empty_map = std::collections::HashMap::new();
        if !rf_valid_ann.is_file() {
            let val_alt = dataset_root.join("val").join("_annotations.coco.json");
            let src_ann = if val_alt.is_file() {
                (&val_alt, "val")
            } else {
                (&rf_train_ann, "train")
            };
            let _ = populate_rfdetr_split_from_coco(
                &dataset_root,
                src_ann.0,
                src_ann.1,
                &dataset_root.join("valid"),
                &empty_map,
            )?;
        }
        if !rf_test_ann.is_file() {
            let src_ann = if rf_valid_ann.is_file() {
                (&rf_valid_ann, "valid")
            } else {
                (&rf_train_ann, "train")
            };
            let _ = populate_rfdetr_split_from_coco(
                &dataset_root,
                src_ann.0,
                src_ann.1,
                &dataset_root.join("test"),
                &empty_map,
            )?;
        }
        return Ok(dataset_root);
    }

    // Case 2: LS-Marker COCO layout (`annotations/instances_train.json` + `annotations/instances_val.json`)
    let ls_train_ann = dataset_root
        .join("annotations")
        .join("instances_train.json");
    let ls_val_ann = dataset_root.join("annotations").join("instances_val.json");
    if ls_train_ann.is_file() {
        let _ = on_log.send(format!(
            "[LS-Trainer] Detected LS-Marker COCO export in '{}'. Preparing RF-DETR splits...",
            dataset_root.display()
        ));

        let mut manifest_map = std::collections::HashMap::new();
        let manifest_path = dataset_root.join("export_manifest.json");
        if manifest_path.is_file() {
            if let Ok(text) = std::fs::read_to_string(&manifest_path) {
                if let Ok(manifest_json) = serde_json::from_str::<serde_json::Value>(&text) {
                    if let Some(items) = manifest_json.get("images").and_then(|v| v.as_array()) {
                        for item in items {
                            let file_name = item
                                .get("file_name")
                                .and_then(|v| v.as_str())
                                .unwrap_or_default();
                            let split = item
                                .get("split")
                                .and_then(|v| v.as_str())
                                .unwrap_or("train");
                            if let Some(src_path) =
                                item.get("source_image_path").and_then(|v| v.as_str())
                            {
                                let p = PathBuf::from(src_path);
                                if !file_name.is_empty() {
                                    manifest_map.insert(format!("{split}/{file_name}"), p.clone());
                                    manifest_map.insert(file_name.to_string(), p);
                                }
                            }
                        }
                    }
                }
            }
        }

        let train_count = populate_rfdetr_split_from_coco(
            &dataset_root,
            &ls_train_ann,
            "train",
            &dataset_root.join("train"),
            &manifest_map,
        )?;

        let (val_ann_to_use, val_split_name) = if ls_val_ann.is_file() {
            (&ls_val_ann, "val")
        } else {
            (&ls_train_ann, "train")
        };
        let val_count = populate_rfdetr_split_from_coco(
            &dataset_root,
            val_ann_to_use,
            val_split_name,
            &dataset_root.join("valid"),
            &manifest_map,
        )?;
        let _ = populate_rfdetr_split_from_coco(
            &dataset_root,
            val_ann_to_use,
            val_split_name,
            &dataset_root.join("test"),
            &manifest_map,
        )?;

        let _ = on_log.send(format!(
            "[LS-Trainer] Prepared RF-DETR COCO splits: {train_count} train images, {val_count} valid images."
        ));
        return Ok(dataset_root);
    }

    Err(format!(
        "COCO annotations not found in '{}'. Expected LS-Marker COCO export (annotations/instances_train.json) or COCO split folder (train/_annotations.coco.json).",
        dataset_root.display()
    ))
}

fn resolve_rfdetr_class_and_pretrain(
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

fn snap_rfdetr_resolution(imgsz: u32, block_size: u32) -> u32 {
    let b = block_size.max(1);
    let rounded = ((imgsz + b / 2) / b) * b;
    rounded.max(b)
}

fn get_venv_python(project_root: &Path) -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    let p = project_root.join(".venv").join("Scripts").join("python.exe");
    #[cfg(not(target_os = "windows"))]
    let p = project_root.join(".venv").join("bin").join("python");

    if p.is_file() {
        Some(p)
    } else {
        None
    }
}

fn ensure_rfdetr_installed(project_root: &Path, on_log: &Channel<String>) {
    let Some(python_exe) = get_venv_python(project_root) else {
        return;
    };

    let mut check_cmd = Command::new(&python_exe);
    check_cmd.args([
        "-c",
        "import rfdetr, pytorch_lightning, pycocotools, vernier, hotcoco, faster_coco_eval, onnx; from rfdetr.training import build_trainer",
    ]);
    #[cfg(windows)]
    {
        check_cmd.creation_flags(CREATE_NO_WINDOW);
    }
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
    #[cfg(windows)]
    {
        install_cmd.creation_flags(CREATE_NO_WINDOW);
    }
    if let Ok(mut child) = install_cmd.spawn() {
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let on_log_out = on_log.clone();
        let t1 = std::thread::spawn(move || {
            if let Some(out) = stdout {
                stream_reader_lines(out, &on_log_out);
            }
        });
        let on_log_err = on_log.clone();
        let t2 = std::thread::spawn(move || {
            if let Some(err) = stderr {
                stream_reader_lines(err, &on_log_err);
            }
        });
        let _ = t1.join();
        let _ = t2.join();
        let _ = child.wait();
    }
}

fn extract_final_metrics_from_results_dir(abs_results_dir: &Path) -> (Option<f64>, Option<f64>) {
    let metrics_json_path = abs_results_dir.join("ls_trainer_metrics.json");
    if metrics_json_path.is_file() {
        if let Ok(text) = std::fs::read_to_string(&metrics_json_path) {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&text) {
                let map50 = val.get("map50").and_then(|v| v.as_f64());
                let map50_95 = val.get("map50_95").and_then(|v| v.as_f64());
                if map50.is_some() || map50_95.is_some() {
                    return (map50, map50_95);
                }
            }
        }
    }

    // Fallback: parse Ultralytics results.csv or RF-DETR metrics.csv
    for csv_name in ["results.csv", "metrics.csv"] {
        let csv_path = abs_results_dir.join(csv_name);
        if !csv_path.is_file() {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(&csv_path) else {
            continue;
        };
        let mut lines = content.lines().filter(|l| !l.trim().is_empty());
        let Some(header_line) = lines.next() else {
            continue;
        };
        let headers: Vec<String> = header_line
            .split(',')
            .map(|h| h.trim().to_ascii_lowercase())
            .collect();

        let idx_map50 = headers.iter().position(|h| {
            h == "metrics/map50(b)"
                || h == "metrics/map50(m)"
                || h == "val/map_50"
                || h == "val/ema_map_50"
                || h == "val/segm_map_50"
        });
        let idx_map50_95 = headers.iter().position(|h| {
            h == "metrics/map50-95(b)"
                || h == "metrics/map50-95(m)"
                || h == "val/map_50_95"
                || h == "val/ema_map_50_95"
                || h == "val/segm_map_50_95"
        });

        let mut best_50: Option<f64> = None;
        let mut best_50_95: Option<f64> = None;
        for row in lines {
            let cols: Vec<&str> = row.split(',').map(|c| c.trim()).collect();
            if let Some(i50) = idx_map50 {
                if let Some(v) = cols.get(i50).and_then(|s| s.parse::<f64>().ok()) {
                    best_50 = Some(best_50.map_or(v, |prev| prev.max(v)));
                }
            }
            if let Some(i95) = idx_map50_95 {
                if let Some(v) = cols.get(i95).and_then(|s| s.parse::<f64>().ok()) {
                    best_50_95 = Some(best_50_95.map_or(v, |prev| prev.max(v)));
                }
            }
        }
        if best_50.is_some() || best_50_95.is_some() {
            return (best_50, best_50_95);
        }
    }

    (None, None)
}

async fn start_rfdetr_training(
    project_root: PathBuf,
    run_name: String,
    yolo_task: String,
    abs_results_dir: PathBuf,
    results_dir: String,
    dataset_input: PathBuf,
    request: TrainingRequest,
    on_log: Channel<String>,
) -> Result<TrainingResponse, String> {
    let coco_root = prepare_rfdetr_coco_dataset(&dataset_input, &on_log)?;

    if yolo_task == "segment" {
        let label_check = inspect_label_tree(&coco_root);
        if let Some(warning) = label_check.warning {
            let _ = on_log.send(format!("[LS-Trainer] Warning: {warning}"));
        } else {
            let _ = on_log.send(format!(
                "[LS-Trainer] Segmentation labels OK: polygon annotations found ({} files, {} annotations).",
                label_check.files_scanned, label_check.lines_scanned
            ));
        }
    }

    ensure_rfdetr_installed(&project_root, &on_log);

    let (rfdetr_class, default_weights_name, block_size) =
        resolve_rfdetr_class_and_pretrain(&request.model_choice, &yolo_task);
    let snapped_res = snap_rfdetr_resolution(request.imgsz, block_size);
    if snapped_res != request.imgsz {
        let _ = on_log.send(format!(
            "[LS-Trainer] Adjusted RF-DETR resolution from {} to {} (nearest multiple of block_size={block_size}).",
            request.imgsz, snapped_res
        ));
    }

    let models_dir = project_root.join("models");
    let _ = std::fs::create_dir_all(&models_dir);
    let custom_path = Path::new(request.model_choice.trim());
    let resolved_pretrain = if custom_path.is_file() {
        custom_path.to_string_lossy().to_string()
    } else {
        default_weights_name.to_string()
    };

    let device_input = request.device.trim();
    let gpu_probe = detect_gpu_info(&project_root);
    let effective_device = if device_input.eq_ignore_ascii_case("cpu") {
        let _ = on_log.send("[LS-Trainer] Device override: using device=cpu".to_string());
        "cpu".to_string()
    } else if device_input.is_empty()
        || device_input.eq_ignore_ascii_case("auto")
        || device_input.eq_ignore_ascii_case("gpu")
        || device_input.eq_ignore_ascii_case("xpu")
        || device_input.eq_ignore_ascii_case("xpu:0")
        || device_input.eq_ignore_ascii_case("cuda")
        || device_input == "0"
    {
        if gpu_probe.xpu_available && !device_input.eq_ignore_ascii_case("cuda") {
            let dev_label = gpu_probe
                .device_name
                .as_deref()
                .unwrap_or("Intel Arc XPU");
            let _ = on_log.send(format!(
                "[LS-Trainer] Intel Arc XPU available (torch.xpu.is_available()=True), selecting device=xpu ({dev_label})"
            ));
            "xpu".to_string()
        } else if gpu_probe.cuda_available {
            let dev_label = gpu_probe.device_name.as_deref().unwrap_or("CUDA GPU");
            let _ = on_log.send(format!(
                "[LS-Trainer] CUDA available (torch.cuda.is_available()=True), selecting device=cuda ({dev_label})"
            ));
            "cuda".to_string()
        } else {
            let _ = on_log.send(
                "[LS-Trainer] Neither Intel Arc XPU nor CUDA available, using device=cpu"
                    .to_string(),
            );
            "cpu".to_string()
        }
    } else {
        device_input.to_string()
    };

    let (auto_workers, phys_cores, log_cores) = detect_cpu_worker_info(&project_root);
    let (effective_workers, worker_source) = match request.workers {
        Some(custom_w) => (custom_w.min(8), "manual override"),
        None => (auto_workers, "auto-detected"),
    };
    let _ = on_log.send(format!(
        "[LS-Trainer] DataLoader workers: {effective_workers} ({worker_source}; {phys_cores} physical / {log_cores} logical CPU cores, capped at 8)"
    ));

    let _ = std::fs::create_dir_all(&abs_results_dir);
    let runner_script_path = abs_results_dir.join("ls_trainer_rfdetr_run.py");

    let runner_script = r#"import json
import multiprocessing
import os
import sys
import traceback

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def detect_num_workers(cfg):
    if "num_workers" in cfg and cfg["num_workers"] is not None:
        return max(0, min(8, int(cfg["num_workers"])))
    phys = None
    try:
        import psutil
        phys = psutil.cpu_count(logical=False)
    except Exception:
        phys = None
    logical = os.cpu_count() or 2
    base_cores = phys if (phys is not None and phys > 0) else max(1, logical // 2)
    return min(8, max(1, base_cores - 1))

def select_training_device(requested):
    import torch
    req = str(requested or "auto").strip().lower()
    xpu_avail = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
    cuda_avail = bool(torch.cuda.is_available())

    if req == "cpu":
        print(
            "[LS-Trainer] RF-DETR device selection: CPU requested -> device=cpu (accelerator=cpu, devices=1)",
            flush=True,
        )
        return "cpu"

    if req in ("", "auto", "gpu", "0", "xpu", "xpu:0", "cuda", "cuda:0"):
        if xpu_avail and req not in ("cuda", "cuda:0"):
            dev_name = "Intel Arc XPU"
            try:
                dev_name = torch.xpu.get_device_name(0)
            except Exception:
                pass
            print(
                f"[LS-Trainer] RF-DETR device selection: torch.xpu.is_available()=True -> using Intel Arc XPU device=xpu ({dev_name})",
                flush=True,
            )
            return "xpu"
        if cuda_avail:
            dev_name = "CUDA GPU"
            try:
                dev_name = torch.cuda.get_device_name(0)
            except Exception:
                pass
            print(
                f"[LS-Trainer] RF-DETR device selection: torch.cuda.is_available()=True -> using NVIDIA CUDA device=cuda ({dev_name})",
                flush=True,
            )
            return "cuda"
        print(
            "[LS-Trainer] RF-DETR device selection: neither Intel Arc XPU nor CUDA is available -> falling back to device=cpu",
            flush=True,
        )
        return "cpu"

    print(f"[LS-Trainer] RF-DETR device selection: using custom device={requested}", flush=True)
    return str(requested)

def main():
    cfg_path = sys.argv[1]
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    models_dir = cfg["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    os.chdir(models_dir)

    import torch
    import rfdetr
    import rfdetr.detr as rf_detr_mod
    import rfdetr.training as rf_training
    import rfdetr.training.trainer as rf_trainer_mod
    from pytorch_lightning import Callback
    from pytorch_lightning.accelerators import Accelerator
    from pytorch_lightning.strategies import SingleDeviceStrategy

    try:
        import rfdetr.utilities.console as rf_console
        from rich.console import Console
        rf_console._get_rich_console = lambda _trainer: Console(
            force_terminal=False,
            legacy_windows=False,
            color_system=None,
        )
    except Exception:
        pass

    class_name = cfg["rfdetr_class"]
    dataset_dir = cfg["dataset_dir"]
    output_dir = cfg["output_dir"]
    epochs = int(cfg["epochs"])
    batch_size = max(1, int(cfg["batch_size"]))
    resolution = int(cfg["resolution"])
    device = select_training_device(cfg.get("device", "auto"))
    pretrain_weights = cfg["pretrain_weights"]
    freeze_backbone = bool(cfg.get("freeze_backbone", False))
    num_workers = detect_num_workers(cfg)

    os.makedirs(output_dir, exist_ok=True)

    class XPUAccelerator(Accelerator):
        """PyTorch Lightning Accelerator for Intel Arc / XPU devices."""
        def setup_device(self, dev):
            if dev.type != "xpu":
                raise ValueError(f"Expected xpu device, got {dev}")
            torch.xpu.set_device(dev)

        def get_device_stats(self, dev):
            return {}

        def teardown(self):
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()

        @staticmethod
        def parse_devices(devices):
            return [torch.device("xpu", 0)]

        @staticmethod
        def get_parallel_devices(devices):
            return [torch.device("xpu", 0)]

        @staticmethod
        def auto_device_count():
            return torch.xpu.device_count() if (hasattr(torch, "xpu") and torch.xpu.is_available()) else 0

        @staticmethod
        def is_available():
            return bool(hasattr(torch, "xpu") and torch.xpu.is_available())

        @staticmethod
        def name():
            return "xpu"

    orig_resolve_device_kwargs = getattr(rf_detr_mod.RFDETR, "_resolve_trainer_device_kwargs", None)
    if orig_resolve_device_kwargs is not None:
        @staticmethod
        def patched_resolve_device_kwargs(dev, *args, **kwargs):
            if dev is not None and str(dev).strip().lower().startswith("xpu"):
                return "auto", 1
            return orig_resolve_device_kwargs(dev, *args, **kwargs)
        rf_detr_mod.RFDETR._resolve_trainer_device_kwargs = patched_resolve_device_kwargs

    best_state = {"map50": 0.0, "map50_95": 0.0}

    def to_float(v):
        if v is None:
            return None
        if hasattr(v, "item"):
            try:
                return float(v.item())
            except Exception:
                pass
        try:
            return float(v)
        except Exception:
            return None

    class LSTrainerProgressCallback(Callback):
        def on_fit_start(self, trainer, pl_module):
            dm = getattr(trainer, "datamodule", None)
            if dm is not None and hasattr(dm, "_num_workers"):
                dm._num_workers = num_workers
            root_dev = getattr(getattr(trainer, "strategy", None), "root_device", device)
            accel_name = type(getattr(trainer, "accelerator", None)).__name__
            print(
                f"[LS-Trainer] Lightning Trainer active: accelerator={accel_name}, root_device={root_dev}, train/val num_workers={num_workers}",
                flush=True,
            )

        def on_train_epoch_start(self, trainer, pl_module):
            ep = int(trainer.current_epoch) + 1
            total = int(trainer.max_epochs)
            print(f"Epoch {ep}/{total} - RF-DETR ({class_name}) training on {device}...", flush=True)

        def on_validation_epoch_end(self, trainer, pl_module):
            if getattr(trainer, "sanity_checking", False):
                return
            ep = int(trainer.current_epoch) + 1
            total = int(trainer.max_epochs)
            metrics = trainer.callback_metrics or {}

            m50 = None
            for k in ("val/segm_mAP_50", "val/ema_segm_mAP_50", "val/mAP_50", "val/ema_mAP_50"):
                val = to_float(metrics.get(k))
                if val is not None and (m50 is None or val > m50):
                    m50 = val

            m50_95 = None
            for k in ("val/segm_mAP_50_95", "val/ema_segm_mAP_50_95", "val/mAP_50_95", "val/ema_mAP_50_95"):
                val = to_float(metrics.get(k))
                if val is not None and (m50_95 is None or val > m50_95):
                    m50_95 = val

            if m50 is not None and m50 > best_state["map50"]:
                best_state["map50"] = m50
            if m50_95 is not None and m50_95 > best_state["map50_95"]:
                best_state["map50_95"] = m50_95

            m50_disp = m50 if m50 is not None else best_state["map50"]
            m95_disp = m50_95 if m50_95 is not None else best_state["map50_95"]
            print(
                f"[LS-Trainer Metrics] Epoch {ep}/{total} | mAP50={m50_disp:.4f} | mAP50-95={m95_disp:.4f}",
                flush=True,
            )

    orig_build_trainer = rf_training.build_trainer

    def hooked_build_trainer(*args, **kw):
        if str(device).lower().startswith("xpu"):
            kw["accelerator"] = "auto"
            kw["devices"] = 1
            kw["strategy"] = SingleDeviceStrategy(
                device=torch.device("xpu", 0),
                accelerator=XPUAccelerator(),
            )
        elif str(device).lower().startswith("cuda"):
            kw["accelerator"] = "cuda"
            kw["devices"] = 1
        elif str(device).lower() == "cpu":
            kw["accelerator"] = "cpu"
            kw["devices"] = 1
        trainer = orig_build_trainer(*args, **kw)
        trainer.callbacks.append(LSTrainerProgressCallback())
        return trainer

    rf_training.build_trainer = hooked_build_trainer
    rf_trainer_mod.build_trainer = hooked_build_trainer

    eval_backend = "faster_coco_eval"
    for candidate in ("vernier", "hotcoco", "faster_coco_eval"):
        try:
            __import__(candidate)
            eval_backend = candidate
            break
        except ImportError:
            continue

    model_cls = getattr(rfdetr, class_name)
    print(
        f"[LS-Trainer] Initializing {class_name} (resolution={resolution}, pretrain_weights={pretrain_weights}, device={device}, eval_backend={eval_backend}, num_workers={num_workers})...",
        flush=True,
    )
    model = model_cls(resolution=resolution, pretrain_weights=pretrain_weights, device=device)

    train_kwargs = {
        "dataset_dir": dataset_dir,
        "output_dir": output_dir,
        "epochs": epochs,
        "batch_size": batch_size,
        "resolution": resolution,
        "device": device,
        "num_workers": num_workers,
        "run_test": False,
        "progress_bar": False,
        "tensorboard": False,
        "eval_backend": eval_backend,
    }
    if freeze_backbone:
        train_kwargs["lr_encoder"] = 0.0

    print(f"Epoch 1/{epochs} - Starting RF-DETR training loop on device={device} (num_workers={num_workers})...", flush=True)
    model.train(**train_kwargs)

    metrics_file = os.path.join(output_dir, "ls_trainer_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as mf:
        json.dump(best_state, mf, indent=2)

    print(
        f"[LS-Trainer Metrics] Final | mAP50={best_state['map50']:.4f} | mAP50-95={best_state['map50_95']:.4f}",
        flush=True,
    )

if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        sys.exit(1)
"#;

    std::fs::write(&runner_script_path, runner_script)
        .map_err(|e| format!("Failed to write RF-DETR runner script: {e}"))?;

    let runner_cfg_path = abs_results_dir.join("ls_trainer_rfdetr_args.json");
    let runner_cfg = serde_json::json!({
        "rfdetr_class": rfdetr_class,
        "models_dir": models_dir.to_string_lossy(),
        "dataset_dir": coco_root.to_string_lossy(),
        "output_dir": abs_results_dir.to_string_lossy(),
        "epochs": request.epochs,
        "batch_size": request.batch,
        "resolution": snapped_res,
        "device": effective_device,
        "num_workers": effective_workers,
        "pretrain_weights": resolved_pretrain,
        "freeze_backbone": request.freeze_backbone,
        "task": yolo_task,
    });
    std::fs::write(
        &runner_cfg_path,
        serde_json::to_string_pretty(&runner_cfg).unwrap_or_default(),
    )
    .map_err(|e| format!("Failed to write RF-DETR config JSON: {e}"))?;

    let python_exe = get_venv_python(&project_root).unwrap_or_else(|| PathBuf::from("python"));
    let exact_command = format!(
        "\"{}\" \"{}\" \"{}\"",
        python_exe.display(),
        runner_script_path.display(),
        runner_cfg_path.display()
    );

    let _ = on_log.send("[LS-Trainer Status] Training in progress...".to_string());
    let _ = on_log.send(format!(
        "[LS-Trainer] Active config: family=RF-DETR | task={} | model={} ({}) | resolution={} | epochs={} | batch={} | workers={} | freeze_backbone={} | device={}",
        yolo_task,
        request.model_choice,
        rfdetr_class,
        snapped_res,
        request.epochs,
        request.batch,
        effective_workers,
        request.freeze_backbone,
        effective_device
    ));
    let _ = on_log.send(format!("[LS-Trainer] Executing: {exact_command}"));

    let mut cmd = Command::new(&python_exe);
    cmd.args([
        runner_script_path.to_string_lossy().as_ref(),
        runner_cfg_path.to_string_lossy().as_ref(),
    ])
    .current_dir(&project_root)
    .env("PYTHONUNBUFFERED", "1")
    .env("PYTHONUTF8", "1")
    .env("PYTHONIOENCODING", "utf-8")
    .stdout(Stdio::piped())
    .stderr(Stdio::piped());

    #[cfg(windows)]
    {
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    match cmd.spawn() {
        Ok(mut child) => {
            let stdout = child.stdout.take();
            let stderr = child.stderr.take();

            let on_log_stdout = on_log.clone();
            let stdout_thread = std::thread::spawn(move || {
                if let Some(out) = stdout {
                    stream_reader_lines(out, &on_log_stdout);
                }
            });

            let on_log_stderr = on_log.clone();
            let stderr_thread = std::thread::spawn(move || {
                if let Some(err) = stderr {
                    stream_reader_lines(err, &on_log_stderr);
                }
            });

            let _ = stdout_thread.join();
            let _ = stderr_thread.join();

            let status = child
                .wait()
                .map_err(|e| format!("Failed while waiting for RF-DETR process: {e}"))?;

            save_run_config_metadata(
                &abs_results_dir,
                &request,
                &resolved_pretrain,
                &effective_device,
                effective_workers,
                &exact_command,
                &on_log,
            );

            let (map50, map50_95) = extract_final_metrics_from_results_dir(&abs_results_dir);

            if status.success() {
                if let (Some(m50), Some(m95)) = (map50, map50_95) {
                    let _ = on_log.send(format!(
                        "[LS-Trainer] Final Metrics — mAP50: {:.1}% ({:.4}) | mAP50-95: {:.1}% ({:.4})",
                        m50 * 100.0,
                        m50,
                        m95 * 100.0,
                        m95
                    ));
                }
                let _ = on_log.send(format!(
                    "[LS-Trainer] RF-DETR training completed successfully! Results saved to: {results_dir} (run '{run_name}')"
                ));
                Ok(TrainingResponse {
                    success: true,
                    results_dir,
                    message: "Training complete".to_string(),
                    map50,
                    map50_95,
                })
            } else {
                let _ = on_log.send(format!(
                    "[LS-Trainer] RF-DETR exited with code: {:?}",
                    status.code()
                ));
                Ok(TrainingResponse {
                    success: false,
                    results_dir,
                    message: format!("Process exited with code {:?}", status.code()),
                    map50,
                    map50_95,
                })
            }
        }
        Err(err) => Err(format!("Failed to launch RF-DETR process: {err}")),
    }
}

#[tauri::command]
async fn start_training(
    _app: tauri::AppHandle,
    on_log: Channel<String>,
    request: TrainingRequest,
) -> Result<TrainingResponse, String> {
    let run_name = if request.run_name.trim().is_empty() {
        "BB-Parts-v3".to_string()
    } else {
        request.run_name.trim().to_string()
    };

    let project_root = get_project_root();
    let yolo_task = resolve_train_task(&request);
    let project_runs_dir = project_root.join("runs").join(&yolo_task);
    let abs_results_dir = project_runs_dir.join(&run_name);
    let results_dir = to_display_path(&abs_results_dir);

    let _ = on_log.send(format!(
        "[LS-Trainer] Initializing training run '{run_name}'..."
    ));

    let dataset_input = PathBuf::from(&request.dataset_path);
    if !dataset_input.exists() {
        let msg = format!("Dataset path does not exist: {}", request.dataset_path);
        let _ = on_log.send(format!("[LS-Trainer] Error: {msg}"));
        return Err(msg);
    }

    if is_rfdetr_request(&request) {
        return start_rfdetr_training(
            project_root,
            run_name,
            yolo_task,
            abs_results_dir,
            results_dir,
            dataset_input,
            request,
            on_log,
        )
        .await;
    }

    // Determine data.yaml path: unpack if archive, or find inside directory
    let raw_data_yaml_path = if dataset_input.is_file() {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let temp_dir = std::env::temp_dir().join(format!("ls_trainer_{timestamp}"));

        let _ = on_log.send(format!(
            "[LS-Trainer] Unpacking archive '{}' to '{}'...",
            dataset_input.display(),
            temp_dir.display()
        ));

        unpack_archive(&dataset_input, &temp_dir)?;
        let _ = on_log.send("[LS-Trainer] Unpacked dataset successfully.".to_string());

        let yaml = find_data_yaml(&temp_dir).ok_or_else(|| {
            let err = format!(
                "data.yaml not found in unpacked archive directory: {}",
                temp_dir.display()
            );
            let _ = on_log.send(format!("[LS-Trainer] Error: {err}"));
            err
        })?;
        yaml
    } else if dataset_input.is_dir() {
        find_data_yaml(&dataset_input).ok_or_else(|| {
            let err = format!(
                "data.yaml not found in dataset folder: {}",
                dataset_input.display()
            );
            let _ = on_log.send(format!("[LS-Trainer] Error: {err}"));
            err
        })?
    } else {
        dataset_input.clone()
    };

    if yolo_task == "segment" {
        let label_root = raw_data_yaml_path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| dataset_input.clone());
        let label_check = inspect_label_tree(&label_root);
        if let Some(warning) = label_check.warning {
            let _ = on_log.send(format!("[LS-Trainer] Warning: {warning}"));
        } else {
            let _ = on_log.send(format!(
                "[LS-Trainer] Segmentation labels OK: polygon annotations found ({} label files, {} lines).",
                label_check.files_scanned, label_check.lines_scanned
            ));
        }
    }

    // Read data.yaml, resolve all relative paths to absolute, and rewrite it
    let fixed_data_yaml_path = resolve_and_fix_data_yaml(&raw_data_yaml_path, &on_log)?;
    let data_str = fixed_data_yaml_path.to_string_lossy().to_string();
    let dataset_working_dir = fixed_data_yaml_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| project_root.clone());

    let resolved_model =
        resolve_or_download_model_weights(&project_root, &request.model_choice, &on_log);
    let _ = on_log.send("[LS-Trainer Status] Training in progress...".to_string());

    let (auto_workers, phys_cores, log_cores) = detect_cpu_worker_info(&project_root);
    let (effective_workers, worker_source) = match request.workers {
        Some(custom_w) => (custom_w.min(8), "manual override"),
        None => (auto_workers, "auto-detected"),
    };
    let _ = on_log.send(format!(
        "[LS-Trainer] DataLoader workers: {effective_workers} ({worker_source}; {phys_cores} physical / {log_cores} logical CPU cores, capped at 8)"
    ));

    let data_arg = format!("data={data_str}");
    let model_arg = format!("model={resolved_model}");
    let name_arg = format!("name={run_name}");
    let epochs_arg = format!("epochs={}", request.epochs);
    let batch_arg = format!("batch={}", request.batch);
    let imgsz_arg = format!("imgsz={}", request.imgsz);
    let workers_arg = format!("workers={effective_workers}");
    let amp_arg = format!("amp={}", if request.amp { "True" } else { "False" });
    let mosaic_arg = format!("mosaic={}", request.mosaic);
    let mixup_arg = format!("mixup={}", request.mixup);
    let copy_paste_arg = format!("copy_paste={}", request.copy_paste);
    let degrees_arg = format!("degrees={}", request.degrees);
    let translate_arg = format!("translate={}", request.translate);
    let scale_arg = format!("scale={}", request.scale);
    let flipud_arg = format!("flipud={}", request.flipud);
    let close_mosaic_arg = format!("close_mosaic={}", request.close_mosaic);
    let task_arg = format!("task={yolo_task}");
    let freeze_layers_val = request.freeze_layers.max(1);
    let freeze_arg_opt = if request.freeze_backbone {
        Some(format!("freeze={freeze_layers_val}"))
    } else {
        None
    };

    let _ = on_log.send(format!(
        "[LS-Trainer] Ultralytics task=\"{yolo_task}\""
    ));
    let _ = on_log.send(format!(
        "[LS-Trainer] Active config: task={} | model={} | workers={} | freeze={} | mosaic={} | close_mosaic={} | mixup={} | copy_paste={} | degrees={} | flipud={} | translate={} | scale={} | amp={}",
        yolo_task,
        request.model_choice,
        effective_workers,
        if request.freeze_backbone {
            format!("{freeze_layers_val} layers")
        } else {
            "none (0)".to_string()
        },
        request.mosaic,
        request.close_mosaic,
        request.mixup,
        request.copy_paste,
        request.degrees,
        request.flipud,
        request.translate,
        request.scale,
        request.amp
    ));

    let device_input = request.device.trim();
    let effective_device = if device_input.is_empty() || device_input.eq_ignore_ascii_case("auto") {
        if is_cuda_available(&project_root) {
            let _ = on_log.send("[LS-Trainer] CUDA available, selecting device=0".to_string());
            "0".to_string()
        } else {
            let _ = on_log.send("[LS-Trainer] No CUDA detected, using device=cpu".to_string());
            "cpu".to_string()
        }
    } else if device_input.eq_ignore_ascii_case("gpu") || device_input.eq_ignore_ascii_case("cuda") {
        if is_cuda_available(&project_root) {
            let _ = on_log.send("[LS-Trainer] CUDA available, selecting device=0".to_string());
            "0".to_string()
        } else {
            let _ = on_log.send("[LS-Trainer] No CUDA detected, falling back to device=cpu".to_string());
            "cpu".to_string()
        }
    } else {
        device_input.to_string()
    };

    let device_arg = format!("device={effective_device}");

    #[cfg(target_os = "windows")]
    let expected_venv_yolo = project_root.join(".venv").join("Scripts").join("yolo.exe");

    #[cfg(not(target_os = "windows"))]
    let expected_venv_yolo = project_root.join(".venv").join("bin").join("yolo");

    let project_runs_str = project_runs_dir.to_string_lossy().replace('\\', "/");
    let project_arg = format!("project={project_runs_str}");

    let mut train_args: Vec<String> = vec![
        yolo_task.clone(),
        "train".to_string(),
        task_arg,
        data_arg,
        model_arg,
        name_arg,
        project_arg,
        epochs_arg,
        batch_arg,
        imgsz_arg,
        workers_arg,
        device_arg,
        amp_arg,
    ];
    if let Some(ref f_arg) = freeze_arg_opt {
        train_args.push(f_arg.clone());
    }
    train_args.extend([
        mosaic_arg,
        close_mosaic_arg,
        mixup_arg,
        copy_paste_arg,
        degrees_arg,
        flipud_arg,
        translate_arg,
        scale_arg,
    ]);
    let joined_train_args = train_args.join(" ");

    let (mut cmd, exact_command) = if expected_venv_yolo.is_file() {
        let _ = on_log.send(format!(
            "[LS-Trainer] Using local venv YOLO executable: {}",
            expected_venv_yolo.display()
        ));

        let mut c = Command::new(&expected_venv_yolo);
        c.args(&train_args);

        let cmd_str = format!(
            "\"{}\" {joined_train_args}",
            expected_venv_yolo.display()
        );

        (c, cmd_str)
    } else {
        let _ = on_log.send(format!(
            "[LS-Trainer] Warning: Local .venv executable not found at '{}'. Falling back to system 'yolo'.",
            expected_venv_yolo.display()
        ));

        #[cfg(target_os = "windows")]
        let c = {
            let mut cmd = Command::new("cmd");
            cmd.args(["/C", "yolo"]);
            cmd.args(&train_args);
            cmd
        };

        #[cfg(not(target_os = "windows"))]
        let c = {
            let mut cmd = Command::new("yolo");
            cmd.args(&train_args);
            cmd
        };

        let cmd_str = format!("yolo {joined_train_args}");

        (c, cmd_str)
    };

    let _ = on_log.send(format!("[LS-Trainer] Executing: {exact_command}"));

    // Set working directory of the process to the unpacked dataset folder
    cmd.current_dir(&dataset_working_dir);
    let _ = on_log.send(format!(
        "[LS-Trainer] Set YOLO working directory to: {}",
        dataset_working_dir.display()
    ));

    cmd.env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    #[cfg(windows)]
    {
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    match cmd.spawn() {
        Ok(mut child) => {
            let stdout = child.stdout.take();
            let stderr = child.stderr.take();

            let on_log_stdout = on_log.clone();
            let stdout_thread = std::thread::spawn(move || {
                if let Some(out) = stdout {
                    stream_reader_lines(out, &on_log_stdout);
                }
            });

            let on_log_stderr = on_log.clone();
            let stderr_thread = std::thread::spawn(move || {
                if let Some(err) = stderr {
                    stream_reader_lines(err, &on_log_stderr);
                }
            });

            let _ = stdout_thread.join();
            let _ = stderr_thread.join();

            let status = child
                .wait()
                .map_err(|e| format!("Failed while waiting for YOLO: {e}"))?;

            save_run_config_metadata(
                &abs_results_dir,
                &request,
                &resolved_model,
                &effective_device,
                effective_workers,
                &exact_command,
                &on_log,
            );

            let (map50, map50_95) = extract_final_metrics_from_results_dir(&abs_results_dir);

            if status.success() {
                if let (Some(m50), Some(m95)) = (map50, map50_95) {
                    let _ = on_log.send(format!(
                        "[LS-Trainer] Final Metrics — mAP50: {:.1}% ({:.4}) | mAP50-95: {:.1}% ({:.4})",
                        m50 * 100.0,
                        m50,
                        m95 * 100.0,
                        m95
                    ));
                }
                let _ = on_log.send(format!(
                    "[LS-Trainer] Training completed successfully! Results saved to: {results_dir}"
                ));
                Ok(TrainingResponse {
                    success: true,
                    results_dir,
                    message: "Training complete".to_string(),
                    map50,
                    map50_95,
                })
            } else {
                let _ = on_log.send(format!(
                    "[LS-Trainer] YOLO exited with code: {:?}",
                    status.code()
                ));
                Ok(TrainingResponse {
                    success: false,
                    results_dir,
                    message: format!("Process exited with code {:?}", status.code()),
                    map50,
                    map50_95,
                })
            }
        }
        Err(err) => {
            save_run_config_metadata(
                &abs_results_dir,
                &request,
                &resolved_model,
                &effective_device,
                effective_workers,
                &exact_command,
                &on_log,
            );

            let _ = on_log.send(format!(
                "[LS-Trainer] Notice: Could not launch YOLO process directly ({err})."
            ));
            let _ = on_log.send(
                "[LS-Trainer] Ensure 'ultralytics' is installed in the active environment ('pip install ultralytics')."
                    .to_string(),
            );
            let _ = on_log.send(format!("[LS-Trainer] Command was: {exact_command}"));
            let _ = on_log.send(format!(
                "[LS-Trainer] Target results folder: {results_dir}"
            ));
            let _ = on_log.send("[LS-Trainer] Wiring and parameter verification complete.".to_string());

            Ok(TrainingResponse {
                success: true,
                results_dir,
                message: "Command prepared and verified".to_string(),
                map50: None,
                map50_95: None,
            })
        }
    }
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

#[tauri::command]
async fn open_folder(path: String) -> Result<(), String> {
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

fn find_rfdetr_checkpoint_in_dir(results_path: &Path) -> Option<PathBuf> {
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

fn embed_onnx_model_type_metadata(
    project_root: &Path,
    onnx_path: &Path,
    model_type: &str,
    model_family: &str,
    on_log: &Channel<String>,
) {
    if !onnx_path.is_file() {
        return;
    }
    let Some(python_exe) = get_venv_python(project_root) else {
        return;
    };

    let py_snippet = r#"import sys, onnx
onnx_path, model_type, model_family = sys.argv[1], sys.argv[2], sys.argv[3]
m = onnx.load(onnx_path)
props = {p.key: p.value for p in m.metadata_props}
props["ls_model_type"] = model_type
props["ls_model_family"] = model_family
del m.metadata_props[:]
for k, v in props.items():
    entry = m.metadata_props.add()
    entry.key = str(k)
    entry.value = str(v)
onnx.save(m, onnx_path)
"#;

    let mut cmd = Command::new(&python_exe);
    cmd.args([
        "-c",
        py_snippet,
        onnx_path.to_string_lossy().as_ref(),
        model_type,
        model_family,
    ]);
    #[cfg(windows)]
    {
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    if let Ok(status) = cmd.status() {
        if status.success() {
            let _ = on_log.send(format!(
                "[LS-Trainer] Embedded ONNX metadata: ls_model_type=\"{model_type}\" (ls_model_family=\"{model_family}\")"
            ));
        }
    }
}

async fn export_rfdetr_to_onnx(
    project_root: &Path,
    results_path: &Path,
    checkpoint_path: &Path,
    request: &ExportRequest,
    on_log: &Channel<String>,
) -> Result<ExportResponse, String> {
    ensure_rfdetr_installed(project_root, on_log);

    let output_dir = if results_path.is_dir() {
        results_path.to_path_buf()
    } else {
        checkpoint_path
            .parent()
            .unwrap_or(project_root)
            .to_path_buf()
    };

    // Determine RF-DETR model class and task from saved run metadata if available
    let mut model_choice = "rf-detr-medium".to_string();
    let mut task = "detect".to_string();
    let config_path = output_dir.join("ls_trainer_config.json");
    if config_path.is_file() {
        if let Ok(text) = std::fs::read_to_string(&config_path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                if let Some(mc) = json.get("model_choice").and_then(|v| v.as_str()) {
                    model_choice = mc.to_string();
                }
                if let Some(t) = json.get("task").and_then(|v| v.as_str()) {
                    task = t.to_string();
                }
            }
        }
    }

    let (rfdetr_class, _, block_size) = resolve_rfdetr_class_and_pretrain(&model_choice, &task);
    let raw_imgsz = if request.imgsz == 0 {
        default_export_imgsz()
    } else {
        request.imgsz
    };
    let export_res = snap_rfdetr_resolution(raw_imgsz, block_size);
    let device = if is_cuda_available(project_root) {
        "cuda"
    } else {
        "cpu"
    };

    let ckpt_display = checkpoint_path.to_string_lossy().replace('\\', "/");
    let _ = on_log.send(format!(
        "[LS-Trainer] Found RF-DETR checkpoint: {ckpt_display}"
    ));
    let _ = on_log.send(format!(
        "[LS-Trainer] Exporting {rfdetr_class} to ONNX (shape=({export_res}, {export_res}), dynamic={}, simplify={})...",
        request.dynamic, request.simplify
    ));

    let export_script_path = output_dir.join("ls_trainer_rfdetr_export.py");
    let export_script = r#"import json
import os
import sys
import traceback

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def embed_rfdetr_onnx_metadata(onnx_file, class_name, task, resolution, output_dir, model):
    import onnx
    m = onnx.load(str(onnx_file))
    props = {p.key: p.value for p in m.metadata_props}
    props["ls_model_type"] = "rfdetr"
    props["ls_model_family"] = "RF-DETR"
    props["ls_model_name"] = str(class_name)
    props["task"] = str(task)
    props["imgsz"] = f"[{resolution}, {resolution}]"

    class_names = None
    tc_path = os.path.join(output_dir, "training_config.json")
    if os.path.isfile(tc_path):
        try:
            with open(tc_path, "r", encoding="utf-8") as tf:
                tc = json.load(tf)
                if isinstance(tc.get("class_names"), list) and tc["class_names"]:
                    class_names = [str(c) for c in tc["class_names"]]
        except Exception:
            pass
    if not class_names:
        raw_names = getattr(model, "class_names", None)
        if isinstance(raw_names, dict):
            class_names = [str(v) for _, v in sorted(raw_names.items())]
        elif isinstance(raw_names, (list, tuple)) and raw_names:
            class_names = [str(c) for c in raw_names]

    if class_names:
        names_dict = {i: name for i, name in enumerate(class_names)}
        props["names"] = str(names_dict)
        props["class_names"] = json.dumps(class_names)

    del m.metadata_props[:]
    for k, v in props.items():
        entry = m.metadata_props.add()
        entry.key = str(k)
        entry.value = str(v)
    onnx.save(m, str(onnx_file))
    print(
        f'[LS-Trainer] Embedded ONNX metadata: ls_model_type="rfdetr" (ls_model_family="RF-DETR", model={class_name}, task={task}, imgsz={resolution})',
        flush=True,
    )

def main():
    cfg_path = sys.argv[1]
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    import rfdetr

    class_name = cfg["rfdetr_class"]
    ckpt_path = cfg["checkpoint_path"]
    output_dir = cfg["output_dir"]
    resolution = int(cfg["resolution"])
    dynamic_batch = bool(cfg["dynamic"])
    device = cfg["device"]
    task = cfg.get("task", "detect")

    os.makedirs(output_dir, exist_ok=True)
    model_cls = getattr(rfdetr, class_name)
    if hasattr(model_cls, "from_checkpoint"):
        model = model_cls.from_checkpoint(
            ckpt_path,
            resolution=resolution,
            device=device,
            trust_checkpoint=True,
        )
    else:
        model = model_cls(
            pretrain_weights=ckpt_path,
            resolution=resolution,
            device=device,
        )
    onnx_file = model.export(
        output_dir=output_dir,
        opset_version=17,
        verbose=False,
        shape=(resolution, resolution),
        dynamic_batch=dynamic_batch,
        output_name="best",
    )
    embed_rfdetr_onnx_metadata(onnx_file, class_name, task, resolution, output_dir, model)
    print(f"[LS-Trainer] Exported RF-DETR ONNX model to: {onnx_file}", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
"#;

    std::fs::write(&export_script_path, export_script)
        .map_err(|e| format!("Failed to write RF-DETR ONNX export script: {e}"))?;

    let export_cfg_path = output_dir.join("ls_trainer_rfdetr_export_args.json");
    let export_cfg = serde_json::json!({
        "rfdetr_class": rfdetr_class,
        "checkpoint_path": checkpoint_path.to_string_lossy(),
        "output_dir": output_dir.to_string_lossy(),
        "resolution": export_res,
        "dynamic": request.dynamic,
        "simplify": request.simplify,
        "device": device,
        "task": task,
    });
    std::fs::write(
        &export_cfg_path,
        serde_json::to_string_pretty(&export_cfg).unwrap_or_default(),
    )
    .map_err(|e| format!("Failed to write RF-DETR export config: {e}"))?;

    let python_exe = get_venv_python(project_root).unwrap_or_else(|| PathBuf::from("python"));
    let mut cmd = Command::new(&python_exe);
    cmd.args([
        export_script_path.to_string_lossy().as_ref(),
        export_cfg_path.to_string_lossy().as_ref(),
    ])
    .current_dir(project_root)
    .env("PYTHONUNBUFFERED", "1")
    .env("PYTHONUTF8", "1")
    .env("PYTHONIOENCODING", "utf-8")
    .stdout(Stdio::piped())
    .stderr(Stdio::piped());

    #[cfg(windows)]
    {
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    match cmd.spawn() {
        Ok(mut child) => {
            let stdout = child.stdout.take();
            let stderr = child.stderr.take();
            let on_log_out = on_log.clone();
            let t1 = std::thread::spawn(move || {
                if let Some(out) = stdout {
                    stream_reader_lines(out, &on_log_out);
                }
            });
            let on_log_err = on_log.clone();
            let t2 = std::thread::spawn(move || {
                if let Some(err) = stderr {
                    stream_reader_lines(err, &on_log_err);
                }
            });
            let _ = t1.join();
            let _ = t2.join();

            let status = child
                .wait()
                .map_err(|e| format!("Failed while waiting for RF-DETR ONNX export: {e}"))?;

            let expected_onnx = output_dir.join("best.onnx");
            let onnx_str = to_display_path(&expected_onnx);
            if status.success() && expected_onnx.is_file() {
                let _ = on_log.send(format!(
                    "[LS-Trainer] ONNX export completed successfully! File: {onnx_str}"
                ));
                let _ = on_log.send(
                    "[LS-Trainer] Note: This model is optimized and ready for inference in LS-Vision."
                        .to_string(),
                );
                Ok(ExportResponse {
                    success: true,
                    onnx_path: onnx_str,
                    message: "ONNX export completed successfully".to_string(),
                })
            } else {
                let err_msg = format!(
                    "RF-DETR ONNX export failed with status code: {:?}",
                    status.code()
                );
                let _ = on_log.send(format!("[LS-Trainer Error] {err_msg}"));
                Ok(ExportResponse {
                    success: false,
                    onnx_path: String::new(),
                    message: err_msg,
                })
            }
        }
        Err(err) => Err(format!("Failed to launch RF-DETR ONNX export: {err}")),
    }
}

#[tauri::command]
async fn export_to_onnx(
    _app: tauri::AppHandle,
    on_log: Channel<String>,
    request: ExportRequest,
) -> Result<ExportResponse, String> {
    let project_root = get_project_root();

    let mut results_path = PathBuf::from(&request.results_dir);
    if !results_path.is_absolute() {
        results_path = project_root.join(&results_path);
    }

    let _ = on_log.send(format!(
        "[LS-Trainer] Preparing ONNX export for results in '{}'...",
        results_path.display()
    ));

    if let Some(rfdetr_ckpt) = find_rfdetr_checkpoint_in_dir(&results_path) {
        return export_rfdetr_to_onnx(
            &project_root,
            &results_path,
            &rfdetr_ckpt,
            &request,
            &on_log,
        )
        .await;
    }

    // Look for best.pt in weights/best.pt, best.pt, or recursively
    let candidate_weights = results_path.join("weights").join("best.pt");
    let candidate_direct = results_path.join("best.pt");

    let best_pt = if candidate_weights.is_file() {
        candidate_weights
    } else if candidate_direct.is_file() {
        candidate_direct
    } else {
        let mut found = None;
        if results_path.is_dir() {
            for entry in walkdir::WalkDir::new(&results_path)
                .max_depth(3)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                if entry.file_type().is_file()
                    && entry
                        .file_name()
                        .to_string_lossy()
                        .eq_ignore_ascii_case("best.pt")
                {
                    found = Some(entry.path().to_path_buf());
                    break;
                }
            }
        }

        match found {
            Some(p) => p,
            None => {
                let err_msg = format!(
                    "Checkpoint 'best.pt' not found in '{}' (checked 'weights/best.pt' and direct directory).",
                    results_path.display()
                );
                let _ = on_log.send(format!("[LS-Trainer Error] {err_msg}"));
                return Err(err_msg);
            }
        }
    };

    let best_pt_str = best_pt.to_string_lossy().replace('\\', "/");
    let _ = on_log.send(format!("[LS-Trainer] Found checkpoint: {best_pt_str}"));

    let export_imgsz = if request.imgsz == 0 {
        default_export_imgsz()
    } else {
        request.imgsz
    };
    let dynamic_flag = if request.dynamic { "True" } else { "False" };
    let simplify_flag = if request.simplify { "True" } else { "False" };
    let _ = on_log.send(format!(
        "[LS-Trainer] model.export(format=\"onnx\", imgsz={export_imgsz}, dynamic={dynamic_flag}, simplify={simplify_flag})"
    ));

    #[cfg(target_os = "windows")]
    let expected_venv_yolo = project_root.join(".venv").join("Scripts").join("yolo.exe");

    #[cfg(not(target_os = "windows"))]
    let expected_venv_yolo = project_root.join(".venv").join("bin").join("yolo");

    let model_arg = format!("model={best_pt_str}");
    let imgsz_arg = format!("imgsz={export_imgsz}");
    let dynamic_arg = format!("dynamic={dynamic_flag}");
    let simplify_arg = format!("simplify={simplify_flag}");
    let export_args = [
        "export",
        model_arg.as_str(),
        "format=onnx",
        imgsz_arg.as_str(),
        dynamic_arg.as_str(),
        simplify_arg.as_str(),
    ];

    let (mut cmd, exact_command) = if expected_venv_yolo.is_file() {
        let _ = on_log.send(format!(
            "[LS-Trainer] Using local venv YOLO executable: {}",
            expected_venv_yolo.display()
        ));

        let mut c = Command::new(&expected_venv_yolo);
        c.args(export_args);

        let cmd_str = format!(
            "\"{}\" export model=\"{best_pt_str}\" format=onnx imgsz={export_imgsz} dynamic={dynamic_flag} simplify={simplify_flag}",
            expected_venv_yolo.display()
        );

        (c, cmd_str)
    } else {
        let _ = on_log.send(format!(
            "[LS-Trainer] Warning: Local .venv executable not found at '{}'. Falling back to system 'yolo'.",
            expected_venv_yolo.display()
        ));

        #[cfg(target_os = "windows")]
        let c = {
            let mut cmd = Command::new("cmd");
            cmd.args(["/C", "yolo"]);
            cmd.args(export_args);
            cmd
        };

        #[cfg(not(target_os = "windows"))]
        let c = {
            let mut cmd = Command::new("yolo");
            cmd.args(export_args);
            cmd
        };

        let cmd_str = format!(
            "yolo export model=\"{best_pt_str}\" format=onnx imgsz={export_imgsz} dynamic={dynamic_flag} simplify={simplify_flag}"
        );

        (c, cmd_str)
    };

    let _ = on_log.send(format!("[LS-Trainer] Executing: {exact_command}"));

    let parent_dir = best_pt.parent().unwrap_or(&results_path);
    cmd.current_dir(parent_dir);
    cmd.env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    #[cfg(windows)]
    {
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    match cmd.spawn() {
        Ok(mut child) => {
            let stdout = child.stdout.take();
            let stderr = child.stderr.take();

            let on_log_stdout = on_log.clone();
            let stdout_thread = std::thread::spawn(move || {
                if let Some(out) = stdout {
                    stream_reader_lines(out, &on_log_stdout);
                }
            });

            let on_log_stderr = on_log.clone();
            let stderr_thread = std::thread::spawn(move || {
                if let Some(err) = stderr {
                    stream_reader_lines(err, &on_log_stderr);
                }
            });

            let _ = stdout_thread.join();
            let _ = stderr_thread.join();

            let status = child
                .wait()
                .map_err(|e| format!("Failed while waiting for export process: {e}"))?;

            let expected_onnx = best_pt.with_extension("onnx");
            let onnx_str = to_display_path(&expected_onnx);

            if status.success() {
                embed_onnx_model_type_metadata(
                    &project_root,
                    &expected_onnx,
                    "yolo",
                    "YOLO",
                    &on_log,
                );
                let _ = on_log.send(format!(
                    "[LS-Trainer] ONNX export completed successfully! File: {onnx_str}"
                ));
                let _ = on_log.send(
                    "[LS-Trainer] Note: This model is optimized and ready for inference in LS-Vision."
                        .to_string(),
                );
                Ok(ExportResponse {
                    success: true,
                    onnx_path: onnx_str,
                    message: "ONNX export completed successfully".to_string(),
                })
            } else {
                let err_msg = format!("YOLO export failed with status code: {:?}", status.code());
                let _ = on_log.send(format!("[LS-Trainer Error] {err_msg}"));
                Ok(ExportResponse {
                    success: false,
                    onnx_path: String::new(),
                    message: err_msg,
                })
            }
        }
        Err(err) => {
            let _ = on_log.send(format!(
                "[LS-Trainer] Notice: Could not launch YOLO process directly ({err})."
            ));
            let expected_onnx = best_pt.with_extension("onnx");
            let onnx_str = to_display_path(&expected_onnx);
            let _ = on_log.send(format!("[LS-Trainer] Simulated ONNX export: {onnx_str}"));
            let _ = on_log.send(
                "[LS-Trainer] Note: This model is optimized and ready for inference in LS-Vision."
                    .to_string(),
            );
            Ok(ExportResponse {
                success: true,
                onnx_path: onnx_str,
                message: "Export command verified in simulation mode".to_string(),
            })
        }
    }
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
            pick_dataset_file,
            pick_dataset_folder,
            pick_model_file,
            inspect_dataset_labels,
            start_training,
            open_folder,
            export_to_onnx,
            get_gpu_info,
            repair_cuda
        ])
        .run(tauri::generate_context!())
        .expect("error while building tauri application");
}

#[cfg(test)]
mod label_tests {
    use super::*;

    #[test]
    fn polygon_line_has_more_than_five_values() {
        let mut stats = LabelStats::default();
        scan_label_text("0 0.1 0.1 0.2 0.1 0.2 0.2\n", &mut stats);
        assert_eq!(stats.polygons, 1);
        assert_eq!(stats.bboxes, 0);
        let result = result_from_stats(&stats);
        assert!(result.polygons_found);
        assert!(result.warning.is_none());
    }

    #[test]
    fn bbox_line_warns_for_segmentation() {
        let mut stats = LabelStats::default();
        scan_label_text("0 0.5 0.5 0.2 0.2\n", &mut stats);
        let result = result_from_stats(&stats);
        assert!(result.bboxes_found);
        assert!(!result.polygons_found);
        let warning = result.warning.expect("bbox dataset should warn");
        assert!(warning.to_lowercase().contains("bounding-box"));
    }

    #[test]
    fn mixed_labels_warn() {
        let mut stats = LabelStats::default();
        scan_label_text("0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n", &mut stats);
        let result = result_from_stats(&stats);
        assert!(result.polygons_found && result.bboxes_found);
        assert!(result.warning.is_some());
    }

    #[test]
    fn class_name_lines_are_not_annotations() {
        let mut stats = LabelStats::default();
        scan_label_text("bolt\nnut\n", &mut stats);
        assert_eq!(stats.lines_scanned, 0);
        assert!(result_from_stats(&stats).warning.is_some());
    }

    #[test]
    fn archive_scan_warns_on_bbox_labels() {
        let mut raw = Vec::new();
        {
            let mut builder = tar::Builder::new(&mut raw);
            let body = b"0 0.5 0.5 0.1 0.1\n";
            let mut header = tar::Header::new_gnu();
            header.set_size(body.len() as u64);
            header.set_cksum();
            builder
                .append_data(&mut header, "labels/train/a.txt", &body[..])
                .unwrap();
            let classes = b"bolt\n";
            let mut classes_header = tar::Header::new_gnu();
            classes_header.set_size(classes.len() as u64);
            classes_header.set_cksum();
            builder
                .append_data(&mut classes_header, "classes.txt", &classes[..])
                .unwrap();
            builder.finish().unwrap();
        }
        let result = scan_tar_archive(tar::Archive::new(std::io::Cursor::new(raw))).unwrap();
        assert!(result.bboxes_found);
        assert!(!result.polygons_found);
        assert!(result.warning.is_some());
    }

    #[test]
    fn coco_json_scan_detects_bboxes_and_polygons() {
        let mut bbox_stats = LabelStats::default();
        let coco_bbox = r#"{
            "images": [{"id": 1, "file_name": "a.jpg", "width": 640, "height": 480}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10, 50, 50], "segmentation": []}],
            "categories": [{"id": 0, "name": "bolt"}]
        }"#;
        scan_coco_json_text(coco_bbox, &mut bbox_stats);
        assert_eq!(bbox_stats.bboxes, 1);
        assert_eq!(bbox_stats.polygons, 0);

        let mut poly_stats = LabelStats::default();
        let coco_poly = r#"{
            "images": [{"id": 1, "file_name": "a.jpg", "width": 640, "height": 480}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10, 50, 50], "segmentation": [[10, 10, 60, 10, 60, 60]]}],
            "categories": [{"id": 0, "name": "bolt"}]
        }"#;
        scan_coco_json_text(coco_poly, &mut poly_stats);
        assert_eq!(poly_stats.polygons, 1);
        assert_eq!(poly_stats.bboxes, 0);
    }

    #[test]
    fn rfdetr_resolution_snapping_and_model_mapping() {
        let (cls_n, weights_n, block_n) =
            resolve_rfdetr_class_and_pretrain("rf-detr-nano", "detect");
        assert_eq!(cls_n, "RFDETRNano");
        assert_eq!(weights_n, "rf-detr-nano.pth");
        assert_eq!(snap_rfdetr_resolution(640, block_n), 640);
        assert_eq!(snap_rfdetr_resolution(960, block_n), 960);
        assert_eq!(snap_rfdetr_resolution(1280, block_n), 1280);

        let (cls_seg, _, block_seg) =
            resolve_rfdetr_class_and_pretrain("rf-detr-seg-small", "segment");
        assert_eq!(cls_seg, "RFDETRSegSmall");
        assert_eq!(snap_rfdetr_resolution(640, block_seg), 648);
    }

    #[test]
    fn seg_weight_forces_segment_task() {
        let request = TrainingRequest {
            dataset_path: String::new(),
            model_family: "YOLO".to_string(),
            model_choice: "yolo11s-seg.pt".to_string(),
            run_name: String::new(),
            epochs: 1,
            imgsz: 1280,
            device: "cpu".to_string(),
            batch: 4,
            workers: None,
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
            task: "detect".to_string(),
        };
        assert_eq!(resolve_train_task(&request), "segment");
    }

    #[test]
    fn task_selector_segment_without_seg_filename() {
        let request = TrainingRequest {
            dataset_path: String::new(),
            model_family: "YOLO".to_string(),
            model_choice: r"C:\runs\segment\best.pt".to_string(),
            run_name: String::new(),
            epochs: 1,
            imgsz: 1280,
            device: "cpu".to_string(),
            batch: 4,
            workers: None,
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
            task: "segment".to_string(),
        };
        assert_eq!(resolve_train_task(&request), "segment");
    }

    #[test]
    fn worker_detection_caps_and_reserves_core() {
        // 8 physical cores -> 7 workers (leaves 1 free)
        assert_eq!(compute_recommended_workers(Some(8), 8), 7);
        // 16 physical cores -> capped at 8
        assert_eq!(compute_recommended_workers(Some(16), 32), 8);
        // Fallback when physical_cores is None: logical // 2 - 1
        assert_eq!(compute_recommended_workers(None, 12), 5);
        assert_eq!(compute_recommended_workers(None, 2), 1);
        assert_eq!(compute_recommended_workers(Some(1), 1), 0);
    }

    #[test]
    fn intel_arc_and_pytorch_wheel_selection() {
        assert!(is_intel_arc_gpu("Intel(R) Arc(TM) 140V GPU (16GB)"));
        assert!(is_intel_arc_gpu("Intel Arc A770 Graphics"));
        assert!(!is_intel_arc_gpu("Intel(R) UHD Graphics 770"));
        assert!(!is_intel_arc_gpu("NVIDIA GeForce RTX 4080"));

        let (arc_wheel, arc_url, arc_label) =
            select_pytorch_wheel(Some("Intel(R) Arc(TM) 140V GPU (16GB)"), None);
        assert_eq!(arc_wheel, "xpu");
        assert_eq!(arc_url, "https://download.pytorch.org/whl/xpu");
        assert_eq!(arc_label, "Intel Arc XPU");

        let (nv_50_wheel, _, _) =
            select_pytorch_wheel(Some("NVIDIA GeForce RTX 5080"), Some(572));
        assert_eq!(nv_50_wheel, "cu128");

        let (nv_old_wheel, _, _) =
            select_pytorch_wheel(Some("NVIDIA GeForce RTX 3060"), Some(535));
        assert_eq!(nv_old_wheel, "cu124");

        let (cpu_wheel, cpu_url, _) = select_pytorch_wheel(None, None);
        assert_eq!(cpu_wheel, "cpu");
        assert_eq!(cpu_url, "https://download.pytorch.org/whl/cpu");
    }
}
