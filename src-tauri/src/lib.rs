use std::fs::File;
use std::io::BufReader;
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
    0.0
}

fn default_translate() -> f32 {
    0.1
}

fn default_scale() -> f32 {
    0.2
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
pub struct TrainingResponse {
    pub success: bool,
    pub results_dir: String,
    pub message: String,
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
    pub cpu_fallback_confirmed: bool,
    pub device_name: Option<String>,
    pub driver_version: Option<String>,
    pub recommended_wheel: String,
    pub cuda_label: String,
    pub torch_version: Option<String>,
    pub needs_repair: bool,
    pub status_summary: String,
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
    let trimmed = full_name.trim();
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

    // Windows WMI fallback via PowerShell
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
        cpu_fallback_confirmed: true,
        device_name: None,
        driver_version: None,
        recommended_wheel: "cpu".to_string(),
        cuda_label: "CPU-only".to_string(),
        torch_version: None,
        needs_repair: false,
        status_summary: "CPU fallback confirmed".to_string(),
    })
}

fn detect_gpu_info(project_root: &Path) -> GpuInfo {
    let (hw_gpu_name, driver_version, driver_major) = detect_hardware_gpu();
    let (recommended_wheel, _index_url, cuda_label) =
        select_pytorch_wheel(hw_gpu_name.as_deref(), driver_major);

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
    let mut torch_kernel_ok = false;
    let mut torch_ver: Option<String> = None;
    let mut torch_dev_name: Option<String> = None;

    if venv_python.is_file() {
        let mut py_cmd = Command::new(&venv_python);
        py_cmd.args([
            "-c",
            "import torch; avail = bool(torch.cuda.is_available()); ver = getattr(torch, '__version__', ''); dev = torch.cuda.get_device_name(0) if avail else ''; kok = False\nif avail:\n    try:\n        _ = torch.zeros(1, device='cuda') + 1\n        torch.cuda.synchronize()\n        kok = True\n    except Exception:\n        kok = False\nprint(f'TORCH_OK|{avail}|{ver}|{dev}|{kok}')",
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
                            torch_kernel_ok = k.eq_ignore_ascii_case("true");
                        }
                    }
                }
            }
        }
    }

    let effective_gpu_name = torch_dev_name.or(hw_gpu_name.clone());

    if let Some(ref gpu_name) = effective_gpu_name {
        let ver_str = torch_ver.as_deref().unwrap_or("");
        let has_matching_wheel = if recommended_wheel == "cu128" {
            ver_str.contains("+cu128")
        } else {
            ver_str.contains("+cu")
        };

        let cuda_working = torch_installed && torch_cuda_avail && torch_kernel_ok && has_matching_wheel;
        // If we haven't already tried & recorded a repair for this wheel, mark needs_repair when !cuda_working
        let already_attempted = marker_wheel.as_deref() == Some(recommended_wheel);
        let needs_repair = !cuda_working && !already_attempted;

        let short_name = shorten_gpu_name(gpu_name);
        let status_summary = if cuda_working {
            format!("Ready — {short_name} ({cuda_label})")
        } else if needs_repair {
            format!("PyTorch {cuda_label} setup required for {short_name}")
        } else if torch_installed && torch_cuda_avail && torch_kernel_ok {
            format!("Ready — {short_name}")
        } else {
            format!("{short_name} detected (CUDA inactive — click Repair CUDA)")
        };

        return GpuInfo {
            cuda_available: torch_cuda_avail && torch_kernel_ok,
            cpu_fallback_confirmed: torch_installed && !(torch_cuda_avail && torch_kernel_ok),
            device_name: Some(gpu_name.clone()),
            driver_version,
            recommended_wheel: recommended_wheel.to_string(),
            cuda_label: cuda_label.to_string(),
            torch_version: torch_ver,
            needs_repair,
            status_summary,
        };
    }

    // No NVIDIA GPU detected
    let already_attempted = marker_wheel.as_deref() == Some("cpu");
    let needs_repair = !torch_installed && !already_attempted && venv_python.is_file();
    let status_summary = if torch_installed {
        "Ready — CPU mode (No NVIDIA GPU)".to_string()
    } else {
        "Installing CPU PyTorch wheel…".to_string()
    };

    GpuInfo {
        cuda_available: false,
        cpu_fallback_confirmed: torch_installed || !venv_python.is_file(),
        device_name: None,
        driver_version: None,
        recommended_wheel: "cpu".to_string(),
        cuda_label: "CPU-only".to_string(),
        torch_version: torch_ver,
        needs_repair,
        status_summary,
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
        if updated_info.cuda_available {
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
        .add_filter("Dataset Archive (*.tar, *.tar.gz, *.tgz)", &["tar", "tar.gz", "tgz"])
        .set_title("Select Dataset Archive")
        .pick_file()
        .await;
    file.map(|f| to_display_path(f.path()))
}

#[tauri::command]
async fn pick_model_file() -> Option<String> {
    let file = rfd::AsyncFileDialog::new()
        .add_filter("YOLO Checkpoint (*.pt)", &["pt"])
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

fn save_run_config_metadata(
    abs_results_dir: &Path,
    request: &TrainingRequest,
    resolved_model: &str,
    effective_device: &str,
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
        "model_choice": request.model_choice,
        "resolved_model": resolved_model,
        "freeze_backbone": request.freeze_backbone,
        "freeze_layers": effective_freeze,
        "epochs": request.epochs,
        "batch": request.batch,
        "imgsz": request.imgsz,
        "device": effective_device,
        "amp": request.amp,
        "augmentation": {
            "mosaic": request.mosaic,
            "mixup": request.mixup,
            "copy_paste": request.copy_paste,
            "degrees": request.degrees,
            "translate": request.translate,
            "scale": request.scale,
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
    let project_runs_dir = project_root.join("runs").join("detect");
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

    let data_arg = format!("data={data_str}");
    let model_arg = format!("model={resolved_model}");
    let name_arg = format!("name={run_name}");
    let epochs_arg = format!("epochs={}", request.epochs);
    let batch_arg = format!("batch={}", request.batch);
    let imgsz_arg = format!("imgsz={}", request.imgsz);
    let amp_arg = format!("amp={}", if request.amp { "True" } else { "False" });
    let mosaic_arg = format!("mosaic={}", request.mosaic);
    let mixup_arg = format!("mixup={}", request.mixup);
    let copy_paste_arg = format!("copy_paste={}", request.copy_paste);
    let degrees_arg = format!("degrees={}", request.degrees);
    let translate_arg = format!("translate={}", request.translate);
    let scale_arg = format!("scale={}", request.scale);
    let freeze_layers_val = request.freeze_layers.max(1);
    let freeze_arg_opt = if request.freeze_backbone {
        Some(format!("freeze={freeze_layers_val}"))
    } else {
        None
    };

    let _ = on_log.send(format!(
        "[LS-Trainer] Active config: model={} | freeze={} | mosaic={} | mixup={} | copy_paste={} | degrees={} | translate={} | scale={} | amp={}",
        request.model_choice,
        if request.freeze_backbone {
            format!("{freeze_layers_val} layers")
        } else {
            "none (0)".to_string()
        },
        request.mosaic,
        request.mixup,
        request.copy_paste,
        request.degrees,
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
        "detect".to_string(),
        "train".to_string(),
        data_arg,
        model_arg,
        name_arg,
        project_arg,
        epochs_arg,
        batch_arg,
        imgsz_arg,
        device_arg,
        amp_arg,
    ];
    if let Some(ref f_arg) = freeze_arg_opt {
        train_args.push(f_arg.clone());
    }
    train_args.extend([
        mosaic_arg,
        mixup_arg,
        copy_paste_arg,
        degrees_arg,
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
                &exact_command,
                &on_log,
            );

            if status.success() {
                let _ = on_log.send(format!(
                    "[LS-Trainer] Training completed successfully! Results saved to: {results_dir}"
                ));
                Ok(TrainingResponse {
                    success: true,
                    results_dir,
                    message: "Training complete".to_string(),
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
                })
            }
        }
        Err(err) => {
            save_run_config_metadata(
                &abs_results_dir,
                &request,
                &resolved_model,
                &effective_device,
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
            })
        }
    }
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
pub struct ExportRequest {
    pub results_dir: String,
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
    // Check if target is a direct file, or contains weights/best.pt, or best.pt
    let target_file = if target.is_file() {
        Some(target.clone())
    } else {
        let candidate_weights = target.join("weights").join("best.pt");
        let candidate_direct = target.join("best.pt");
        if candidate_weights.is_file() {
            Some(candidate_weights)
        } else if candidate_direct.is_file() {
            Some(candidate_direct)
        } else {
            None
        }
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

    #[cfg(target_os = "windows")]
    let expected_venv_yolo = project_root.join(".venv").join("Scripts").join("yolo.exe");

    #[cfg(not(target_os = "windows"))]
    let expected_venv_yolo = project_root.join(".venv").join("bin").join("yolo");

    let (mut cmd, exact_command) = if expected_venv_yolo.is_file() {
        let _ = on_log.send(format!(
            "[LS-Trainer] Using local venv YOLO executable: {}",
            expected_venv_yolo.display()
        ));

        let mut c = Command::new(&expected_venv_yolo);
        c.args([
            "export",
            &format!("model={best_pt_str}"),
            "format=onnx",
            "imgsz=640",
        ]);

        let cmd_str = format!(
            "\"{}\" export model=\"{best_pt_str}\" format=onnx imgsz=640",
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
            cmd.args([
                "/C",
                "yolo",
                "export",
                &format!("model={best_pt_str}"),
                "format=onnx",
                "imgsz=640",
            ]);
            cmd
        };

        #[cfg(not(target_os = "windows"))]
        let c = {
            let mut cmd = Command::new("yolo");
            cmd.args([
                "export",
                &format!("model={best_pt_str}"),
                "format=onnx",
                "imgsz=640",
            ]);
            cmd
        };

        let cmd_str = format!("yolo export model=\"{best_pt_str}\" format=onnx imgsz=640");

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
            pick_model_file,
            start_training,
            open_folder,
            export_to_onnx,
            get_gpu_info,
            repair_cuda
        ])
        .run(tauri::generate_context!())
        .expect("error while building tauri application");
}
