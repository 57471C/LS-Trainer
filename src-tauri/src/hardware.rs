//! GPU/CPU detection, PyTorch wheel selection (Intel Arc XPU -> CUDA -> CPU), and GPU repair.

use std::path::Path;
use std::process::{Command, Stdio};
use tauri::ipc::Channel;

use crate::process::{get_project_root, get_venv_python, hide_console_window, setup_script_name, stream_child_output, venv_python_path};

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

pub(crate) fn default_recommended_workers() -> u32 {
    4
}

pub(crate) fn default_physical_cores() -> u32 {
    4
}

pub(crate) fn default_logical_cores() -> u32 {
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

pub(crate) fn detect_cpu_worker_info(project_root: &Path) -> (u32, u32, u32) {
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
        hide_console_window(&mut cmd);
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

pub(crate) fn parse_driver_version(raw: &str) -> (String, Option<u32>) {
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

pub(crate) fn shorten_gpu_name(full_name: &str) -> String {
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

pub(crate) fn is_intel_arc_gpu(name: &str) -> bool {
    let lower = name.to_lowercase();
    (lower.contains("intel")
        && (lower.contains("arc")
            || lower.contains("xpu")
            || lower.contains("dg2")
            || lower.contains("battlemage")
            || lower.contains("alchemist")
            || lower.contains("xe")
            || lower.contains("pro a")
            || lower.contains("pro b")))
        || lower.starts_with("arc ")
}

pub(crate) fn is_blackwell_gpu(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower.contains("rtx 50")
        || lower.contains("rtx50")
        || lower.contains("blackwell")
        || lower.contains("gb10")
        || lower.contains("gb20")
        || lower.contains("b100")
        || lower.contains("b200")
}

pub(crate) fn select_pytorch_wheel(
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

pub(crate) fn detect_hardware_gpu() -> (Option<String>, Option<String>, Option<u32>) {
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
        hide_console_window(&mut arc_cmd);
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

    // 1b. Prefer Intel Arc / XPU when present on Linux
    #[cfg(target_os = "linux")]
    {
        // Check xpu-smi discovery
        let mut xpu_cmd = Command::new("xpu-smi");
        xpu_cmd.args(["discovery", "-d", "0"]);
        if let Ok(output) = xpu_cmd.output() {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    let trimmed = line.trim();
                    if let Some(rest) = trimmed.strip_prefix("Device Name:") {
                        let name = rest.trim().to_string();
                        if !name.is_empty() {
                            return (Some(name), None, None);
                        }
                    }
                }
            }
        }

        // Check lspci for Intel Arc / DG2 / Battlemage / Alchemist / Pro GPUs
        for lspci_bin in ["lspci", "/usr/bin/lspci"] {
            if let Ok(output) = Command::new(lspci_bin).output() {
                if output.status.success() {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    for line in stdout.lines() {
                        let lower = line.to_lowercase();
                        if lower.contains("intel")
                            && (lower.contains("arc")
                                || lower.contains("dg2")
                                || lower.contains("battlemage")
                                || lower.contains("alchemist")
                                || lower.contains("xpu")
                                || lower.contains("pro a")
                                || lower.contains("pro b")
                                || lower.contains("graphics")
                                || lower.contains("device 56"))
                        {
                            let parsed_name = if let Some((_, after)) = line.split_once(": ") {
                                after.trim().to_string()
                            } else {
                                line.trim().to_string()
                            };
                            return (Some(parsed_name), None, None);
                        }
                    }
                }
            }
        }

        // Check Linux sysfs DRM devices for Intel vendor 0x8086
        if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
            for entry in entries.flatten() {
                let file_name = entry.file_name().to_string_lossy().to_string();
                if file_name.starts_with("card") && !file_name.contains('-') {
                    let vendor_path = entry.path().join("device").join("vendor");
                    if let Ok(vendor) = std::fs::read_to_string(&vendor_path) {
                        if vendor.trim().eq_ignore_ascii_case("0x8086") {
                            let driver_link = entry.path().join("device").join("driver");
                            if let Ok(target) = std::fs::read_link(&driver_link) {
                                let drv = target.file_name().unwrap_or_default().to_string_lossy();
                                if drv == "i915" || drv == "xe" {
                                    return (Some("Intel Arc GPU".to_string()), None, None);
                                }
                            }
                        }
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
        hide_console_window(&mut smi_cmd);
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
        hide_console_window(&mut smi_l);
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
        hide_console_window(&mut ps_cmd);
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
pub async fn get_gpu_info() -> GpuInfo {
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

pub(crate) fn detect_gpu_info(project_root: &Path) -> GpuInfo {
    let (hw_gpu_name, driver_version, driver_major) = detect_hardware_gpu();
    let (recommended_wheel, _index_url, cuda_label) =
        select_pytorch_wheel(hw_gpu_name.as_deref(), driver_major);
    let (recommended_workers, physical_cores, logical_cores) = detect_cpu_worker_info(project_root);

    let venv_python = venv_python_path(&project_root);

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
        hide_console_window(&mut py_cmd);
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

#[tauri::command]
pub async fn repair_cuda(on_log: Channel<String>) -> Result<GpuInfo, String> {
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

        let venv_python = venv_python_path(&project_root);

        if !venv_python.is_file() {
            let script = setup_script_name();
            let msg = format!(
                "Virtual environment Python not found at '{}'. Please run {script} first.",
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
        hide_console_window(&mut uninstall_cmd);
        if let Ok(mut child) = uninstall_cmd.spawn() {
            stream_child_output(&mut child, &on_log);
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
        hide_console_window(&mut install_cmd);

        match install_cmd.spawn() {
            Ok(mut child) => {
                stream_child_output(&mut child, &on_log);
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

pub(crate) fn is_cuda_available(project_root: &Path) -> bool {
    detect_gpu_info(project_root).cuda_available
}

/// Resolve the DataLoader worker count (manual override capped at 8, else auto-detected) and log it.
pub(crate) fn resolve_dataloader_workers(
    project_root: &Path,
    override_workers: Option<u32>,
    on_log: &Channel<String>,
) -> u32 {
    let (auto_workers, phys_cores, log_cores) = detect_cpu_worker_info(project_root);
    let (effective_workers, worker_source) = match override_workers {
        Some(custom_w) => (custom_w.min(8), "manual override"),
        None => (auto_workers, "auto-detected"),
    };
    let _ = on_log.send(format!(
        "[LS-Trainer] DataLoader workers: {effective_workers} ({worker_source}; {phys_cores} physical / {log_cores} logical CPU cores, capped at 8)"
    ));
    effective_workers
}
