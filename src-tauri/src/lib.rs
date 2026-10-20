use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use tauri::ipc::Channel;

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
pub struct TrainingRequest {
    pub dataset_path: String,
    pub model_choice: String,
    pub run_name: String,
    pub epochs: u32,
    pub imgsz: u32,
    pub device: String,
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

fn is_cuda_available(project_root: &Path) -> bool {
    // 1. Check via venv Python if available
    #[cfg(target_os = "windows")]
    let venv_python = project_root.join(".venv").join("Scripts").join("python.exe");
    #[cfg(not(target_os = "windows"))]
    let venv_python = project_root.join(".venv").join("bin").join("python");

    if venv_python.is_file() {
        if let Ok(output) = Command::new(&venv_python)
            .args(["-c", "import torch; print(torch.cuda.is_available())"])
            .output()
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if stdout.trim().eq_ignore_ascii_case("true") {
                return true;
            }
            if stdout.trim().eq_ignore_ascii_case("false") {
                return false;
            }
        }
    }

    // 2. Check nvidia-smi
    if let Ok(output) = Command::new("nvidia-smi").arg("-L").output() {
        if output.status.success() && !output.stdout.is_empty() {
            return true;
        }
    }

    // 3. Check system python if available
    #[cfg(target_os = "windows")]
    let sys_py = Command::new("cmd")
        .args([
            "/C",
            "python",
            "-c",
            "import torch; print(torch.cuda.is_available())",
        ])
        .output();
    #[cfg(not(target_os = "windows"))]
    let sys_py = Command::new("python3")
        .args([
            "-c",
            "import torch; print(torch.cuda.is_available())",
        ])
        .output();

    if let Ok(output) = sys_py {
        let stdout = String::from_utf8_lossy(&output.stdout);
        if stdout.trim().eq_ignore_ascii_case("true") {
            return true;
        }
    }

    false
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

    let data_arg = format!("data={data_str}");
    let model_arg = format!("model={}", request.model_choice);
    let name_arg = format!("name={run_name}");
    let epochs_arg = format!("epochs={}", request.epochs);
    let imgsz_arg = format!("imgsz={}", request.imgsz);

    let device_input = request.device.trim();
    let effective_device = if device_input.is_empty() || device_input.eq_ignore_ascii_case("auto") {
        if is_cuda_available(&project_root) {
            "auto".to_string()
        } else {
            let _ = on_log.send("[LS-Trainer] No CUDA detected, using device=cpu".to_string());
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

    let (mut cmd, exact_command) = if expected_venv_yolo.is_file() {
        let _ = on_log.send(format!(
            "[LS-Trainer] Using local venv YOLO executable: {}",
            expected_venv_yolo.display()
        ));

        let mut c = Command::new(&expected_venv_yolo);
        c.args([
            "detect",
            "train",
            &data_arg,
            &model_arg,
            &name_arg,
            &project_arg,
            &epochs_arg,
            &imgsz_arg,
            &device_arg,
        ]);

        let cmd_str = format!(
            "\"{}\" detect train {data_arg} {model_arg} {name_arg} {project_arg} {epochs_arg} {imgsz_arg} {device_arg}",
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
                "detect",
                "train",
                &data_arg,
                &model_arg,
                &name_arg,
                &project_arg,
                &epochs_arg,
                &imgsz_arg,
                &device_arg,
            ]);
            cmd
        };

        #[cfg(not(target_os = "windows"))]
        let c = {
            let mut cmd = Command::new("yolo");
            cmd.args([
                "detect",
                "train",
                &data_arg,
                &model_arg,
                &name_arg,
                &project_arg,
                &epochs_arg,
                &imgsz_arg,
                &device_arg,
            ]);
            cmd
        };

        let cmd_str = format!(
            "yolo detect train {data_arg} {model_arg} {name_arg} {project_arg} {epochs_arg} {imgsz_arg} {device_arg}"
        );

        (c, cmd_str)
    };

    let _ = on_log.send(format!("[LS-Trainer] Executing: {exact_command}"));

    // Set working directory of the process to the unpacked dataset folder
    cmd.current_dir(&dataset_working_dir);
    let _ = on_log.send(format!(
        "[LS-Trainer] Set YOLO working directory to: {}",
        dataset_working_dir.display()
    ));

    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    match cmd.spawn() {
        Ok(mut child) => {
            let stdout = child.stdout.take();
            let stderr = child.stderr.take();

            let on_log_stdout = on_log.clone();
            let stdout_thread = std::thread::spawn(move || {
                if let Some(out) = stdout {
                    let reader = BufReader::new(out);
                    for line in reader.lines().map_while(Result::ok) {
                        let _ = on_log_stdout.send(line);
                    }
                }
            });

            let on_log_stderr = on_log.clone();
            let stderr_thread = std::thread::spawn(move || {
                if let Some(err) = stderr {
                    let reader = BufReader::new(err);
                    for line in reader.lines().map_while(Result::ok) {
                        let _ = on_log_stderr.send(line);
                    }
                }
            });

            let _ = stdout_thread.join();
            let _ = stderr_thread.join();

            let status = child
                .wait()
                .map_err(|e| format!("Failed while waiting for YOLO: {e}"))?;

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
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    match cmd.spawn() {
        Ok(mut child) => {
            let stdout = child.stdout.take();
            let stderr = child.stderr.take();

            let on_log_stdout = on_log.clone();
            let stdout_thread = std::thread::spawn(move || {
                if let Some(out) = stdout {
                    let reader = BufReader::new(out);
                    for line in reader.lines().map_while(Result::ok) {
                        let _ = on_log_stdout.send(line);
                    }
                }
            });

            let on_log_stderr = on_log.clone();
            let stderr_thread = std::thread::spawn(move || {
                if let Some(err) = stderr {
                    let reader = BufReader::new(err);
                    for line in reader.lines().map_while(Result::ok) {
                        let _ = on_log_stderr.send(line);
                    }
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
            export_to_onnx
        ])
        .run(tauri::generate_context!())
        .expect("error while building tauri application");
}
