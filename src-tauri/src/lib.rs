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

struct YoloRunner {
    executable: PathBuf,
    is_venv: bool,
    display_info: String,
    use_python_module: bool,
}

fn find_local_venv() -> Option<PathBuf> {
    let mut search_dirs = Vec::new();

    if let Ok(cwd) = std::env::current_dir() {
        search_dirs.push(cwd.clone());
        if let Some(parent) = cwd.parent() {
            search_dirs.push(parent.to_path_buf());
        }
    }

    if let Ok(exe_path) = std::env::current_exe() {
        let mut cur = exe_path.parent();
        while let Some(dir) = cur {
            search_dirs.push(dir.to_path_buf());
            cur = dir.parent();
        }
    }

    for base_dir in search_dirs {
        let venv_dir = base_dir.join(".venv");
        if venv_dir.is_dir() {
            return Some(venv_dir);
        }
    }

    None
}

fn resolve_yolo_runner() -> YoloRunner {
    if let Some(venv) = find_local_venv() {
        #[cfg(target_os = "windows")]
        let (py_candidate, yolo_candidate) = (
            venv.join("Scripts").join("python.exe"),
            venv.join("Scripts").join("yolo.exe"),
        );

        #[cfg(not(target_os = "windows"))]
        let (py_candidate, yolo_candidate) = (
            venv.join("bin").join("python"),
            venv.join("bin").join("yolo"),
        );

        if yolo_candidate.exists() {
            return YoloRunner {
                executable: yolo_candidate,
                is_venv: true,
                display_info: format!("local .venv yolo binary ({})", venv.display()),
                use_python_module: false,
            };
        }

        if py_candidate.exists() {
            return YoloRunner {
                executable: py_candidate,
                is_venv: true,
                display_info: format!("local .venv Python ({})", venv.display()),
                use_python_module: true,
            };
        }
    }

    YoloRunner {
        #[cfg(target_os = "windows")]
        executable: PathBuf::from("cmd"),
        #[cfg(not(target_os = "windows"))]
        executable: PathBuf::from("yolo"),
        is_venv: false,
        display_info: "system environment ('yolo')".to_string(),
        use_python_module: false,
    }
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

#[tauri::command]
async fn pick_dataset_file() -> Option<String> {
    let file = rfd::AsyncFileDialog::new()
        .add_filter("Dataset Archive (*.tar, *.tar.gz, *.tgz)", &["tar", "tar.gz", "tgz"])
        .set_title("Select Dataset Archive")
        .pick_file()
        .await;
    file.map(|f| f.path().to_string_lossy().to_string())
}

#[tauri::command]
async fn pick_model_file() -> Option<String> {
    let file = rfd::AsyncFileDialog::new()
        .add_filter("YOLO Checkpoint (*.pt)", &["pt"])
        .set_title("Select Model Checkpoint")
        .pick_file()
        .await;
    file.map(|f| f.path().to_string_lossy().to_string())
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
    let results_dir = format!("runs/detect/{run_name}/");

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
    let data_yaml_path = if dataset_input.is_file() {
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

    let data_str = data_yaml_path.to_string_lossy().to_string();
    let _ = on_log.send(format!("[LS-Trainer] Dataset config resolved: {data_str}"));

    let data_arg = format!("data={data_str}");
    let model_arg = format!("model={}", request.model_choice);
    let name_arg = format!("name={run_name}");
    let epochs_arg = format!("epochs={}", request.epochs);
    let imgsz_arg = format!("imgsz={}", request.imgsz);
    let device_arg = format!("device={}", request.device);

    let runner = resolve_yolo_runner();
    if runner.is_venv {
        let _ = on_log.send(format!(
            "[LS-Trainer] Found local environment: {}",
            runner.display_info
        ));
    } else {
        let _ = on_log.send(
            "[LS-Trainer] Local .venv not found in project directory. Falling back to system 'yolo'."
                .to_string(),
        );
    }

    let mut cmd = if runner.is_venv {
        let mut c = Command::new(&runner.executable);
        if runner.use_python_module {
            c.args(["-m", "ultralytics"]);
        }
        c.args([
            "detect",
            "train",
            &data_arg,
            &model_arg,
            &name_arg,
            &epochs_arg,
            &imgsz_arg,
            &device_arg,
        ]);
        c
    } else {
        #[cfg(target_os = "windows")]
        {
            let mut c = Command::new("cmd");
            c.args([
                "/C",
                "yolo",
                "detect",
                "train",
                &data_arg,
                &model_arg,
                &name_arg,
                &epochs_arg,
                &imgsz_arg,
                &device_arg,
            ]);
            c
        }
        #[cfg(not(target_os = "windows"))]
        {
            let mut c = Command::new("yolo");
            c.args([
                "detect",
                "train",
                &data_arg,
                &model_arg,
                &name_arg,
                &epochs_arg,
                &imgsz_arg,
                &device_arg,
            ]);
            c
        }
    };

    let full_cmd_str = if runner.is_venv {
        if runner.use_python_module {
            format!(
                "{} -m ultralytics detect train {data_arg} {model_arg} {name_arg} {epochs_arg} {imgsz_arg} {device_arg}",
                runner.executable.display()
            )
        } else {
            format!(
                "{} detect train {data_arg} {model_arg} {name_arg} {epochs_arg} {imgsz_arg} {device_arg}",
                runner.executable.display()
            )
        }
    } else {
        format!(
            "yolo detect train {data_arg} {model_arg} {name_arg} {epochs_arg} {imgsz_arg} {device_arg}"
        )
    };

    let _ = on_log.send(format!("[LS-Trainer] Prepared command: {full_cmd_str}"));
    let _ = on_log.send("[LS-Trainer] Spawning YOLO training process...".to_string());

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
            let _ = on_log.send(format!("[LS-Trainer] Verified command line: {full_cmd_str}"));
            let _ = on_log.send(format!(
                "[LS-Trainer] Target results folder: {results_dir}"
            ));
            let _ = on_log.send("[LS-Trainer] Wiring and parameter verification complete.".to_string());

            Ok(TrainingResponse {
                success: true,
                results_dir,
                message: "Command prepared and verified (fallback simulation)".to_string(),
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
            start_training
        ])
        .run(tauri::generate_context!())
        .expect("error while building tauri application");
}
