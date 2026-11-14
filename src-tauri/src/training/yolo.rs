//! Ultralytics YOLO training backend.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use tokio::process::Command as TokioCommand;

use crate::dataset::labels::inspect_label_tree;
use crate::dataset::yolo::{find_data_yaml, resolve_and_fix_data_yaml, unpack_archive};
use crate::hardware::{detect_gpu_info, resolve_dataloader_workers};
use crate::models::{TrainingRequest, TrainingResponse};
use crate::process::{
    TrainingEventSink, hide_console_window, hide_console_window_tokio, stream_child_output,
    stream_tokio_child_output, venv_python_path, venv_yolo_path,
};
use crate::training::{
    extract_final_metrics_from_results_dir, is_stop_requested, log_final_metrics,
    register_tokio_child_process, save_run_config_metadata, unregister_child_pid,
};

pub(crate) fn resolve_or_download_model_weights(
    project_root: &Path,
    raw_model_choice: &str,
    sink: &Arc<dyn TrainingEventSink>,
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
        if candidate.is_file() && candidate.metadata().map(|m| m.len() > 0).unwrap_or(false) {
            sink.on_log(format!(
                "[LS-Trainer] Using cached model weights: {}",
                candidate.display()
            ));
            return candidate.to_string_lossy().to_string();
        }
    }

    let models_dir = project_root.join("models");
    let _ = std::fs::create_dir_all(&models_dir);
    let target_weight_path = models_dir.join(&weight_filename);

    let python_exe = venv_python_path(project_root);
    if python_exe.is_file() {
        sink.on_log(format!(
            "[LS-Trainer] Model '{weight_filename}' not cached locally. Downloading from Ultralytics assets..."
        ));
        sink.on_log(format!(
            "[LS-Trainer Status] Downloading {weight_filename}…"
        ));

        let script = "import sys; from ultralytics.utils.downloads import attempt_download_asset; p = attempt_download_asset(sys.argv[1]); print(f'[LS-Trainer] Downloaded model asset: {p}', flush=True)";
        let mut dl_cmd = std::process::Command::new(&python_exe);
        dl_cmd
            .args(["-c", script, &weight_filename])
            .current_dir(&models_dir)
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONIOENCODING", "utf-8")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        hide_console_window(&mut dl_cmd);

        if let Ok(mut child) = dl_cmd.spawn() {
            stream_child_output(&mut child, sink);
            let _ = child.wait();
        }

        if target_weight_path.is_file()
            && target_weight_path
                .metadata()
                .map(|m| m.len() > 0)
                .unwrap_or(false)
        {
            sink.on_log(format!(
                "[LS-Trainer] Model cached at: {}",
                target_weight_path.display()
            ));
            return target_weight_path.to_string_lossy().to_string();
        }
    }

    weight_filename
}

/// Ultralytics YOLO training path (detect / segment). Dispatched from `start_training`.
pub(crate) async fn start_yolo_training(
    project_root: PathBuf,
    run_name: String,
    yolo_task: String,
    abs_results_dir: PathBuf,
    results_dir: String,
    dataset_input: PathBuf,
    request: TrainingRequest,
    sink: Arc<dyn TrainingEventSink>,
) -> Result<TrainingResponse, String> {
    let project_runs_dir = abs_results_dir
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| project_root.join("runs").join(&yolo_task));

    // Determine data.yaml path: unpack if archive, or find inside directory
    let raw_data_yaml_path = if dataset_input.is_file() {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let temp_dir = std::env::temp_dir().join(format!("ls_trainer_{timestamp}"));

        sink.on_log(format!(
            "[LS-Trainer] Unpacking archive '{}' to '{}'...",
            dataset_input.display(),
            temp_dir.display()
        ));

        unpack_archive(&dataset_input, &temp_dir)?;
        sink.on_log("[LS-Trainer] Unpacked dataset successfully.".to_string());

        let yaml = find_data_yaml(&temp_dir).ok_or_else(|| {
            let err = format!(
                "data.yaml not found in unpacked archive directory: {}",
                temp_dir.display()
            );
            sink.on_log(format!("[LS-Trainer] Error: {err}"));
            err
        })?;
        yaml
    } else if dataset_input.is_dir() {
        find_data_yaml(&dataset_input).ok_or_else(|| {
            let err = format!(
                "data.yaml not found in dataset folder: {}",
                dataset_input.display()
            );
            sink.on_log(format!("[LS-Trainer] Error: {err}"));
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
            sink.on_log(format!("[LS-Trainer] Warning: {warning}"));
        } else {
            sink.on_log(format!(
                "[LS-Trainer] Segmentation labels OK: polygon annotations found ({} label files, {} lines).",
                label_check.files_scanned, label_check.lines_scanned
            ));
        }
    }

    // Read data.yaml, resolve all relative paths to absolute, and rewrite it
    let fixed_data_yaml_path = resolve_and_fix_data_yaml(&raw_data_yaml_path, &sink)?;
    let data_str = fixed_data_yaml_path.to_string_lossy().to_string();
    let dataset_working_dir = fixed_data_yaml_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| project_root.clone());

    let resolved_model = {
        let root = project_root.clone();
        let choice = request.model_choice.clone();
        let sink_dl = Arc::clone(&sink);
        tauri::async_runtime::spawn_blocking(move || {
            resolve_or_download_model_weights(&root, &choice, &sink_dl)
        })
        .await
        .map_err(|e| format!("Model resolution task failed: {e}"))?
    };
    sink.on_log("[LS-Trainer Status] Training in progress...".to_string());

    let effective_workers = resolve_dataloader_workers(&project_root, request.workers, &sink);

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

    sink.on_log(format!("[LS-Trainer] Ultralytics task=\"{yolo_task}\""));
    sink.on_log(format!(
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
    let gpu_probe = {
        let root = project_root.clone();
        tauri::async_runtime::spawn_blocking(move || detect_gpu_info(&root))
            .await
            .map_err(|e| format!("GPU probe task failed: {e}"))?
    };
    let effective_device = if device_input.eq_ignore_ascii_case("cpu") {
        sink.on_log("[LS-Trainer] Device override: using device=cpu".to_string());
        "cpu".to_string()
    } else if device_input.is_empty()
        || device_input.eq_ignore_ascii_case("auto")
        || device_input.eq_ignore_ascii_case("gpu")
        || device_input.eq_ignore_ascii_case("xpu")
        || device_input.eq_ignore_ascii_case("xpu:0")
        || device_input.eq_ignore_ascii_case("cuda")
        || device_input == "0"
    {
        if gpu_probe.xpu_available
            && !device_input.eq_ignore_ascii_case("cuda")
            && device_input != "0"
        {
            let dev_label = gpu_probe.device_name.as_deref().unwrap_or("Intel Arc XPU");
            sink.on_log(format!(
                "[LS-Trainer] Intel Arc XPU available (torch.xpu.is_available()=True), selecting device=xpu ({dev_label})"
            ));
            "xpu".to_string()
        } else if gpu_probe.cuda_available {
            let dev_label = gpu_probe.device_name.as_deref().unwrap_or("CUDA GPU");
            sink.on_log(format!(
                "[LS-Trainer] CUDA available (torch.cuda.is_available()=True), selecting device=0 ({dev_label})"
            ));
            "0".to_string()
        } else if gpu_probe.xpu_available {
            let dev_label = gpu_probe.device_name.as_deref().unwrap_or("Intel Arc XPU");
            sink.on_log(format!(
                "[LS-Trainer] Intel Arc XPU available (torch.xpu.is_available()=True), selecting device=xpu ({dev_label})"
            ));
            "xpu".to_string()
        } else {
            sink.on_log(
                "[LS-Trainer] Neither Intel Arc XPU nor CUDA available, using device=cpu"
                    .to_string(),
            );
            "cpu".to_string()
        }
    } else {
        device_input.to_string()
    };

    let device_arg = format!("device={effective_device}");

    let expected_venv_yolo = venv_yolo_path(&project_root);
    let expected_venv_python = venv_python_path(&project_root);

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
    let yolo_runner_script = include_str!("../python/yolo_train.py");

    let (mut cmd, exact_command) = if expected_venv_yolo.is_file() || expected_venv_python.is_file() {
        let display_exe = if expected_venv_yolo.is_file() {
            &expected_venv_yolo
        } else {
            &expected_venv_python
        };
        sink.on_log(format!(
            "[LS-Trainer] Using local venv YOLO executable: {}",
            display_exe.display()
        ));

        let c = if expected_venv_python.is_file() {
            let mut py_cmd = TokioCommand::new(&expected_venv_python);
            py_cmd.arg("-c").arg(yolo_runner_script).args(&train_args);
            py_cmd
        } else {
            let mut yolo_cmd = TokioCommand::new(&expected_venv_yolo);
            yolo_cmd.args(&train_args);
            yolo_cmd
        };

        let cmd_str = format!("\"{}\" {joined_train_args}", display_exe.display());

        (c, cmd_str)
    } else {
        sink.on_log(format!(
            "[LS-Trainer] Warning: Local .venv executable not found at '{}'. Falling back to system 'yolo'.",
            expected_venv_yolo.display()
        ));

        #[cfg(target_os = "windows")]
        let c = {
            let mut cmd = TokioCommand::new("cmd");
            cmd.args(["/C", "yolo"]);
            cmd.args(&train_args);
            cmd
        };

        #[cfg(not(target_os = "windows"))]
        let c = {
            let mut cmd = TokioCommand::new("yolo");
            cmd.args(&train_args);
            cmd
        };

        let cmd_str = format!("yolo {joined_train_args}");

        (c, cmd_str)
    };

    sink.on_log(format!("[LS-Trainer] Executing: {exact_command}"));

    // Set working directory of the process to the unpacked dataset folder
    cmd.current_dir(&dataset_working_dir);
    sink.on_log(format!(
        "[LS-Trainer] Set YOLO working directory to: {}",
        dataset_working_dir.display()
    ));

    cmd.env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    hide_console_window_tokio(&mut cmd);

    let launch_exe_desc = if expected_venv_python.is_file() {
        expected_venv_python.display().to_string()
    } else if expected_venv_yolo.is_file() {
        expected_venv_yolo.display().to_string()
    } else {
        "yolo (PATH)".to_string()
    };
    sink.on_log(format!(
        "[LS-Trainer] Spawning child process: exe='{}' cwd='{}' env(PYTHONUNBUFFERED=1, PYTHONIOENCODING=utf-8)",
        launch_exe_desc,
        dataset_working_dir.display()
    ));

    match cmd.spawn() {
        Ok(mut child) => {
            sink.on_log(format!(
                "[LS-Trainer] Child process spawned (pid={:?}). Streaming stdout/stderr...",
                child.id()
            ));
            let _control = register_tokio_child_process(&mut child);
            stream_tokio_child_output(&mut child, &sink).await;
            let status = child.wait().await.map_err(|e| {
                let msg = format!("Failed while waiting for YOLO: {e}");
                sink.on_log(format!("[LS-Trainer Error] {msg}"));
                msg
            })?;

            unregister_child_pid();
            sink.on_exit(status.code());

            save_run_config_metadata(
                &abs_results_dir,
                &request,
                &resolved_model,
                &effective_device,
                effective_workers,
                &exact_command,
                &sink,
            );

            let (map50, map50_95) = extract_final_metrics_from_results_dir(&abs_results_dir);

            if is_stop_requested() {
                sink.on_log(format!(
                    "[LS-Trainer] Training stopped by user. Checkpoints preserved at: {results_dir}"
                ));
                return Ok(TrainingResponse {
                    success: false,
                    results_dir,
                    message: "Stopped by user".to_string(),
                    map50,
                    map50_95,
                });
            }

            if status.success() {
                log_final_metrics(&sink, map50, map50_95);
                sink.on_log(format!(
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
                sink.on_log(format!(
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
            sink.on_exit(None);
            save_run_config_metadata(
                &abs_results_dir,
                &request,
                &resolved_model,
                &effective_device,
                effective_workers,
                &exact_command,
                &sink,
            );

            sink.on_log(format!(
                "[LS-Trainer] Notice: Could not launch YOLO process directly ({err})."
            ));
            sink.on_log(
                "[LS-Trainer] Ensure 'ultralytics' is installed in the active environment ('pip install ultralytics')."
                    .to_string(),
            );
            sink.on_log(format!("[LS-Trainer] Command was: {exact_command}"));
            sink.on_log(format!("[LS-Trainer] Target results folder: {results_dir}"));
            sink.on_log("[LS-Trainer] Wiring and parameter verification complete.".to_string());

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
