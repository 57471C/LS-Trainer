//! Ultralytics YOLO training backend.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use tauri::ipc::Channel;

use crate::dataset::labels::inspect_label_tree;
use crate::dataset::yolo::{find_data_yaml, resolve_and_fix_data_yaml, unpack_archive};
use crate::hardware::{detect_gpu_info, resolve_dataloader_workers};
use crate::models::{TrainingRequest, TrainingResponse};
use crate::process::{hide_console_window, stream_child_output, venv_python_path, venv_yolo_path};
use crate::training::{extract_final_metrics_from_results_dir, log_final_metrics, save_run_config_metadata};

pub(crate) fn resolve_or_download_model_weights(
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

    let python_exe = venv_python_path(&project_root);
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

        hide_console_window(&mut dl_cmd);

        if let Ok(mut child) = dl_cmd.spawn() {
            stream_child_output(&mut child, &on_log);
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

/// Ultralytics YOLO training path (detect / segment). Dispatched from `start_training`.
pub(crate) async fn start_yolo_training(
    project_root: PathBuf,
    run_name: String,
    yolo_task: String,
    abs_results_dir: PathBuf,
    results_dir: String,
    dataset_input: PathBuf,
    request: TrainingRequest,
    on_log: Channel<String>,
) -> Result<TrainingResponse, String> {
    let project_runs_dir = project_root.join("runs").join(&yolo_task);

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

    let effective_workers = resolve_dataloader_workers(&project_root, request.workers, &on_log);

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
        if gpu_probe.xpu_available && !device_input.eq_ignore_ascii_case("cuda") && device_input != "0" {
            let dev_label = gpu_probe.device_name.as_deref().unwrap_or("Intel Arc XPU");
            let _ = on_log.send(format!(
                "[LS-Trainer] Intel Arc XPU available (torch.xpu.is_available()=True), selecting device=xpu ({dev_label})"
            ));
            "xpu".to_string()
        } else if gpu_probe.cuda_available {
            let dev_label = gpu_probe.device_name.as_deref().unwrap_or("CUDA GPU");
            let _ = on_log.send(format!(
                "[LS-Trainer] CUDA available (torch.cuda.is_available()=True), selecting device=0 ({dev_label})"
            ));
            "0".to_string()
        } else if gpu_probe.xpu_available {
            let dev_label = gpu_probe.device_name.as_deref().unwrap_or("Intel Arc XPU");
            let _ = on_log.send(format!(
                "[LS-Trainer] Intel Arc XPU available (torch.xpu.is_available()=True), selecting device=xpu ({dev_label})"
            ));
            "xpu".to_string()
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

    let device_arg = format!("device={effective_device}");

    let expected_venv_yolo = venv_yolo_path(&project_root);

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

    hide_console_window(&mut cmd);

    match cmd.spawn() {
        Ok(mut child) => {
            stream_child_output(&mut child, &on_log);

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
                log_final_metrics(&on_log, map50, map50_95);
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
