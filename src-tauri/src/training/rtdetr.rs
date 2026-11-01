//! RT-DETR training backend (runs `python/rtdetr_train.py` in the project venv).

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use tauri::ipc::Channel;

use crate::dataset::coco::prepare_rfdetr_coco_dataset;
use crate::hardware::{detect_gpu_info, resolve_dataloader_workers};
use crate::models::{TrainingRequest, TrainingResponse};
use crate::process::{get_venv_python, hide_console_window, stream_child_output};
use crate::training::{extract_final_metrics_from_results_dir, log_final_metrics, save_run_config_metadata};

pub(crate) async fn start_rtdetr_training(
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

    let models_dir = project_root.join("models");
    let _ = std::fs::create_dir_all(&models_dir);

    let custom_path = Path::new(request.model_choice.trim());
    let resolved_pretrain = if custom_path.is_file() {
        custom_path.to_string_lossy().to_string()
    } else {
        "PekingU/rtdetr_v2_r18vd".to_string()
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

    let effective_workers = resolve_dataloader_workers(&project_root, request.workers, &on_log);

    let _ = std::fs::create_dir_all(&abs_results_dir);
    let runner_script_path = abs_results_dir.join("ls_trainer_rtdetr_run.py");
    let runner_script = include_str!("../python/rtdetr_train.py");

    std::fs::write(&runner_script_path, runner_script)
        .map_err(|e| format!("Failed to write RT-DETR runner script: {e}"))?;

    let runner_cfg_path = abs_results_dir.join("ls_trainer_rtdetr_args.json");
    let runner_cfg = serde_json::json!({
        "model_name": "RT-DETR v2-R18",
        "models_dir": models_dir.to_string_lossy(),
        "dataset_dir": coco_root.to_string_lossy(),
        "output_dir": abs_results_dir.to_string_lossy(),
        "epochs": request.epochs,
        "batch_size": request.batch,
        "resolution": request.imgsz,
        "device": effective_device,
        "num_workers": effective_workers,
        "pretrain_weights": resolved_pretrain,
        "freeze_backbone": request.freeze_backbone,
        "amp": request.amp,
        "task": yolo_task,
    });
    std::fs::write(
        &runner_cfg_path,
        serde_json::to_string_pretty(&runner_cfg).unwrap_or_default(),
    )
    .map_err(|e| format!("Failed to write RT-DETR config JSON: {e}"))?;

    let python_exe = get_venv_python(&project_root).unwrap_or_else(|| PathBuf::from("python"));
    let exact_command = format!(
        "\"{}\" \"{}\" \"{}\"",
        python_exe.display(),
        runner_script_path.display(),
        runner_cfg_path.display()
    );

    let _ = on_log.send("[LS-Trainer Status] Training in progress...".to_string());
    let _ = on_log.send(format!(
        "[LS-Trainer] Active config: family=RT-DETR | task={} | model=RT-DETR v2-R18 ({}) | resolution={} | epochs={} | batch={} | workers={} | freeze_backbone={} | amp={} | device={}",
        yolo_task,
        resolved_pretrain,
        request.imgsz,
        request.epochs,
        request.batch,
        effective_workers,
        request.freeze_backbone,
        request.amp,
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

    hide_console_window(&mut cmd);

    match cmd.spawn() {
        Ok(mut child) => {
            stream_child_output(&mut child, &on_log);

            let status = child
                .wait()
                .map_err(|e| format!("Failed while waiting for RT-DETR process: {e}"))?;

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
                log_final_metrics(&on_log, map50, map50_95);
                let _ = on_log.send(format!(
                    "[LS-Trainer] RT-DETR training completed successfully! Results saved to: {results_dir} (run '{run_name}')"
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
                    "[LS-Trainer] RT-DETR exited with code: {:?}",
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
        Err(err) => Err(format!("Failed to launch RT-DETR process: {err}")),
    }
}

