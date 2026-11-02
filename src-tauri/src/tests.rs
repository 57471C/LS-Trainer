//! Unit tests for label scanning, task resolution, RF-DETR mapping, workers, and wheel selection.

use crate::dataset::labels::{LabelStats, result_from_stats, scan_coco_json_text, scan_label_text, scan_tar_archive};
use crate::hardware::{compute_recommended_workers, is_intel_arc_gpu, select_pytorch_wheel};
use crate::models::TrainingRequest;
use crate::rfdetr::{
    resolve_rfdetr_class_and_pretrain, snap_rfdetr_resolution, verify_rfdetr_dependencies,
};
use crate::rtdetr::{
    find_rtdetr_checkpoint_in_dir, is_rtdetr_request, is_rtdetr_run_dir, resolve_rtdetr_model_meta,
};
use crate::training::resolve_train_task;

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
fn rfdetr_dependency_verification() {
    let project_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    let result = verify_rfdetr_dependencies(&project_root);
    assert!(
        result.is_ok(),
        "Expected torch_linear_assignment to be verified in venv: {:?}",
        result
    );

    let bogus_root = std::env::temp_dir().join(format!("ls_trainer_bogus_{}", std::process::id()));
    let bogus_result = verify_rfdetr_dependencies(&bogus_root);
    assert!(bogus_result.is_err(), "Expected failure when .venv is missing");
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
    assert!(is_intel_arc_gpu("Intel Corporation Battlemage [Arc B580]"));
    assert!(is_intel_arc_gpu("Intel Corporation DG2 [Arc Pro A60]"));
    assert!(is_intel_arc_gpu("Intel Arc Pro Graphics"));
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

#[test]
fn rtdetr_request_classification_and_checkpoint_detection() {
    let mut req = TrainingRequest {
        dataset_path: "datasets/coco".to_string(),
        model_family: "RT-DETR".to_string(),
        model_choice: "rtdetr-v2-r18".to_string(),
        run_name: "test-rtdetr".to_string(),
        epochs: 50,
        batch: 4,
        imgsz: 640,
        workers: None,
        device: "xpu".to_string(),
        amp: true,
        freeze_backbone: false,
        freeze_layers: 0,
        mosaic: 0.0,
        mixup: 0.0,
        copy_paste: 0.0,
        degrees: 0.0,
        flipud: 0.0,
        translate: 0.0,
        scale: 0.0,
        close_mosaic: 0,
        task: "detect".to_string(),
    };
    assert!(is_rtdetr_request(&req));

    req.model_family = "YOLO".to_string();
    assert!(is_rtdetr_request(&req)); // model_choice contains rtdetr

    req.model_choice = "yolo11s.pt".to_string();
    assert!(!is_rtdetr_request(&req));

    let temp_dir = std::env::temp_dir().join(format!("ls_trainer_test_rtdetr_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&temp_dir);
    let cfg_file = temp_dir.join("training_config.json");
    let _ = std::fs::write(&cfg_file, r#"{"model_family": "RT-DETR", "model_name": "RT-DETR v2-R18"}"#);
    let best_pth = temp_dir.join("best.pth");
    let _ = std::fs::write(&best_pth, b"dummy checkpoint");

    assert!(is_rtdetr_run_dir(&temp_dir));
    assert_eq!(find_rtdetr_checkpoint_in_dir(&temp_dir), Some(best_pth));

    let _ = std::fs::remove_dir_all(&temp_dir);
}


#[test]
fn rtdetr_model_meta_resolution() {
    let (w18, l18) = resolve_rtdetr_model_meta("rtdetr-v2-r18");
    assert_eq!(w18, "PekingU/rtdetr_v2_r18vd");
    assert_eq!(l18, "RT-DETR v2-R18");

    let (wl, ll) = resolve_rtdetr_model_meta("rtdetr-l.pt");
    assert_eq!(wl, "PekingU/rtdetr_v2_r50vd");
    assert_eq!(ll, "RT-DETR-L");

    let (wl_alias, ll_alias) = resolve_rtdetr_model_meta("rtdetr-l");
    assert_eq!(wl_alias, "PekingU/rtdetr_v2_r50vd");
    assert_eq!(ll_alias, "RT-DETR-L");

    let (wx, lx) = resolve_rtdetr_model_meta("rtdetr-x.pt");
    assert_eq!(wx, "PekingU/rtdetr_v2_r101vd");
    assert_eq!(lx, "RT-DETR-X");

    let (wx_alias, lx_alias) = resolve_rtdetr_model_meta("rtdetr-x");
    assert_eq!(wx_alias, "PekingU/rtdetr_v2_r101vd");
    assert_eq!(lx_alias, "RT-DETR-X");
}

#[test]
fn test_training_stop_lifecycle() {
    tauri::async_runtime::block_on(async {
        use crate::training::{
            clear_active_training, is_stop_requested, register_active_training,
            stop_training,
        };

        let temp_run_dir = std::env::temp_dir().join("ls_trainer_test_run");
        let _ = std::fs::create_dir_all(&temp_run_dir);

        // Initial state: not stopped
        clear_active_training();
        assert!(!is_stop_requested());

        // Register active training
        register_active_training(temp_run_dir.clone());
        assert!(!is_stop_requested());

        // Stop training
        let stopped = stop_training().await.expect("stop_training call failed");
        assert!(stopped);
        assert!(is_stop_requested());

        // Verify .stop_requested file exists in run dir
        assert!(temp_run_dir.join(".stop_requested").is_file());

        // Clear active training
        clear_active_training();

        // Clean up
        let _ = std::fs::remove_dir_all(&temp_run_dir);
    });
}

#[tokio::test]
async fn test_training_stop_lifecycle() {
    use std::path::PathBuf;
    use crate::training::{
        clear_active_training, is_stop_requested, register_active_training, register_child_pid,
        stop_training,
    };

    let temp_run_dir = std::env::temp_dir().join("ls_trainer_test_run");
    let _ = std::fs::create_dir_all(&temp_run_dir);

    // Initial state: not stopped
    clear_active_training();
    assert!(!is_stop_requested());

    // Register active training
    register_active_training(temp_run_dir.clone());
    assert!(!is_stop_requested());

    // Stop training
    let stopped = stop_training().await.expect("stop_training call failed");
    assert!(stopped);
    assert!(is_stop_requested());

    // Verify .stop_requested file exists in run dir
    assert!(temp_run_dir.join(".stop_requested").is_file());

    // Clear active training
    clear_active_training();

    // Clean up
    let _ = std::fs::remove_dir_all(&temp_run_dir);
}
