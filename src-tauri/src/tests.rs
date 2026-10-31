//! Unit tests for label scanning, task resolution, RF-DETR mapping, workers, and wheel selection.

use crate::dataset::labels::{LabelStats, result_from_stats, scan_coco_json_text, scan_label_text, scan_tar_archive};
use crate::hardware::{compute_recommended_workers, is_intel_arc_gpu, select_pytorch_wheel};
use crate::models::TrainingRequest;
use crate::rfdetr::{resolve_rfdetr_class_and_pretrain, snap_rfdetr_resolution};
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
