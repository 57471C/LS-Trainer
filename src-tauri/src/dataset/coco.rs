//! LS-Marker COCO dataset preparation into the Roboflow layout expected by RF-DETR.

use std::path::{Path, PathBuf};
use tauri::ipc::Channel;

use crate::dataset::yolo::unpack_archive;

pub(crate) fn copy_or_hardlink_file(src: &Path, dst: &Path) -> Result<(), String> {
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

pub(crate) fn find_coco_dataset_root(base_dir: &Path) -> Option<PathBuf> {
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

pub(crate) fn resolve_coco_image_source(
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

pub(crate) fn populate_rfdetr_split_from_coco(
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

pub(crate) fn prepare_rfdetr_coco_dataset(
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
