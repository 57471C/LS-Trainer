//! Label sampling for segmentation warnings (YOLO txt polygons/boxes and COCO JSON).

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

pub(crate) const MAX_LABEL_FILES: u32 = 80;

pub(crate) const MAX_LINES_PER_FILE: u32 = 40;

#[derive(serde::Serialize, Debug, Clone)]
pub struct LabelCheckResult {
    pub polygons_found: bool,
    pub bboxes_found: bool,
    pub files_scanned: u32,
    pub lines_scanned: u32,
    pub warning: Option<String>,
}

#[derive(Default)]
pub(crate) struct LabelStats {
    pub(crate) polygons: u32,
    pub(crate) bboxes: u32,
    pub(crate) files_scanned: u32,
    pub(crate) lines_scanned: u32,
}

/// YOLO detect lines are `class xc yc w h` (5 values).
/// Segmentation polygons are `class x1 y1 x2 y2 ...` (more than 5 values).
pub(crate) fn classify_annotation_line(line: &str) -> Option<bool> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let mut count = 0u32;
    for token in line.split_whitespace() {
        if token.parse::<f64>().is_err() {
            return None;
        }
        count += 1;
    }
    if count == 5 {
        Some(false)
    } else if count > 5 {
        Some(true)
    } else {
        None
    }
}

pub(crate) fn path_is_candidate_label(path: &str) -> bool {
    let lower = path.replace('\\', "/").to_ascii_lowercase();
    if lower.contains("__macosx/") || !lower.ends_with(".txt") {
        return false;
    }
    let file_name = lower.rsplit('/').next().unwrap_or(&lower);
    if file_name.starts_with("._") {
        return false;
    }
    if matches!(
        file_name,
        "classes.txt" | "class_names.txt" | "notes.txt" | "readme.txt" | "license.txt"
    ) {
        return false;
    }
    let in_labels = lower.contains("/labels/") || lower.starts_with("labels/");
    if !in_labels
        && matches!(
            file_name,
            "train.txt" | "val.txt" | "valid.txt" | "test.txt" | "data.txt"
        )
    {
        return false;
    }
    true
}

pub(crate) fn scan_label_text(text: &str, stats: &mut LabelStats) {
    let mut lines_in_file = 0u32;
    let mut counted = false;
    for line in text.lines() {
        if lines_in_file >= MAX_LINES_PER_FILE {
            break;
        }
        match classify_annotation_line(line) {
            Some(true) => {
                stats.polygons += 1;
                stats.lines_scanned += 1;
                lines_in_file += 1;
                counted = true;
            }
            Some(false) => {
                stats.bboxes += 1;
                stats.lines_scanned += 1;
                lines_in_file += 1;
                counted = true;
            }
            None => {}
        }
    }
    if counted {
        stats.files_scanned += 1;
    }
}

pub(crate) fn label_sample_complete(stats: &LabelStats) -> bool {
    if stats.polygons > 0 && stats.bboxes > 0 {
        return true;
    }
    stats.files_scanned >= MAX_LABEL_FILES
}

pub(crate) fn result_from_stats(stats: &LabelStats) -> LabelCheckResult {
    let warning = if stats.lines_scanned == 0 {
        Some(
            "No polygon annotations were found in the dataset labels. Instance segmentation needs label lines with more than 5 values (class plus polygon points)."
                .to_string(),
        )
    } else if stats.polygons == 0 && stats.bboxes > 0 {
        Some(format!(
            "Plain bounding-box labels detected ({} lines with 5 values, 0 polygon lines). Instance segmentation needs polygon annotations (more than 5 values per line).",
            stats.bboxes
        ))
    } else if stats.bboxes > 0 {
        Some(format!(
            "Mixed labels detected: {} polygon lines and {} plain bounding-box lines (5 values). Segmentation training expects polygons (more than 5 values per line).",
            stats.polygons, stats.bboxes
        ))
    } else {
        None
    };

    LabelCheckResult {
        polygons_found: stats.polygons > 0,
        bboxes_found: stats.bboxes > 0,
        files_scanned: stats.files_scanned,
        lines_scanned: stats.lines_scanned,
        warning,
    }
}

pub(crate) fn path_is_candidate_coco_json(path: &str) -> bool {
    let lower = path.replace('\\', "/").to_ascii_lowercase();
    if lower.contains("__macosx/") || !lower.ends_with(".json") {
        return false;
    }
    let file_name = lower.rsplit('/').next().unwrap_or(&lower);
    if file_name.starts_with("._") {
        return false;
    }
    file_name.starts_with("instances_")
        || file_name.ends_with(".coco.json")
        || file_name == "annotations.json"
}

pub(crate) fn scan_coco_json_text(text: &str, stats: &mut LabelStats) {
    let Ok(val) = serde_json::from_str::<serde_json::Value>(text) else {
        return;
    };
    let Some(anns) = val.get("annotations").and_then(|v| v.as_array()) else {
        return;
    };
    let mut counted = false;
    let mut items_in_file = 0u32;
    for ann in anns {
        if items_in_file >= MAX_LINES_PER_FILE * 4 {
            break;
        }
        let has_polygon = ann
            .get("segmentation")
            .map(|seg| match seg {
                serde_json::Value::Array(polys) => polys.iter().any(|p| {
                    p.as_array()
                        .map(|coords| coords.len() >= 6)
                        .unwrap_or(false)
                }),
                serde_json::Value::Object(obj) => obj.contains_key("counts"),
                _ => false,
            })
            .unwrap_or(false);
        let has_bbox = ann
            .get("bbox")
            .and_then(|b| b.as_array())
            .map(|arr| arr.len() >= 4)
            .unwrap_or(false);

        if has_polygon {
            stats.polygons += 1;
            stats.lines_scanned += 1;
            items_in_file += 1;
            counted = true;
        } else if has_bbox {
            stats.bboxes += 1;
            stats.lines_scanned += 1;
            items_in_file += 1;
            counted = true;
        }
    }
    if counted {
        stats.files_scanned += 1;
    }
}

pub(crate) fn inspect_label_tree(root: &Path) -> LabelCheckResult {
    let mut stats = LabelStats::default();
    let mut coco_files: Vec<PathBuf> = Vec::new();
    for entry in walkdir::WalkDir::new(root)
        .max_depth(6)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if label_sample_complete(&stats) {
            break;
        }
        if !entry.file_type().is_file() {
            continue;
        }
        let path_str = entry.path().to_string_lossy().to_string();
        if path_is_candidate_coco_json(&path_str) {
            coco_files.push(entry.path().to_path_buf());
            continue;
        }
        if !path_is_candidate_label(&path_str) {
            continue;
        }
        let Ok(bytes) = std::fs::read(entry.path()) else {
            continue;
        };
        let slice = if bytes.len() > 256 * 1024 {
            &bytes[..256 * 1024]
        } else {
            &bytes[..]
        };
        let text = String::from_utf8_lossy(slice);
        scan_label_text(&text, &mut stats);
    }
    if stats.lines_scanned == 0 && !coco_files.is_empty() {
        for coco_path in coco_files {
            if label_sample_complete(&stats) {
                break;
            }
            if let Ok(text) = std::fs::read_to_string(&coco_path) {
                scan_coco_json_text(&text, &mut stats);
            }
        }
    }
    result_from_stats(&stats)
}

pub(crate) fn scan_tar_archive<R: Read>(mut archive: tar::Archive<R>) -> Result<LabelCheckResult, String> {
    let mut stats = LabelStats::default();
    let mut coco_payloads: Vec<String> = Vec::new();
    let entries = archive
        .entries()
        .map_err(|e| format!("Failed to read dataset archive: {e}"))?;
    for entry in entries {
        if label_sample_complete(&stats) {
            break;
        }
        let mut entry = entry.map_err(|e| format!("Failed to read archive entry: {e}"))?;
        let path_str = {
            let path = entry
                .path()
                .map_err(|e| format!("Failed to read archive entry path: {e}"))?;
            path.to_string_lossy().to_string()
        };
        if path_is_candidate_coco_json(&path_str) && coco_payloads.len() < 4 {
            let mut buf = Vec::new();
            let mut limited = (&mut entry).take(16 * 1024 * 1024);
            let _ = limited.read_to_end(&mut buf);
            if let Ok(text) = String::from_utf8(buf) {
                coco_payloads.push(text);
            }
            continue;
        }
        if !path_is_candidate_label(&path_str) {
            continue;
        }
        let mut buf = Vec::new();
        let mut limited = (&mut entry).take(256 * 1024);
        let _ = limited.read_to_end(&mut buf);
        let text = String::from_utf8_lossy(&buf);
        scan_label_text(&text, &mut stats);
    }
    if stats.lines_scanned == 0 && !coco_payloads.is_empty() {
        for text in &coco_payloads {
            scan_coco_json_text(text, &mut stats);
        }
    }
    Ok(result_from_stats(&stats))
}

pub(crate) fn inspect_archive_labels(archive_path: &Path) -> Result<LabelCheckResult, String> {
    let file = File::open(archive_path)
        .map_err(|e| format!("Failed to open dataset archive: {e}"))?;
    let path_str = archive_path.to_string_lossy().to_lowercase();
    if path_str.ends_with(".tar.gz") || path_str.ends_with(".tgz") {
        let gz = flate2::read::GzDecoder::new(BufReader::new(file));
        scan_tar_archive(tar::Archive::new(gz))
    } else if path_str.ends_with(".tar") {
        scan_tar_archive(tar::Archive::new(BufReader::new(file)))
    } else {
        Err(format!(
            "Unsupported dataset archive format: {}",
            archive_path.display()
        ))
    }
}

pub(crate) fn inspect_dataset_path(path: &Path) -> Result<LabelCheckResult, String> {
    if !path.exists() {
        return Err(format!(
            "Dataset path does not exist: {}",
            path.display()
        ));
    }
    if path.is_dir() {
        return Ok(inspect_label_tree(path));
    }
    let lower = path.to_string_lossy().to_lowercase();
    if lower.ends_with(".yaml") || lower.ends_with(".yml") || lower.ends_with(".json") {
        let mut root = path.parent().unwrap_or(path);
        if let Some(dir_name) = root.file_name().map(|n| n.to_string_lossy().to_lowercase()) {
            if matches!(dir_name.as_str(), "annotations" | "train" | "valid" | "val" | "test") {
                if let Some(grandparent) = root.parent() {
                    root = grandparent;
                }
            }
        }
        return Ok(inspect_label_tree(root));
    }
    if lower.ends_with(".tar") || lower.ends_with(".tar.gz") || lower.ends_with(".tgz") {
        return inspect_archive_labels(path);
    }
    Err(format!(
        "Unsupported dataset path for label check: {}",
        path.display()
    ))
}

#[tauri::command]
pub async fn inspect_dataset_labels(dataset_path: String) -> Result<LabelCheckResult, String> {
    tauri::async_runtime::spawn_blocking(move || inspect_dataset_path(Path::new(&dataset_path)))
        .await
        .map_err(|e| format!("Label check failed: {e}"))?
}
