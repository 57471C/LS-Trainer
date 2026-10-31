//! YOLO dataset handling: archive unpacking and `data.yaml` path resolution.

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use tauri::ipc::Channel;

pub(crate) fn unpack_archive(archive_path: &Path, target_dir: &Path) -> Result<(), String> {
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

pub(crate) fn find_data_yaml(dir: &Path) -> Option<PathBuf> {
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

pub(crate) fn resolve_single_path(base_dir: &Path, raw: &str) -> (String, bool) {
    let trimmed = raw.trim().trim_matches(|c| c == '\'' || c == '"');
    let p = Path::new(trimmed);
    if p.is_absolute() {
        return (trimmed.replace('\\', "/"), false);
    }
    let combined = base_dir.join(p);
    let resolved_str = combined.to_string_lossy().replace('\\', "/");
    (resolved_str, true)
}

pub(crate) fn resolve_and_fix_data_yaml(
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
