//! Persistent store for unpacked dataset archives (`<app-local-data>/LS-Trainer/unpacked/`).

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::process::to_display_path;

static LAST_UNPACKED_DIR: Mutex<Option<PathBuf>> = Mutex::new(None);

#[derive(serde::Serialize, Debug, Clone)]
pub struct UnpackedDatasetInfo {
    pub path: String,
    pub image_count: u64,
    pub size_bytes: u64,
}

/// Record the unique unpacked directory created for the current training run.
pub(crate) fn record_last_unpacked_dir(path: &Path) {
    if let Ok(mut lock) = LAST_UNPACKED_DIR.lock() {
        *lock = Some(path.to_path_buf());
    }
}

/// Clear any previously recorded unpacked directory before starting a new run.
pub(crate) fn clear_last_unpacked_dir() {
    if let Ok(mut lock) = LAST_UNPACKED_DIR.lock() {
        *lock = None;
    }
}

/// Return the root directory where unpacked dataset archives are stored:
/// - Windows: `%LOCALAPPDATA%\LS-Trainer\unpacked`
/// - macOS: `~/Library/Application Support/LS-Trainer/unpacked`
/// - Linux/other Unix: `$XDG_DATA_HOME/LS-Trainer/unpacked` or `~/.local/share/LS-Trainer/unpacked`
pub(crate) fn unpacked_store_root() -> Result<PathBuf, String> {
    #[cfg(target_os = "windows")]
    let base = {
        std::env::var_os("LOCALAPPDATA")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("USERPROFILE")
                    .filter(|v| !v.is_empty())
                    .map(|p| PathBuf::from(p).join("AppData").join("Local"))
            })
    };

    #[cfg(target_os = "macos")]
    let base = {
        std::env::var_os("HOME")
            .filter(|v| !v.is_empty())
            .map(|p| PathBuf::from(p).join("Library").join("Application Support"))
    };

    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    let base = {
        std::env::var_os("XDG_DATA_HOME")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME")
                    .filter(|v| !v.is_empty())
                    .map(|p| PathBuf::from(p).join(".local").join("share"))
            })
    };

    base.map(|root| root.join("LS-Trainer").join("unpacked"))
        .ok_or_else(|| {
            "Could not determine local app-data directory for unpacked datasets".to_string()
        })
}

/// Format a `SystemTime` as UTC `YYYY-MM-DD_HHMMSS` without external date crates.
pub(crate) fn format_timestamp_utc(now: SystemTime) -> String {
    let secs = now
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = (secs / 86_400) as i64;
    let secs_of_day = secs % 86_400;
    let hour = secs_of_day / 3_600;
    let minute = (secs_of_day % 3_600) / 60;
    let second = secs_of_day % 60;

    // Howard Hinnant's civil_from_days algorithm (days since 1970-01-01).
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = y + if m <= 2 { 1 } else { 0 };

    format!("{year:04}-{m:02}-{d:02}_{hour:02}{minute:02}{second:02}")
}

/// Derive a short, filesystem-safe name from an archive filename.
pub(crate) fn short_safe_dataset_name(archive_path: &Path) -> String {
    let raw_name = archive_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("dataset");
    let lower = raw_name.to_ascii_lowercase();
    let stem = if lower.ends_with(".tar.gz") {
        &raw_name[..raw_name.len() - ".tar.gz".len()]
    } else if lower.ends_with(".tgz") {
        &raw_name[..raw_name.len() - ".tgz".len()]
    } else if lower.ends_with(".tar") {
        &raw_name[..raw_name.len() - ".tar".len()]
    } else if lower.ends_with(".zip") {
        &raw_name[..raw_name.len() - ".zip".len()]
    } else {
        raw_name
    };

    let mut sanitized = String::with_capacity(stem.len().min(32));
    let mut prev_underscore = false;
    for ch in stem.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            if ch == '_' {
                if !prev_underscore && !sanitized.is_empty() {
                    sanitized.push('_');
                    prev_underscore = true;
                }
            } else {
                sanitized.push(ch);
                prev_underscore = false;
            }
        } else if !prev_underscore && !sanitized.is_empty() {
            sanitized.push('_');
            prev_underscore = true;
        }
        if sanitized.len() >= 32 {
            break;
        }
    }

    let trimmed = sanitized.trim_matches(|c| c == '_' || c == '-');
    if trimmed.is_empty() {
        "dataset".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Create a unique subfolder under `unpacked_store_root()` for this run.
pub(crate) fn create_unpacked_run_dir(archive_path: &Path) -> Result<PathBuf, String> {
    let store_root = unpacked_store_root()?;
    create_unpacked_run_dir_in(&store_root, archive_path)
}

/// Create a unique subfolder under `store_root` using atomic `create_dir` so
/// concurrent or repeated runs never collide.
pub(crate) fn create_unpacked_run_dir_in(
    store_root: &Path,
    archive_path: &Path,
) -> Result<PathBuf, String> {
    std::fs::create_dir_all(store_root).map_err(|e| {
        format!(
            "Failed to create unpacked dataset store '{}': {e}",
            store_root.display()
        )
    })?;

    let timestamp = format_timestamp_utc(SystemTime::now());
    let short_name = short_safe_dataset_name(archive_path);
    let base_name = format!("{timestamp}_{short_name}");

    for attempt in 0..1000u32 {
        let folder_name = if attempt == 0 {
            base_name.clone()
        } else {
            format!("{base_name}_{attempt}")
        };
        let candidate = store_root.join(folder_name);
        match std::fs::create_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => {
                return Err(format!(
                    "Failed to create unpacked dataset folder '{}': {e}",
                    candidate.display()
                ));
            }
        }
    }

    Err(format!(
        "Failed to allocate unique unpacked dataset folder under '{}'",
        store_root.display()
    ))
}

fn is_image_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "jpg" | "jpeg" | "png" | "bmp" | "webp" | "tif" | "tiff"
            )
        })
        .unwrap_or(false)
}

pub(crate) fn inspect_unpacked_dir(dir: &Path) -> UnpackedDatasetInfo {
    let mut image_count = 0u64;
    let mut size_bytes = 0u64;

    for entry in walkdir::WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
        if entry.file_type().is_file() {
            if let Ok(meta) = entry.metadata() {
                size_bytes = size_bytes.saturating_add(meta.len());
            }
            if is_image_file(entry.path()) {
                image_count += 1;
            }
        }
    }

    UnpackedDatasetInfo {
        path: to_display_path(dir),
        image_count,
        size_bytes,
    }
}

/// Remove `target` only if it is a direct child directory inside `store_root`.
/// Never deletes `store_root` itself or anything outside it.
pub(crate) fn delete_unpacked_dataset_in(store_root: &Path, target: &Path) -> Result<(), String> {
    let canon_store = std::fs::canonicalize(store_root).map_err(|e| {
        format!(
            "Unpacked dataset store '{}' is not accessible: {e}",
            store_root.display()
        )
    })?;

    let meta = std::fs::symlink_metadata(target).map_err(|e| {
        format!(
            "Unpacked dataset folder '{}' could not be accessed: {e}",
            target.display()
        )
    })?;
    if meta.file_type().is_symlink() {
        return Err(format!(
            "Refusing to delete symlink '{}'",
            target.display()
        ));
    }

    let canon_target = std::fs::canonicalize(target).map_err(|e| {
        format!(
            "Unpacked dataset folder '{}' could not be resolved: {e}",
            target.display()
        )
    })?;

    if canon_target == canon_store || canon_target.parent() != Some(canon_store.as_path()) {
        return Err(format!(
            "Refusing to delete '{}': path is outside the unpacked dataset store '{}'",
            target.display(),
            store_root.display()
        ));
    }

    if !canon_target.is_dir() {
        return Err(format!(
            "Refusing to delete '{}': not a directory",
            target.display()
        ));
    }

    std::fs::remove_dir_all(&canon_target).map_err(|e| {
        format!(
            "Failed to delete unpacked dataset folder '{}': {e}",
            canon_target.display()
        )
    })
}

#[tauri::command]
pub async fn take_last_unpacked_dataset() -> Option<UnpackedDatasetInfo> {
    let dir = LAST_UNPACKED_DIR
        .lock()
        .ok()
        .and_then(|mut lock| lock.take())?;
    if !dir.is_dir() {
        return None;
    }
    tokio::task::spawn_blocking(move || inspect_unpacked_dir(&dir))
        .await
        .ok()
}

#[tauri::command]
pub async fn delete_unpacked_dataset(path: String) -> Result<(), String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err("Unpacked dataset path is empty".to_string());
    }
    let store_root = unpacked_store_root()?;
    let target = PathBuf::from(trimmed);
    tokio::task::spawn_blocking(move || delete_unpacked_dataset_in(&store_root, &target))
        .await
        .map_err(|e| format!("Failed to delete unpacked dataset: {e}"))?
}
